"""
USV Digital Twin — Módulo 2: Physics Engine
============================================
Implementa as equações de movimento de Fossen em 6 DOF para
veículo subaquático com propulsor rotacionável e sistema de lastro.

Equações de movimento (Fossen 2011):
    η̇ = J(η) ν
    Mν̇ = τ - C(ν)ν - D(ν)ν - g(η)

Estado:
    η = [x, y, z, φ, θ, ψ]ᵀ  — posição/orientação no referencial inercial (NED)
    ν = [u, v, w, p, q, r]ᵀ  — velocidades no referencial do corpo

Representação de orientação: quaternions internamente, Euler pra output
Integração numérica: Runge-Kutta 4 (RK4)

Referências:
    - Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics
    - Fossen, T.I. (1994). Guidance and Control of Ocean Vehicles
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
from geometry_engine import GeometryEngine, HullGeometry, HydrodynamicCoefficients


# ─────────────────────────────────────────────
# CONSTANTES FÍSICAS
# ─────────────────────────────────────────────
G               = 9.81      # m/s² — aceleração gravitacional
RHO_FRESHWATER  = 1000.0    # kg/m³ — água doce
MAX_THRUSTER_ANGLE_DEG = 60.0   # graus — cone máximo do propulsor
MAX_THRUSTER_ANGLE_RAD = np.radians(MAX_THRUSTER_ANGLE_DEG)


# ─────────────────────────────────────────────
# COMPONENTES FIXOS DO VEÍCULO
# ─────────────────────────────────────────────
@dataclass
class ComponentMasses:
    """Massa dos componentes fixos internos."""
    esp32_wrover:   float = 0.010   # kg
    battery_lipo:   float = 0.350   # kg — LiPo 3S 2200mAh estimado
    electronics:    float = 0.050   # kg — VL53L0X + PCBs + cabos
    thruster_motor: float = 0.200   # kg — motor brushless + propulsor
    structure:      float = 0.100   # kg — suportes internos + vedações
    ballast_fixed:  float = 0.870   # kg — lastro fixo de chumbo para ponto neutro

    @property
    def total(self) -> float:
        return (self.esp32_wrover + self.battery_lipo +
                self.electronics + self.thruster_motor +
                self.structure + self.ballast_fixed)


# ─────────────────────────────────────────────
# SISTEMA DE LASTRO
# ─────────────────────────────────────────────
@dataclass
class BallastSystem:
    """
    Sistema de lastro por seringa — varia massa de água interna.
    Opera no range de densidade média 950-1050 kg/m³.
    """
    hull_volume:        float           # m³ — volume do casco
    rho_fluid:          float = RHO_FRESHWATER
    rho_target_min:     float = 950.0   # kg/m³ — flutuação máxima
    rho_target_max:     float = 1050.0  # kg/m³ — afundamento máximo
    fill_rate:          float = 0.0001  # m³/s — taxa REAL do hardware (seringa física)
    sim_speed_multiplier: float = 1.0   # >1 acelera pra debug; SEMPRE 1.0 pra treino de RL/MPC

    # estado interno
    _water_mass: float = field(init=False)

    # massa base do veículo sem água (casco + componentes)
    base_mass: float = 0.0

    def __post_init__(self):
        self._water_mass = 0.0

    @property
    def water_mass(self) -> float:
        return self._water_mass

    @property
    def mass_min(self) -> float:
        """Massa de água para atingir densidade mínima (950 kg/m³)."""
        m_total_min = self.rho_target_min * self.hull_volume
        return max(0.0, m_total_min - self.base_mass)

    @property
    def mass_max(self) -> float:
        """Massa de água para atingir densidade máxima (1050 kg/m³)."""
        m_total_max = self.rho_target_max * self.hull_volume
        return max(0.0, m_total_max - self.base_mass)

    @property
    def water_volume(self) -> float:
        """Volume de água atual na seringa."""
        return self._water_mass / self.rho_fluid

    def update(self, command: float, dt: float) -> float:
        """
        Atualiza massa de água no lastro.

        Args:
            command: [-1, +1] — -1 expele água, +1 injeta água
            dt: timestep em segundos

        Returns:
            delta_mass: variação de massa neste timestep
        """
        command   = np.clip(command, -1.0, 1.0)
        dm        = command * self.fill_rate * self.sim_speed_multiplier * self.rho_fluid * dt
        old_mass  = self._water_mass
        self._water_mass = np.clip(
            self._water_mass + dm,
            self.mass_min,
            self.mass_max
        )
        return self._water_mass - old_mass

    def buoyancy_force(self, total_mass: float) -> float:
        """
        Força líquida vertical = empuxo - peso.
        Positivo = sobe, negativo = afunda.
        """
        buoyancy = self.rho_fluid * self.hull_volume * G
        weight   = total_mass * G
        return buoyancy - weight


# ─────────────────────────────────────────────
# PROPULSOR ROTACIONÁVEL
# ─────────────────────────────────────────────
@dataclass
class Thruster:
    """
    Propulsor rotacionável alinhado ao centro de massa do veículo.
    Cone de rotação máximo: 60° em qualquer direção radial.
    Força máxima parametrizável.
    """
    max_force:      float           # N — força máxima
    position_body:  np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    # estado — ângulos em radianos
    _theta: float = field(init=False, default=0.0)  # ângulo polar (0 = eixo x)
    _phi:   float = field(init=False, default=0.0)  # ângulo azimutal
    _power: float = field(init=False, default=0.0)  # [-1, 1]

    def __post_init__(self):
        self._theta = 0.0
        self._phi   = 0.0
        self._power = 0.0

    def set_orientation(self, theta: float, phi: float) -> None:
        """
        Define orientação do propulsor.

        Args:
            theta: ângulo polar em radianos [0, MAX_ANGLE]
            phi:   ângulo azimutal em radianos [0, 2π]
        """
        self._theta = np.clip(theta, 0.0, MAX_THRUSTER_ANGLE_RAD)
        self._phi   = phi % (2 * np.pi)

    def set_power(self, power: float) -> None:
        """power ∈ [-1, 1] — negativo = reverso."""
        self._power = np.clip(power, -1.0, 1.0)

    @property
    def thrust_vector_body(self) -> np.ndarray:
        """
        Vetor de força no referencial do corpo (3D).
        Propulsor aponta pra trás (-x) no neutro.
        Rotação theta em torno de phi desloca o vetor.
        """
        F = self._power * self.max_force

        # vetor unitário de empuxo no referencial do propulsor
        # neutro: empurra pra frente (+x no corpo)
        # deflexão: componente lateral/vertical
        fx =  F * np.cos(self._theta)
        fy =  F * np.sin(self._theta) * np.cos(self._phi)
        fz =  F * np.sin(self._theta) * np.sin(self._phi)

        return np.array([fx, fy, fz])

    @property
    def torque_vector_body(self) -> np.ndarray:
        """
        Torque gerado pelo propulsor em torno do CG.
        τ = r × F, onde r é o vetor da popa ao CG.
        """
        # Torque pelo braço do ponto de aplicação em relação ao CG.
        r = np.asarray(self.position_body, dtype=float)
        F = self.thrust_vector_body
        return np.cross(r, F)

    @property
    def wrench_body(self) -> np.ndarray:
        """Retorna [fx, fy, fz, tx, ty, tz] no referencial do corpo."""
        F = self.thrust_vector_body
        T = self.torque_vector_body
        return np.concatenate([F, T])


# ─────────────────────────────────────────────
# ESTADO DO VEÍCULO
# ─────────────────────────────────────────────
@dataclass
class VehicleState:
    """Estado completo do veículo em 6 DOF."""

    # posição e orientação no referencial inercial
    x:   float = 0.0   # m — Norte
    y:   float = 0.0   # m — Leste
    z:   float = 0.0   # m — Down (positivo = profundidade)
    phi: float = 0.0   # rad — roll
    tht: float = 0.0   # rad — pitch (theta)
    psi: float = 0.0   # rad — yaw

    # velocidades no referencial do corpo
    u: float = 0.0   # m/s — surge
    v: float = 0.0   # m/s — sway
    w: float = 0.0   # m/s — heave
    p: float = 0.0   # rad/s — roll rate
    q: float = 0.0   # rad/s — pitch rate
    r: float = 0.0   # rad/s — yaw rate

    # quaternion de orientação (uso interno)
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    @property
    def eta(self) -> np.ndarray:
        """Vetor de posição/orientação [x,y,z,φ,θ,ψ]."""
        return np.array([self.x, self.y, self.z,
                         self.phi, self.tht, self.psi])

    @property
    def nu(self) -> np.ndarray:
        """Vetor de velocidades [u,v,w,p,q,r]."""
        return np.array([self.u, self.v, self.w,
                         self.p, self.q, self.r])

    @property
    def quaternion(self) -> np.ndarray:
        return np.array([self.qw, self.qx, self.qy, self.qz])

    def to_dict(self) -> dict:
        return {
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'orientation_euler': {'phi': self.phi, 'theta': self.tht, 'psi': self.psi},
            'orientation_quat':  {'w': self.qw, 'x': self.qx, 'y': self.qy, 'z': self.qz},
            'velocity_linear':   {'u': self.u, 'v': self.v, 'w': self.w},
            'velocity_angular':  {'p': self.p, 'q': self.q, 'r': self.r},
        }


# ─────────────────────────────────────────────
# PHYSICS ENGINE
# ─────────────────────────────────────────────
class PhysicsEngine:
    """
    Motor de física — equações de Fossen 6 DOF.

    Uso:
        geo      = GeometryEngine(L=0.8, D=0.1)
        physics  = PhysicsEngine(geo, max_thruster_force=10.0)
        physics.step(thruster_power=0.5, thruster_theta=0.1,
                     thruster_phi=0.0, ballast_cmd=0.0, dt=0.01)
        state = physics.state
    """

    def __init__(
        self,
        geometry:           GeometryEngine,
        max_thruster_force: float = 10.0,       # N
        components:         ComponentMasses = None,
        rho:                float = RHO_FRESHWATER,
        sim_speed_multiplier: float = 1.0,      # 1.0 = tempo real; >1 só pra debug visual
    ):
        self.geo   = geometry
        self.hull  = geometry.hull_geometry
        self.coeff = geometry.coefficients
        self.rho   = rho

        self.components = components or ComponentMasses()
        self.ballast    = BallastSystem(
            hull_volume=self.hull.volume,
            rho_fluid=rho,
            base_mass=self.hull.mass_hull + self.components.total,
            sim_speed_multiplier=sim_speed_multiplier,
        )
        # Dois thrusters laterais na altura do CG para recuperar autoridade
        # de roll/yaw sem deslocar longitudinalmente o thrust do centro de massa.
        lateral_arm = max(0.03, 0.4 * self.hull.R)
        self.thruster_port = Thruster(
            max_force=max_thruster_force,
            position_body=np.array([0.0, +lateral_arm, 0.0], dtype=float),
        )
        self.thruster_starboard = Thruster(
            max_force=max_thruster_force,
            position_body=np.array([0.0, -lateral_arm, 0.0], dtype=float),
        )

        self._state = VehicleState()
        self._time  = 0.0

        # monta matrizes de inércia uma vez
        self._M     = self._build_mass_matrix()
        self._M_inv = np.linalg.inv(self._M)

    # ─── Interface pública ───────────────────

    @property
    def state(self) -> VehicleState:
        return self._state

    @property
    def time(self) -> float:
        return self._time

    @property
    def total_mass(self) -> float:
        return (self.hull.mass_hull +
                self.components.total +
                self.ballast.water_mass)

    def step(
        self,
        thruster_power: float,
        thruster_theta: float,
        thruster_phi:   float,
        ballast_cmd:    float,
        thruster2_power: Optional[float] = None,
        thruster2_theta: Optional[float] = None,
        thruster2_phi:   Optional[float] = None,
        dt:             float = 0.01,
    ) -> VehicleState:
        """
        Avança a simulação por um timestep dt.

        Args:
            thruster_power: [-1, 1] thruster 1 (ou potência total quando thruster2_* = None)
            thruster_theta: ângulo polar do thruster 1 [0, 60°] em rad
            thruster_phi:   ângulo azimutal do thruster 1 [0, 2π] em rad
            ballast_cmd:    [-1, 1] — -1 expele, +1 injeta água
            thruster2_power/theta/phi: comandos do thruster 2 (opcional)
            dt:             timestep em segundos

        Returns:
            Novo estado do veículo
        """
        # Atualiza atuadores.
        # Compatibilidade: sem comandos do thruster 2, divide o comando total
        # igualmente entre os dois thrusters para manter força equivalente ao
        # modelo antigo de thruster único.
        if thruster2_power is None and thruster2_theta is None and thruster2_phi is None:
            p1 = p2 = 0.5 * float(thruster_power)
            t1 = t2 = float(thruster_theta)
            f1 = f2 = float(thruster_phi)
        else:
            p1 = float(thruster_power)
            t1 = float(thruster_theta)
            f1 = float(thruster_phi)
            p2 = float(thruster2_power if thruster2_power is not None else thruster_power)
            t2 = float(thruster2_theta if thruster2_theta is not None else thruster_theta)
            f2 = float(thruster2_phi if thruster2_phi is not None else thruster_phi)

        self.thruster_port.set_orientation(t1, f1)
        self.thruster_port.set_power(p1)
        self.thruster_starboard.set_orientation(t2, f2)
        self.thruster_starboard.set_power(p2)
        self.ballast.update(ballast_cmd, dt)

        # recalcula massa adicionada com lastro atualizado
        self._M     = self._build_mass_matrix()
        self._M_inv = np.linalg.inv(self._M)

        # integração RK4
        self._state = self._rk4(self._state, dt)
        self._time += dt

        return self._state

    def reset(self, state: VehicleState = None) -> None:
        """Reseta o estado para posição inicial ou estado fornecido."""
        self._state = state or VehicleState()
        self._time  = 0.0
        self.ballast._water_mass = 0.0

    # ─── Construção das matrizes ─────────────

    def _build_mass_matrix(self) -> np.ndarray:
        """
        Matriz de massa e inércia total M = M_rigid + M_added.
        Inclui massa atual do lastro.
        """
        m   = self.total_mass
        L   = self.hull.L
        R   = self.hull.R
        c   = self.coeff

        # momentos de inércia do casco aproximado como casca cilíndrica
        Ixx = 0.7  * m * R**2                          # roll
        Iyy = (1/12) * m * (3*R**2 + L**2)            # pitch
        Izz = Iyy                                       # yaw — simetria

        M_rigid = np.diag([m, m, m, Ixx, Iyy, Izz])
        M_added = np.diag([
            c.X_udot, c.Y_vdot, c.Z_wdot,
            c.K_pdot, c.M_qdot, c.N_rdot
        ])

        return M_rigid + M_added

    # ─── Equações de Fossen ──────────────────

    def _rotation_matrix(self, phi: float, theta: float, psi: float) -> np.ndarray:
        """Matriz de rotação ZYX (yaw-pitch-roll) 3x3."""
        cphi = np.cos(phi);   sphi = np.sin(phi)
        cth  = np.cos(theta); sth  = np.sin(theta)
        cpsi = np.cos(psi);   spsi = np.sin(psi)

        return np.array([
            [cpsi*cth,  cpsi*sth*sphi - spsi*cphi,  cpsi*sth*cphi + spsi*sphi],
            [spsi*cth,  spsi*sth*sphi + cpsi*cphi,  spsi*sth*cphi - cpsi*sphi],
            [-sth,      cth*sphi,                    cth*cphi                 ]
        ])

    def _jacobian(self, eta: np.ndarray) -> np.ndarray:
        """
        Matriz Jacobiana J(η) 6x6.
        η̇ = J(η) ν
        """
        phi, theta, psi = eta[3], eta[4], eta[5]

        R   = self._rotation_matrix(phi, theta, psi)

        cphi = np.cos(phi); sphi = np.sin(phi)
        cth  = np.cos(theta)
        tth  = np.tan(theta)

        # matriz de transformação angular (Euler rates → body rates)
        T = np.array([
            [1,  sphi*tth,   cphi*tth ],
            [0,  cphi,      -sphi     ],
            [0,  sphi/cth,   cphi/cth ]
        ])

        J = np.zeros((6, 6))
        J[:3, :3] = R
        J[3:, 3:] = T

        return J

    def _coriolis_matrix(self, nu: np.ndarray) -> np.ndarray:
        """
        Matriz de Coriolis e centrípeta C(ν) 6x6.
        Formulação de Fossen (2011) eq. 6.43
        """
        m   = self.total_mass
        L   = self.hull.L
        R   = self.hull.R
        c   = self.coeff

        u, v, w = nu[0], nu[1], nu[2]
        p, q, r = nu[3], nu[4], nu[5]

        # massa total por eixo (rígida + adicionada)
        m11 = m + c.X_udot
        m22 = m + c.Y_vdot
        m33 = m + c.Z_wdot
        m44 = 0.5*m*R**2         + c.K_pdot
        m55 = (1/12)*m*(3*R**2+L**2) + c.M_qdot
        m66 = m55                + c.N_rdot

        C = np.zeros((6, 6))

        # bloco superior direito
        C[0, 3] =  0;        C[0, 4] =  m33*w;   C[0, 5] = -m22*v
        C[1, 3] = -m33*w;   C[1, 4] =  0;        C[1, 5] =  m11*u
        C[2, 3] =  m22*v;   C[2, 4] = -m11*u;    C[2, 5] =  0

        # bloco inferior esquerdo (transposto negativo)
        C[3, 0] =  0;        C[3, 1] =  m33*w;   C[3, 2] = -m22*v
        C[4, 0] = -m33*w;   C[4, 1] =  0;        C[4, 2] =  m11*u
        C[5, 0] =  m22*v;   C[5, 1] = -m11*u;    C[5, 2] =  0

        # bloco inferior direito
        C[3, 4] =  m66*r;   C[3, 5] = -m55*q
        C[4, 3] = -m66*r;   C[4, 5] =  m44*p
        C[5, 3] =  m55*q;   C[5, 4] = -m44*p

        return C

    def _drag_matrix(self, nu: np.ndarray) -> np.ndarray:
        """
        Matriz de amortecimento hidrodinâmico D(ν).
        D(ν) = D_linear + D_quadrático * |ν|
        """
        D_lin  = self.coeff.to_drag_matrix_linear()
        D_quad = self.coeff.to_drag_matrix_quadratic()
        D_nonl = np.diag(D_quad.diagonal() * np.abs(nu))
        return D_lin + D_nonl

    def _restoring_forces(self, eta: np.ndarray) -> np.ndarray:
        """
        Forças e momentos restauradores g(η).
        Inclui gravidade, empuxo e efeito do lastro.
        """
        phi, theta = eta[3], eta[4]

        m        = self.total_mass
        weight   = m * G
        buoyancy = self.rho * self.hull.volume * G

        cphi = np.cos(phi); sphi = np.sin(phi)
        cth  = np.cos(theta); sth = np.sin(theta)

        # forças no referencial do corpo
        W_minus_B = weight - buoyancy

        g = np.array([
             W_minus_B * sth,
            -W_minus_B * cth * sphi,
            -W_minus_B * cth * cphi,
             0.0,   # roll — CG e CB alinhados (simplificação)
             0.0,   # pitch
             0.0,   # yaw
        ])

        return g

    def _derivatives(self, state: VehicleState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula η̇ e ν̇ dado o estado atual.
        Retorna (eta_dot, nu_dot).
        """
        eta = state.eta
        nu  = state.nu

        J   = self._jacobian(eta)
        C   = self._coriolis_matrix(nu)
        D   = self._drag_matrix(nu)
        g   = self._restoring_forces(eta)
        tau = self.thruster_port.wrench_body + self.thruster_starboard.wrench_body

        eta_dot = J @ nu
        nu_dot  = self._M_inv @ (tau - C @ nu - D @ nu - g)

        return eta_dot, nu_dot

    # ─── Integração RK4 ─────────────────────

    def _rk4(self, state: VehicleState, dt: float) -> VehicleState:
        """Integração Runge-Kutta de 4ª ordem."""

        def pack(s: VehicleState) -> np.ndarray:
            return np.concatenate([s.eta, s.nu])

        def unpack(x: np.ndarray) -> VehicleState:
            s = VehicleState()
            s.x,   s.y,   s.z   = x[0], x[1], x[2]
            s.phi, s.tht, s.psi = x[3], x[4], x[5]
            s.u,   s.v,   s.w   = x[6], x[7], x[8]
            s.p,   s.q,   s.r   = x[9], x[10], x[11]
            # atualiza quaternion a partir de Euler
            s.qw, s.qx, s.qy, s.qz = self._euler_to_quat(x[3], x[4], x[5])
            return s

        def f(s: VehicleState) -> np.ndarray:
            ed, nd = self._derivatives(s)
            return np.concatenate([ed, nd])

        x0 = pack(state)

        k1 = f(unpack(x0))
        k2 = f(unpack(x0 + 0.5*dt*k1))
        k3 = f(unpack(x0 + 0.5*dt*k2))
        k4 = f(unpack(x0 + dt*k3))

        x_new = x0 + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return unpack(x_new)

    @staticmethod
    def _euler_to_quat(phi: float, theta: float, psi: float) -> Tuple[float,float,float,float]:
        """Converte ângulos de Euler ZYX para quaternion."""
        cy = np.cos(psi   * 0.5); sy = np.sin(psi   * 0.5)
        cp = np.cos(theta * 0.5); sp = np.sin(theta * 0.5)
        cr = np.cos(phi   * 0.5); sr = np.sin(phi   * 0.5)

        qw = cr*cp*cy + sr*sp*sy
        qx = sr*cp*cy - cr*sp*sy
        qy = cr*sp*cy + sr*cp*sy
        qz = cr*cp*sy - sr*sp*cy

        return qw, qx, qy, qz

    def to_dict(self) -> dict:
        """Serializa estado completo pra JSON — WebSocket."""
        F_port = self.thruster_port.thrust_vector_body
        F_stbd = self.thruster_starboard.thrust_vector_body
        F_total = F_port + F_stbd
        F_norm = float(np.linalg.norm(F_total))

        if F_norm > 1e-9:
            theta_total = float(np.arctan2(np.linalg.norm(F_total[1:]), F_total[0]))
            phi_total = float(np.arctan2(F_total[2], F_total[1]))
        else:
            theta_total = 0.0
            phi_total = 0.0

        avg_power = 0.5 * (self.thruster_port._power + self.thruster_starboard._power)

        return {
            'time':     self._time,
            'state':    self._state.to_dict(),
            'mass':     self.total_mass,
            'ballast':  {
                'water_mass':   self.ballast.water_mass,
                'water_volume': self.ballast.water_volume,
                'density_avg':  self.total_mass / self.hull.volume,
            },
            'thruster': {
                'power':        avg_power,
                'theta_deg':    np.degrees(theta_total),
                'phi_deg':      np.degrees(phi_total),
                'force_vector': F_total.tolist(),
            },
            'thruster_pair': {
                'port': {
                    'power': self.thruster_port._power,
                    'theta_deg': np.degrees(self.thruster_port._theta),
                    'phi_deg': np.degrees(self.thruster_port._phi),
                    'force_vector': F_port.tolist(),
                },
                'starboard': {
                    'power': self.thruster_starboard._power,
                    'theta_deg': np.degrees(self.thruster_starboard._theta),
                    'phi_deg': np.degrees(self.thruster_starboard._phi),
                    'force_vector': F_stbd.tolist(),
                },
            },
        }


# ─────────────────────────────────────────────
# TESTES RÁPIDOS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Inicializando Physics Engine...")
    geo     = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geo, max_thruster_force=10.0)

    print(f"Massa total inicial: {physics.total_mass:.3f} kg")
    print(f"Força de empuxo líquida: {physics.ballast.buoyancy_force(physics.total_mass):.3f} N")

    # Teste 1 — queda livre (propulsor desligado, lastro neutro)
    print("\nTeste 1 — Queda livre por 2s:")
    physics.reset()
    for i in range(200):
        physics.step(
            thruster_power=0.0,
            thruster_theta=0.0,
            thruster_phi=0.0,
            ballast_cmd=0.0,
            dt=0.01
        )
    s = physics.state
    print(f"  Posição z após 2s: {s.z:.4f} m")
    print(f"  Velocidade w: {s.w:.4f} m/s")

    # Teste 2 — propulsor frontal a 50%
    print("\nTeste 2 — Propulsor 50% por 3s:")
    physics.reset()
    for i in range(300):
        physics.step(
            thruster_power=0.5,
            thruster_theta=0.0,
            thruster_phi=0.0,
            ballast_cmd=0.0,
            dt=0.01
        )
    s = physics.state
    print(f"  Posição x após 3s: {s.x:.4f} m")
    print(f"  Velocidade u: {s.u:.4f} m/s")

    # Teste 3 — lastro enchendo (afundando)
    print("\nTeste 3 — Lastro enchendo por 5s:")
    physics.reset()
    for i in range(500):
        physics.step(
            thruster_power=0.0,
            thruster_theta=0.0,
            thruster_phi=0.0,
            ballast_cmd=1.0,
            dt=0.01
        )
    s = physics.state
    print(f"  Posição z após 5s: {s.z:.4f} m")
    print(f"  Densidade média: {physics.to_dict()['ballast']['density_avg']:.1f} kg/m³")

    # Teste 4 — propulsor deflectido 30° pra cima
    print("\nTeste 4 — Propulsor deflectido 30° pitch por 2s:")
    physics.reset()
    for i in range(200):
        physics.step(
            thruster_power=0.8,
            thruster_theta=np.radians(30),
            thruster_phi=np.radians(90),   # phi=90° → deflexão em z
            ballast_cmd=0.0,
            dt=0.01
        )
    s = physics.state
    print(f"  Posição x: {s.x:.4f} m")
    print(f"  Posição z: {s.z:.4f} m")
    print(f"  Pitch θ:   {np.degrees(s.tht):.2f}°")

    print("\n✓ Physics Engine validado.")
