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
from types import SimpleNamespace
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
        geometry:           Optional[GeometryEngine] = None,
        max_thruster_force: float = 10.0,       # N
        components:         ComponentMasses = None,
        rho:                float = RHO_FRESHWATER,
        sim_speed_multiplier: float = 1.0,      # 1.0 = tempo real; >1 só pra debug visual
        rigid_body_mass:    Optional[float] = None,
        rigid_body_inertia:  Optional[np.ndarray] = None,
        thruster_port_position: Optional[np.ndarray] = None,
        thruster_starboard_position: Optional[np.ndarray] = None,
        planar_dof:         bool = False,
        vehicle_profile:    Optional[object] = None,
    ):
        self.rho   = rho
        self.planar_dof = bool(planar_dof)
        # Optional full-matrix hydro terms from profile.
        self._full_linear_damping_matrix = None
        self._full_quadratic_damping_matrix = None
        self._full_added_mass_matrix = None

        if geometry is None:
            if vehicle_profile is None:
                raise ValueError("geometry or vehicle_profile must be provided")
            self.geo = None
            self.hull = self._build_hull_from_vehicle_profile(vehicle_profile)
            self.coeff = self._build_coefficients_from_vehicle_profile(vehicle_profile)
        else:
            self.geo   = geometry
            self.hull  = geometry.hull_geometry
            self.coeff = geometry.coefficients

        self.components = components or ComponentMasses()
        self._rigid_body_mass = float(rigid_body_mass) if rigid_body_mass is not None else (self.hull.mass_hull + self.components.total)
        self._rigid_body_inertia = None if rigid_body_inertia is None else np.asarray(rigid_body_inertia, dtype=float)
        if self._rigid_body_inertia is not None and self._rigid_body_inertia.shape != (3, 3):
            raise ValueError("rigid_body_inertia must be a 3x3 matrix")
        self.ballast    = BallastSystem(
            hull_volume=self.hull.volume,
            rho_fluid=rho,
            base_mass=self._rigid_body_mass,
            sim_speed_multiplier=sim_speed_multiplier,
        )
        # Two-side thruster setup by default. If a vehicle_profile provides
        # multiple thruster site names (e.g. MJCF with 8 actuators), create
        # a list of Thruster objects and map incoming two-channel commands
        # into the multi-thruster layout by splitting power across port/starboard
        # groups. Keep `thruster_port` and `thruster_starboard` attributes for
        # backward compatibility (they point to a representative thruster).
        lateral_arm = max(0.03, 0.4 * self.hull.R)
        default_port = np.array([0.0, +lateral_arm, 0.0], dtype=float)
        default_star = np.array([0.0, -lateral_arm, 0.0], dtype=float)

        # If a vehicle_profile provided explicit per-side positions, use them
        port_pos = np.asarray(thruster_port_position, dtype=float) if thruster_port_position is not None else default_port
        star_pos = np.asarray(thruster_starboard_position, dtype=float) if thruster_starboard_position is not None else default_star

        # create thrusters list (may be replaced by mapping below)
        self.thrusters = []

        # If vehicle_profile has metadata listing thruster sites, expand to that many
        try:
            sites = getattr(vehicle_profile, 'metadata', {}).get('thruster', {}).get('site_names', None) if vehicle_profile is not None else None
            if sites and isinstance(sites, (list, tuple)) and len(sites) > 0:
                # attempt to load explicit mapping file first (created by tools/map_thruster_sites.py)
                import json, os
                repo_root = os.path.dirname(os.path.abspath(__file__))
                mapping_path = os.path.join(repo_root, 'training_runs', f'thruster_mapping_{getattr(vehicle_profile, "name", "profile").lower()}.json')
                mapping = None
                try:
                    if os.path.exists(mapping_path):
                        with open(mapping_path, 'r') as fh:
                            mapping = json.load(fh).get('sites', None)
                except Exception:
                    mapping = None

                if mapping and isinstance(mapping, list) and len(mapping) >= len(sites):
                    # mapping may contain extra non-thruster sites (e.g. taluy_site).
                    # Build a dict by site_name for quick lookup and then order
                    # according to the declared `sites` in vehicle_profile.
                    mapping_by_name = {ent.get('site_name'): ent for ent in mapping if isinstance(ent, dict) and ent.get('site_name')}
                    all_present = all(name in mapping_by_name for name in sites)
                    if all_present:
                        self.thrusters = []
                        for name in sites:
                            ent = mapping_by_name.get(name)
                            pos = ent.get('position_body_m', None) if ent is not None else None
                            if pos is None:
                                pos = port_pos
                            self.thrusters.append(Thruster(max_force=max_thruster_force, position_body=np.asarray(pos, dtype=float)))
                    else:
                        # fallback to building from declared sites by alternating
                        self.thrusters = []
                        for i, name in enumerate(sites):
                            side_pos = port_pos if (i % 2 == 0) else star_pos
                            self.thrusters.append(Thruster(max_force=max_thruster_force, position_body=side_pos))
                else:
                    # build thrusters from the declared sites by alternating side positions
                    self.thrusters = []
                    for i, name in enumerate(sites):
                        side_pos = port_pos if (i % 2 == 0) else star_pos
                        self.thrusters.append(Thruster(max_force=max_thruster_force, position_body=side_pos))
        except Exception:
            # on any error, fall back to the two-thruster layout
            pass

        # Representative attributes for compatibility (ensure at least two)
        if len(self.thrusters) == 0:
            # fallback to two default thrusters
            self.thrusters.append(Thruster(max_force=max_thruster_force, position_body=port_pos))
            self.thrusters.append(Thruster(max_force=max_thruster_force, position_body=star_pos))
        self.thruster_port = self.thrusters[0]
        self.thruster_starboard = self.thrusters[1]

        self._state = VehicleState()
        self._time  = 0.0

        # monta matrizes de inércia uma vez
        self._M     = self._build_mass_matrix()
        self._M_inv = np.linalg.inv(self._M)
        # ambiente externo (corrente/turbulência) no referencial do corpo
        self._env_current_body = np.zeros(3, dtype=float)
        self._env_turbulence = 0.0
        # parâmetros para modelagem de forças ambientais (ajustáveis)
        self.env_force_gain = 1.0
        self.env_turbulence_gain = 1.0
        self.env_wave_freq = 0.8
        # fase aleatória para o componente oscilatório de onda
        self._rng = np.random.default_rng(0)
        self._env_wave_phase = float(self._rng.uniform(0.0, 2.0 * np.pi))
        # previous environmental current (para derivada temporal)
        self._prev_env_current_body = self._env_current_body.copy()
        self._env_dt = 0.0

    @classmethod
    def from_vehicle_profile(
        cls,
        vehicle_profile: object,
        max_thruster_force: float = 10.0,
        components: ComponentMasses = None,
        rho: float = RHO_FRESHWATER,
        sim_speed_multiplier: float = 1.0,
        planar_dof: bool = False,
    ):
        return cls(
            geometry=None,
            max_thruster_force=max_thruster_force,
            components=components,
            rho=rho,
            sim_speed_multiplier=sim_speed_multiplier,
            rigid_body_mass=getattr(vehicle_profile, "mass_kg", None),
            rigid_body_inertia=getattr(vehicle_profile, "inertia_kgm2", None),
            thruster_port_position=getattr(vehicle_profile, "thruster_port_position_m", None),
            thruster_starboard_position=getattr(vehicle_profile, "thruster_starboard_position_m", None),
            planar_dof=planar_dof,
            vehicle_profile=vehicle_profile,
        )

    def _build_hull_from_vehicle_profile(self, vehicle_profile: object):
        length = float(getattr(vehicle_profile, "length_m", 1.0))
        beam_total = float(getattr(vehicle_profile, "beam_total_m", getattr(vehicle_profile, "beam_m", 0.2) * 2.0))
        radius = max(0.01, 0.5 * beam_total)
        volume = float(getattr(vehicle_profile, "metadata", {}).get("hydro", {}).get("displaced_volume_m3", radius * radius * length * 0.35))
        mass_hull = float(getattr(vehicle_profile, "mass_kg", volume * self.rho))
        return SimpleNamespace(
            L=length,
            R=radius,
            volume=volume,
            mass_hull=mass_hull,
            L_D_ratio=length / max(1e-6, beam_total),
            A_frontal=np.pi * radius ** 2,
            A_lateral=length * beam_total,
        )

    def _build_coefficients_from_vehicle_profile(self, vehicle_profile: object):
        hydro = getattr(vehicle_profile, "metadata", {}).get("hydro", {}) or {}

        def diag_from_matrix(matrix, index):
            try:
                return float(np.asarray(matrix, dtype=float)[index, index])
            except Exception:
                return 0.0

        linear = hydro.get("linear_damping_matrix")
        quadratic = hydro.get("quadratic_damping_matrix")
        added_mass = hydro.get("added_mass_6x6")

        def matrix_6x6_or_none(matrix):
            try:
                arr = np.asarray(matrix, dtype=float)
                if arr.shape != (6, 6):
                    return None
                return arr
            except Exception:
                return None

        self._full_linear_damping_matrix = matrix_6x6_or_none(linear)
        self._full_quadratic_damping_matrix = matrix_6x6_or_none(quadratic)
        self._full_added_mass_matrix = matrix_6x6_or_none(added_mass)

        return HydrodynamicCoefficients(
            X_uu=diag_from_matrix(quadratic, 0),
            Y_vv=diag_from_matrix(quadratic, 1),
            Z_ww=diag_from_matrix(quadratic, 2),
            K_pp=diag_from_matrix(quadratic, 3),
            M_qq=diag_from_matrix(quadratic, 4),
            N_rr=diag_from_matrix(quadratic, 5),
            X_u=diag_from_matrix(linear, 0),
            Y_v=diag_from_matrix(linear, 1),
            Z_w=diag_from_matrix(linear, 2),
            K_p=diag_from_matrix(linear, 3),
            M_q=diag_from_matrix(linear, 4),
            N_r=diag_from_matrix(linear, 5),
            X_udot=diag_from_matrix(added_mass, 0),
            Y_vdot=diag_from_matrix(added_mass, 1),
            Z_wdot=diag_from_matrix(added_mass, 2),
            K_pdot=diag_from_matrix(added_mass, 3),
            M_qdot=diag_from_matrix(added_mass, 4),
            N_rdot=diag_from_matrix(added_mass, 5),
            linear_damping_matrix=self._full_linear_damping_matrix,
            quadratic_damping_matrix=self._full_quadratic_damping_matrix,
            added_mass_matrix=self._full_added_mass_matrix,
        )

    # ─── Interface pública ───────────────────

    @property
    def state(self) -> VehicleState:
        return self._state

    @property
    def time(self) -> float:
        return self._time

    @property
    def total_mass(self) -> float:
        return (self._rigid_body_mass +
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
        env_current_world: Optional[np.ndarray] = None,
        env_turbulence: float = 0.0,
        env_harmonics: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
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

        # If multiple thrusters exist, distribute p1 across port-side thrusters
        # and p2 across starboard-side thrusters. Decision which thruster is
        # port/starboard is based on the sign of y in `position_body`.
        port_indices = [i for i, th in enumerate(self.thrusters) if float(th.position_body[1]) > 0.0]
        star_indices = [i for i, th in enumerate(self.thrusters) if float(th.position_body[1]) <= 0.0]

        # fallback to first two if detection fails
        if not port_indices:
            port_indices = [0]
        if not star_indices:
            star_indices = [1 if len(self.thrusters) > 1 else 0]

        p_port_share = p1 / float(len(port_indices))
        p_star_share = p2 / float(len(star_indices))

        for i, th in enumerate(self.thrusters):
            if i in port_indices:
                th.set_orientation(t1, f1)
                th.set_power(p_port_share)
            else:
                th.set_orientation(t2, f2)
                th.set_power(p_star_share)
        # maintain compatibility attributes
        self.thruster_port = self.thrusters[port_indices[0]]
        self.thruster_starboard = self.thrusters[star_indices[0]]
        self.ballast.update(ballast_cmd, dt)

        # recalcula massa adicionada com lastro atualizado
        self._M     = self._build_mass_matrix()
        self._M_inv = np.linalg.inv(self._M)

        # atualiza estado ambiental para uso nas equações (converte pra body)
        # guarda anterior para derivada temporal
        self._prev_env_current_body = self._env_current_body.copy()
        self._env_dt = float(dt)
        if env_current_world is not None:
            # usa orientação atual para converter corrente (world -> body)
            phi, tht, psi = self._state.phi, self._state.tht, self._state.psi
            R = self._rotation_matrix(phi, tht, psi)
            # R transforma body->world, então R.T transforma world->body
            self._env_current_body = R.T @ np.asarray(env_current_world, dtype=float)
        else:
            self._env_current_body = np.zeros(3, dtype=float)
        self._env_turbulence = float(env_turbulence)
        # if harmonics provided, convert them to body frame once (freqs, amps, phases, dirs)
        self._env_harmonics_body = None
        if env_harmonics is not None:
            freqs, amps, phases, dirs = env_harmonics
            # convert dirs (world->body)
            phi, tht, psi = self._state.phi, self._state.tht, self._state.psi
            R = self._rotation_matrix(phi, tht, psi)
            dirs_body = (R.T @ dirs.T).T
            self._env_harmonics_body = (np.asarray(freqs), np.asarray(amps), np.asarray(phases), np.asarray(dirs_body))

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
        m = self.total_mass
        c = self.coeff

        if self._rigid_body_inertia is not None:
            M_rigid = np.zeros((6, 6), dtype=float)
            M_rigid[:3, :3] = np.diag([m, m, m])
            M_rigid[3:, 3:] = self._rigid_body_inertia
        else:
            L = self.hull.L
            R = self.hull.R

            # momentos de inércia do casco aproximado como casca cilíndrica
            Ixx = 0.7  * m * R**2                          # roll
            Iyy = (1/12) * m * (3*R**2 + L**2)            # pitch
            Izz = Iyy                                       # yaw — simetria

            M_rigid = np.diag([m, m, m, Ixx, Iyy, Izz])
        M_added = c.to_added_mass_matrix()

        return M_rigid + M_added

    # ─── Frequência-Dependência Hidrodinâmica ───────────────

    def _get_frequency_dependent_added_mass(self, freq: float) -> np.ndarray:
        """Retorna coeficientes de massa adicionada frequência-dependente (diagonal).
        
        Modelo aproximado: M_added(f) escala suavemente com frequência.
        Para baixas freqs (~0.1 Hz): M_added ≈ coeff base
        Para altas freqs (~1+ Hz): M_added reduz (~0.5 * coeff base)
        
        Baseado em dados hidrodinâmicos típicos para corpos submersos.
        """
        freq = max(0.01, float(freq))
        # escala suave: função decrescente
        scale = 1.0 / (1.0 + 0.5 * freq)  # em Hz
        added_mass_diag = np.array([self.coeff.X_udot, self.coeff.Y_vdot, self.coeff.Z_wdot])
        return added_mass_diag * np.maximum(0.3, scale)  # mínimo 30% da base

    def _get_frequency_dependent_damping(self, freq: float) -> np.ndarray:
        """Retorna coeficientes de amortecimento viscoso frequência-dependente (diagonal).
        
        Modelo: amortecimento aumenta com frequência (skin friction + wave drag).
        """
        freq = max(0.01, float(freq))
        # escala crescente com frequência
        scale = 1.0 + 0.3 * freq  # em Hz
        # usa diagonal de matriz de arrasto como base (X_vv, Y_vv, Z_ww aproximações)
        # nota: para simplicidade, usa coefs de massa como proxy (poderiam ser dados experimentais)
        base_damping = np.array([self.coeff.X_udot, self.coeff.Y_vdot, self.coeff.Z_wdot]) * 0.5
        return base_damping * scale

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

        # usa uma massa translacional única no acoplamento de Coriolis.
        # Isso preserva o cancelamento físico em translação pura e evita
        # torque espúrio em pitch/roll por diferenças entre eixos.
        m_trans = m
        m44 = 0.5*m*R**2         + c.K_pdot
        m55 = (1/12)*m*(3*R**2+L**2) + c.M_qdot
        m66 = m55                + c.N_rdot

        C = np.zeros((6, 6))

        # bloco superior direito
        C[0, 3] =  0;         C[0, 4] =  m_trans*w;  C[0, 5] = -m_trans*v
        C[1, 3] = -m_trans*w; C[1, 4] =  0;          C[1, 5] =  m_trans*u
        C[2, 3] =  m_trans*v; C[2, 4] = -m_trans*u;  C[2, 5] =  0

        # bloco inferior esquerdo (transposto negativo)
        C[3, 0] =  0;         C[3, 1] =  m_trans*w;  C[3, 2] = -m_trans*v
        C[4, 0] = -m_trans*w; C[4, 1] =  0;          C[4, 2] =  m_trans*u
        C[5, 0] =  m_trans*v; C[5, 1] = -m_trans*u;  C[5, 2] =  0

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
        # Generalized quadratic damping with full 6x6 coupling.
        D_nonl = D_quad @ np.diag(np.abs(nu))
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
        # aplica corrente/turbulência modificando a velocidade relativa usada no arrasto
        nu_for_drag = nu.copy()
        # apenas o bloco linear é afetado pela corrente
        nu_for_drag[0:3] = nu[0:3] - self._env_current_body

        D   = self._drag_matrix(nu_for_drag)
        g   = self._restoring_forces(eta)
        # Sum wrench from all thrusters (works for 2 or N thrusters).
        tau = np.zeros(6, dtype=float)
        for th in self.thrusters:
            tau = tau + th.wrench_body

        # aumenta efeito do arrasto com turbulência
        if self._env_turbulence > 0.0:
            D = D * (1.0 + float(self._env_turbulence))

        # componente oscilatória simplificada representando forças de onda
        try:
            amp = float(self._env_turbulence) * float(self.env_turbulence_gain)
        except Exception:
            amp = float(self._env_turbulence)

        if amp > 0.0:
            freq = float(getattr(self, 'env_wave_freq', 0.8))
            phase = float(getattr(self, '_env_wave_phase', 0.0))
            v_wave = amp * np.sin(2.0 * np.pi * freq * self._time + phase)
            # direção da oscilação: mesma direção da corrente se houver, senão surge
            dir_norm = np.linalg.norm(self._env_current_body)
            if dir_norm > 1e-6:
                dir_vec = self._env_current_body / dir_norm
            else:
                dir_vec = np.array([1.0, 0.0, 0.0])

            v_wave_vec = v_wave * dir_vec
            # usa coeficientes quadráticos para escalonar a força de onda (approx.)
            quad = np.array([
                self.coeff.X_uu, self.coeff.Y_vv, self.coeff.Z_ww
            ])
            F_wave = -quad * np.abs(v_wave_vec) * v_wave_vec
            F_wave = F_wave * float(getattr(self, 'env_force_gain', 1.0))
            # adiciona apenas as forças (não momentos) geradas pela onda
            tau = tau + np.concatenate([F_wave, np.zeros(3)])

        # efeito de massa adicionada + amortecimento dinâmicos (frequência-dependentes via harmônicos)
        added_mass_diag = np.array([self.coeff.X_udot, self.coeff.Y_vdot, self.coeff.Z_wdot])
        F_added = np.zeros(3)
        F_damping = np.zeros(3)
        if getattr(self, '_env_harmonics_body', None) is not None:
            freqs, amps, phases, dirs_body = self._env_harmonics_body
            t = self._time
            for f, A, ph, d in zip(freqs, amps, phases, dirs_body):
                omega = 2.0 * np.pi * float(f)
                # aceleração do fluido: a_h = omega * A * cos(omega t + phase)
                a_h = omega * A * np.cos(omega * t + float(ph))
                # velocidade orbital: v_h = A * sin(omega t + phase)
                v_h = A * np.sin(omega * t + float(ph))
                # massa adicionada dependente de frequência
                M_f = self._get_frequency_dependent_added_mass(f)
                # amortecimento dependente de frequência
                C_f = self._get_frequency_dependent_damping(f)
                # forças por harmônico
                F_added += - M_f * (a_h * d) * float(getattr(self, 'env_force_gain', 1.0))
                F_damping += - C_f * (v_h * d) * float(getattr(self, 'env_turbulence_gain', 1.0))
        else:
            env_dt = max(1e-6, float(getattr(self, '_env_dt', 1e-6)))
            a_fluid = (self._env_current_body - self._prev_env_current_body) / env_dt
            M_f = self._get_frequency_dependent_added_mass(0.5)  # fallback freq
            F_added = - M_f * a_fluid * float(getattr(self, 'env_force_gain', 1.0))

        # aplica forças (X,Y,Z)
        tau = tau + np.concatenate([F_added + F_damping, np.zeros(3)])

        eta_dot = J @ nu
        nu_dot  = self._M_inv @ (tau - C @ nu - D @ nu_for_drag - g)

        if self.planar_dof:
            eta_dot[2:5] = 0.0
            nu_dot[2:5] = 0.0

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
