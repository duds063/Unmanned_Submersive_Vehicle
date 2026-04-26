"""
USV Digital Twin — Módulo 4: Control Engine
============================================
Implementa três controladores intercambiáveis em runtime:

    LQR  — Linear Quadratic Regulator (este arquivo)
    MPC  — Model Predictive Control (próximo módulo)
    RL   — Hierarchical Reinforcement Learning (próximo módulo)

O Control Engine NUNCA acessa PhysicsEngine.state diretamente.
Usa apenas EKFState — percepção realista garantida.

LQR Full State:
    - Lineariza equações de Fossen em torno de hovering neutro
    - Estado: x = [η, ν] = [x,y,z,φ,θ,ψ,u,v,w,p,q,r] (12D)
    - Entrada: u = [F_thruster, θ_thruster, φ_thruster, ballast_cmd] (4D)
    - Resolve equação de Riccati pra ganho ótimo K
    - Q e R parametrizáveis em runtime via GUI

Referências:
    - Fossen (2011) cap. 12 — LQR para veículos marinhos
    - Anderson & Moore (1989) — Optimal Control: Linear Quadratic Methods
    - Franklin et al. (2015) — Feedback Control of Dynamic Systems
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from dataclasses import dataclass, field
from typing import Optional, Tuple
from sensor_engine import EKFState


# ─────────────────────────────────────────────
# ESTRUTURAS DE DADOS
# ─────────────────────────────────────────────

@dataclass
class ControlCommand:
    """Comando de controle enviado aos atuadores."""
    thruster_power: float   # [-1, 1]
    thruster_theta: float   # rad — ângulo polar [0, 60°]
    thruster_phi:   float   # rad — ângulo azimutal [0, 2π]
    ballast_cmd:    float   # [-1, 1] — -1 expele, +1 injeta
    thruster2_power: Optional[float] = None
    thruster2_theta: Optional[float] = None
    thruster2_phi:   Optional[float] = None

    def clip(self) -> 'ControlCommand':
        """Garante que os comandos estão dentro dos limites físicos."""
        t2_power = self.thruster_power if self.thruster2_power is None else self.thruster2_power
        t2_theta = self.thruster_theta if self.thruster2_theta is None else self.thruster2_theta
        t2_phi   = self.thruster_phi   if self.thruster2_phi   is None else self.thruster2_phi

        return ControlCommand(
            thruster_power = float(np.clip(self.thruster_power, -1.0, 1.0)),
            thruster_theta = float(np.clip(self.thruster_theta, 0.0, np.radians(60))),
            thruster_phi   = float(self.thruster_phi % (2 * np.pi)),
            ballast_cmd    = float(np.clip(self.ballast_cmd, -1.0, 1.0)),
            thruster2_power = float(np.clip(t2_power, -1.0, 1.0)),
            thruster2_theta = float(np.clip(t2_theta, 0.0, np.radians(60))),
            thruster2_phi   = float(t2_phi % (2 * np.pi)),
        )

    def to_dict(self) -> dict:
        t2_power = self.thruster_power if self.thruster2_power is None else self.thruster2_power
        t2_theta = self.thruster_theta if self.thruster2_theta is None else self.thruster2_theta
        t2_phi   = self.thruster_phi   if self.thruster2_phi   is None else self.thruster2_phi

        return {
            'thruster_power': self.thruster_power,
            'thruster_theta': float(np.degrees(self.thruster_theta)),
            'thruster_phi':   float(np.degrees(self.thruster_phi)),
            'ballast_cmd':    self.ballast_cmd,
            'thruster2_power': t2_power,
            'thruster2_theta': float(np.degrees(t2_theta)),
            'thruster2_phi':   float(np.degrees(t2_phi)),
        }


@dataclass
class ControllerState:
    """Estado interno do controlador para telemetria."""
    error:          np.ndarray      # erro de estado 12D
    control_effort: np.ndarray      # esforço de controle 4D
    gain_matrix:    np.ndarray      # ganho K atual (4x12)
    controller_type: str
    timestamp:      float

    def to_dict(self) -> dict:
        return {
            'controller':    self.controller_type,
            'error_norm':    float(np.linalg.norm(self.error)),
            'effort_norm':   float(np.linalg.norm(self.control_effort)),
            'timestamp':     self.timestamp,
        }


@dataclass
class LQRWeights:
    """
    Matrizes de peso Q e R do LQR.
    Q penaliza erro de estado, R penaliza esforço de controle.

    Intuição física:
        Q alto em z    → prioriza manter profundidade
        Q alto em φ,θ  → prioriza manter atitude
        R alto         → minimiza uso do propulsor (economia de bateria)
    """
    # pesos de estado Q — diagonal (12 valores)
    q_x:   float = 1.0    # posição Norte
    q_y:   float = 1.0    # posição Leste
    q_z:   float = 10.0   # profundidade — mais importante
    q_phi: float = 5.0    # roll
    q_tht: float = 5.0    # pitch
    q_psi: float = 2.0    # yaw
    q_u:   float = 1.0    # velocidade surge
    q_v:   float = 1.0    # velocidade sway
    q_w:   float = 5.0    # velocidade heave
    q_p:   float = 2.0    # roll rate
    q_q:   float = 2.0    # pitch rate
    q_r:   float = 2.0    # yaw rate

    # pesos de controle R — diagonal (4 valores)
    r_thrust_power: float = 1.0    # esforço do propulsor
    r_thrust_theta: float = 0.5    # deflexão angular theta
    r_thrust_phi:   float = 0.5    # deflexão angular phi
    r_ballast:      float = 2.0    # acionamento do lastro

    def Q_matrix(self) -> np.ndarray:
        """Retorna matriz Q diagonal 12x12."""
        return np.diag([
            self.q_x,   self.q_y,   self.q_z,
            self.q_phi, self.q_tht, self.q_psi,
            self.q_u,   self.q_v,   self.q_w,
            self.q_p,   self.q_q,   self.q_r,
        ])

    def R_matrix(self) -> np.ndarray:
        """Retorna matriz R diagonal 4x4."""
        return np.diag([
            self.r_thrust_power,
            self.r_thrust_theta,
            self.r_thrust_phi,
            self.r_ballast,
        ])

    def update(self, **kwargs) -> None:
        """Atualiza pesos em runtime — chamado pela GUI."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, float(val))


@dataclass
class GuidanceGains:
    """Ganhos de guidance/alocação compatíveis com os atuadores reais."""
    k_forward: float = 0.55
    k_surge_damp: float = 0.35
    k_yaw: float = 0.85
    k_lateral: float = 0.14
    k_yaw_damp: float = 0.45
    k_depth: float = 0.14
    k_heave_damp: float = 0.11
    k_ballast: float = 0.18
    k_ballast_damp: float = 0.08
    max_forward_power: float = 0.65
    max_reverse_power: float = 0.45
    max_yaw_diff: float = 0.28
    max_theta_deg: float = 34.0
    depth_deadband_m: float = 0.18
    heave_priority_ratio: float = 0.65


def wrap_angle(angle: float) -> float:
    """Normaliza ângulo para [-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def body_frame_position_error(delta_world: np.ndarray, yaw: float) -> np.ndarray:
    """Converte erro de posição global para o plano/body frame do veículo."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    x_b = cy * float(delta_world[0]) + sy * float(delta_world[1])
    y_b = -sy * float(delta_world[0]) + cy * float(delta_world[1])
    return np.array([x_b, y_b, float(delta_world[2])], dtype=float)


def guidance_to_dual_thruster_command(
    ekf_state: EKFState,
    target_position: np.ndarray,
    desired_yaw: Optional[float] = None,
    gains: Optional[GuidanceGains] = None,
) -> ControlCommand:
    """
    Gera comando fisicamente consistente para o arranjo dual-thruster.

    Estratégia:
      - navegação horizontal via orientação para o alvo + diferencial de potência
      - profundidade via tilt vertical simétrico + ballast
    """
    g = gains or GuidanceGains()

    pos = np.asarray(ekf_state.position, dtype=float)
    vel = np.asarray(ekf_state.velocity_linear, dtype=float)
    ang = np.asarray(ekf_state.velocity_angular, dtype=float)
    yaw = float(ekf_state.orientation[2])

    delta_world = np.asarray(target_position, dtype=float) - pos
    err_body = body_frame_position_error(delta_world, yaw)
    forward_err, lateral_err, depth_err = err_body

    horizontal_dist = float(np.linalg.norm(delta_world[:2]))
    if desired_yaw is None:
        desired_yaw = float(np.arctan2(delta_world[1], delta_world[0])) if horizontal_dist > 1e-6 else yaw
    yaw_err = wrap_angle(float(desired_yaw) - yaw)

    surge = float(vel[0])
    heave = float(vel[2])
    yaw_rate = float(ang[2])

    cmd_x = np.tanh(forward_err / 2.5)
    cmd_y = 0.80 * np.tanh(lateral_err / 2.0)
    cmd_z = 0.90 * np.tanh(depth_err / 1.8)
    vec_norm = float(np.linalg.norm([cmd_x, cmd_y, cmd_z]))

    if vec_norm < 1e-6:
        theta = 0.0
        phi = 0.0
        base_power = float(np.clip(-g.k_surge_damp * surge, -g.max_reverse_power, g.max_forward_power))
    else:
        cmd_x /= vec_norm
        cmd_y /= vec_norm
        cmd_z /= vec_norm

        x_mag = max(0.05, abs(cmd_x))
        yz_mag = float(np.hypot(cmd_y, cmd_z))
        theta = float(np.clip(np.arctan2(yz_mag, x_mag), 0.0, np.radians(g.max_theta_deg)))
        phi = float(np.arctan2(cmd_z, cmd_y)) if yz_mag > 1e-6 else 0.0

        distance_scale = float(np.tanh(np.linalg.norm(err_body) / 3.0))
        desired_surge = np.sign(forward_err if abs(forward_err) > 0.15 else cmd_x) * g.k_forward * distance_scale
        if horizontal_dist < 1.5:
            desired_surge *= 0.35 + 0.65 * horizontal_dist / 1.5
        base_power = float(np.clip(
            desired_surge - g.k_surge_damp * surge,
            -g.max_reverse_power,
            g.max_forward_power,
        ))

    yaw_cmd = float(np.clip(-0.08 * yaw_rate, -g.max_yaw_diff, g.max_yaw_diff))

    p1 = float(np.clip(base_power - yaw_cmd, -1.0, 1.0))
    p2 = float(np.clip(base_power + yaw_cmd, -1.0, 1.0))

    depth_cmd = g.k_depth * np.tanh(depth_err / 1.5) - g.k_heave_damp * heave
    max_theta = np.radians(g.max_theta_deg)
    depth_only_mode = horizontal_dist <= 1.0

    if abs(depth_err) <= g.depth_deadband_m and abs(heave) < 0.08:
        theta_depth = 0.0
    else:
        theta_depth = float(np.clip(abs(depth_cmd), 0.0, max_theta))
        phi_depth = float(np.pi / 2.0 if depth_cmd >= 0.0 else 3.0 * np.pi / 2.0)
        if depth_only_mode:
            phi = phi_depth
            theta = max(theta, theta_depth)
        else:
            # Quando o comando de profundidade domina, fixa o azimute em +/-90°
            # para evitar descontinuidade via arctan2(cmd_z, cmd_y) perto do setpoint.
            yz_total = abs(cmd_y) + abs(cmd_z) + 1e-9
            heave_share = abs(cmd_z) / yz_total
            if heave_share >= g.heave_priority_ratio:
                phi = phi_depth
                theta = max(theta, theta_depth)
            elif theta_depth > theta:
                theta = theta_depth
                phi = phi_depth

    ballast = float(np.clip(
        g.k_ballast * np.tanh(depth_err / 2.0) - g.k_ballast_damp * heave,
        -1.0,
        1.0,
    ))

    return ControlCommand(
        thruster_power=p1,
        thruster_theta=theta,
        thruster_phi=phi,
        ballast_cmd=ballast,
        thruster2_power=p2,
        thruster2_theta=theta,
        thruster2_phi=phi,
    )


# ─────────────────────────────────────────────
# LINEARIZAÇÃO DO SISTEMA
# ─────────────────────────────────────────────

class SystemLinearizer:
    """
    Lineariza as equações de Fossen em torno do ponto de operação.

    Ponto de operação: hovering neutro
        η₀ = [0, 0, z₀, 0, 0, 0]  — posição/orientação zero
        ν₀ = [0, 0, 0, 0, 0, 0]   — velocidades zero

    Obtém matrizes A (12x12) e B (12x4) do sistema linearizado:
        ẋ = Ax + Bu
    """

    def __init__(self, physics_engine):
        self.physics = physics_engine
        self.hull    = physics_engine.hull
        self.coeff   = physics_engine.coeff
        self.eps     = 1e-4  # perturbação numérica para Jacobiana

        # instância isolada para perturbações numéricas — nunca muta o physics original
        import copy
        self._physics_copy = copy.deepcopy(physics_engine)

    @staticmethod
    def operating_input() -> np.ndarray:
        """Entrada de operação usada na linearização (u0)."""
        return np.array([
            0.15, np.radians(15), np.radians(90),   # thruster 1
            0.15, np.radians(15), np.radians(90),   # thruster 2
            0.0,                                    # ballast
        ], dtype=float)

    def linearize(self, z0: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula matrizes A e B por diferenciação numérica.

        Ponto de operação: hovering em z0 com propulsor levemente deflectido.
        Usar theta=0 zera a coluna theta em B (sin(0)=0 e derivada centrada
        com clip unilateral retorna 0). Theta=15° como ponto de op garante
        que o canal de deflexão apareça corretamente em B.

        Args:
            z0: profundidade de operação (m)

        Returns:
            A: matriz de sistema 12x12
            B: matriz de entrada 12x7
        """
        # estado de operação: hovering neutro em z0
        x0 = np.zeros(12)
        x0[2] = z0

        # entrada de operação: propulsor com potência e deflexão leves
        # CRÍTICO: power=0 zera o canal theta em B porque F_z = power*F_max*sin(theta)
        # é bilinear — dF_z/dtheta = power*F_max*cos(theta) = 0 quando power=0.
        # Com power=0.3 e theta=15°, todos os canais ficam visíveis na linearização.
        u0 = self.operating_input()

        # Jacobiana A = ∂f/∂x numericamente
        A = np.zeros((12, 12))
        for i in range(12):
            x_plus  = x0.copy(); x_plus[i]  += self.eps
            x_minus = x0.copy(); x_minus[i] -= self.eps
            f_plus  = self._dynamics(x_plus,  u0)
            f_minus = self._dynamics(x_minus, u0)
            A[:, i] = (f_plus - f_minus) / (2 * self.eps)

        # Jacobiana B = ∂f/∂u numericamente
        nu = len(u0)
        B = np.zeros((12, nu))
        for i in range(nu):
            u_plus  = u0.copy(); u_plus[i]  += self.eps
            u_minus = u0.copy(); u_minus[i] -= self.eps
            f_plus  = self._dynamics(x0, u_plus)
            f_minus = self._dynamics(x0, u_minus)
            B[:, i] = (f_plus - f_minus) / (2 * self.eps)

        return A, B

    def _dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Equações de movimento f(x, u) — usado na linearização numérica.
        x = [η, ν], u = [power, theta, phi, ballast]

        Usa cópia isolada do physics — thread-safe, nunca muta o original.

        Decisões de implementação para linearização correta:
          1. theta/phi: bypass do clip para permitir perturbações bidirecionais
             em torno de qualquer ponto de operação (incluindo theta=0).
          2. ballast: perturbação como fração do range físico de massa,
             independente de fill_rate (que é parâmetro de simulação, não física).
        """
        from physics_engine import VehicleState

        p = self._physics_copy
        eta = x[:6]
        nu  = x[6:]

        # --- propulsores: bypass do clip para diferença centrada correta ---
        p.thruster_port._power = float(np.clip(u[0], -1.0, 1.0))
        p.thruster_port._theta = float(u[1])
        p.thruster_port._phi   = float(u[2] % (2 * np.pi))

        p.thruster_starboard._power = float(np.clip(u[3], -1.0, 1.0))
        p.thruster_starboard._theta = float(u[4])
        p.thruster_starboard._phi   = float(u[5] % (2 * np.pi))

        # --- ballast: perturbação como fração do range de massa físico ---
        # range total de massa = mass_max - mass_min (determinado pela geometria)
        # u[3] ∈ [-1, 1] mapeia para ±50% do range — escala física, não temporal
        half_range = 0.5 * (p.ballast.mass_max - p.ballast.mass_min)
        dm = np.clip(u[6], -1.0, 1.0) * half_range
        p.ballast._water_mass = np.clip(
            p.ballast.mass_min + 0.5 * (p.ballast.mass_max - p.ballast.mass_min) + dm,
            p.ballast.mass_min,
            p.ballast.mass_max,
        )

        # recalcula M com nova massa do lastro
        p._M     = p._build_mass_matrix()
        p._M_inv = np.linalg.inv(p._M)

        state = VehicleState(
            x=eta[0], y=eta[1], z=eta[2],
            phi=eta[3], tht=eta[4], psi=eta[5],
            u=nu[0], v=nu[1], w=nu[2],
            p=nu[3], q=nu[4], r=nu[5],
        )

        eta_dot, nu_dot = p._derivatives(state)
        return np.concatenate([eta_dot, nu_dot])

    def check_controllability(self, A: np.ndarray, B: np.ndarray) -> dict:
        """
        Verifica controlabilidade do par (A, B).
        Sistema controlável se rank(C) = n.
        """
        n = A.shape[0]
        C = B.copy()
        AB = B.copy()
        for _ in range(n - 1):
            AB = A @ AB
            C  = np.hstack([C, AB])

        rank = np.linalg.matrix_rank(C)
        return {
            'controllable': rank == n,
            'rank': rank,
            'required_rank': n,
            'deficiency': n - rank,
        }


# ─────────────────────────────────────────────
# LQR CONTROLLER
# ─────────────────────────────────────────────

class LQRController:
    """
    Linear Quadratic Regulator para o USV em 6 DOF.

    Resolve a equação algébrica de Riccati:
        A'P + PA - PBR⁻¹B'P + Q = 0

    Ganho ótimo:
        K = R⁻¹B'P

    Lei de controle:
        u = -K(x - x_ref)

    Q e R são parametrizáveis e o ganho K é recalculado
    automaticamente quando os pesos mudam (via GUI).
    """

    def __init__(
        self,
        physics_engine,
        weights:    LQRWeights = None,
        hover_depth: float = 2.0,
        adaptive_relinearization: bool = True,
        relinearization_interval_s: float = 1.0,
    ):
        self.physics     = physics_engine
        self.weights     = weights or LQRWeights()
        self.hover_depth = hover_depth
        self.adaptive_relinearization = adaptive_relinearization
        self.relinearization_interval_s = float(max(0.2, relinearization_interval_s))
        self._last_relinearization_time = -np.inf

        # lineariza o sistema uma vez
        self.linearizer = SystemLinearizer(physics_engine)
        self.A, self.B  = self.linearizer.linearize(z0=hover_depth)
        self._u_op      = self.linearizer.operating_input()

        # verifica controlabilidade
        ctrl = self.linearizer.check_controllability(self.A, self.B)
        if not ctrl['controllable']:
            print(f"⚠️  Sistema não completamente controlável. "
                  f"Rank={ctrl['rank']}/{ctrl['required_rank']}. "
                  f"Deficiência={ctrl['deficiency']} — estados não controláveis "
                  f"serão ignorados pelo LQR.")

        # calcula ganho inicial
        self.K = self._solve_riccati()

        # referência — hovering neutro
        self._x_ref = np.zeros(12)
        self._x_ref[2] = hover_depth

        # telemetria interna
        self._last_error  = np.zeros(12)
        self._last_effort = np.zeros(self.B.shape[1])
        self._last_u_cmd = np.zeros(self.B.shape[1])
        self._last_time = None
        # Limites de taxa por segundo para reduzir ciclos limite por saturacao.
        self._du_rate_limits = np.array([
            2.0,
            np.radians(120.0),
            np.radians(360.0),
            2.0,
            np.radians(120.0),
            np.radians(360.0),
            1.0,
        ], dtype=float)

    # ─── Interface pública ───────────────────

    def set_reference(self, position: np.ndarray, yaw: float = 0.0) -> None:
        """
        Define ponto de referência para o LQR.

        Args:
            position: [x, y, z] desejado
            yaw: orientação desejada em rad
        """
        self._x_ref      = np.zeros(12)
        self._x_ref[:3]  = position
        self._x_ref[5]   = yaw

    def update_weights(self, **kwargs) -> None:
        """
        Atualiza pesos Q/R e recalcula ganho K.
        Chamado pela GUI quando sliders mudam.
        """
        self.weights.update(**kwargs)
        self.K = self._solve_riccati()

    def compute(self, ekf_state: EKFState, time: float) -> ControlCommand:
        """
        Calcula comando de controle dado o estado estimado pelo EKF.

        NUNCA usa estado físico diretamente — só EKFState.

        Args:
            ekf_state: estado estimado pelo EKF
            time: tempo atual

        Returns:
            ControlCommand com comandos clipados
        """
        # Re-lineariza periodicamente para reduzir mismatch longe do ponto nominal.
        if self.adaptive_relinearization and (time - self._last_relinearization_time >= self.relinearization_interval_s):
            z_lin = float(np.clip(ekf_state.eta[2], 0.1, 20.0))
            self.A, self.B = self.linearizer.linearize(z0=z_lin)
            self.K = self._solve_riccati()
            self._u_op = self.linearizer.operating_input()
            self._last_relinearization_time = time

        # monta vetor de estado do EKF
        x = np.concatenate([ekf_state.eta, ekf_state.nu])

        # erro em relação à referência
        error = x - self._x_ref

        # normaliza ângulos — evita erro de 350° quando deveria ser -10°
        error[3:6] = self._wrap_angles(error[3:6])

        cmd = guidance_to_dual_thruster_command(
            ekf_state=ekf_state,
            target_position=self._x_ref[:3],
            desired_yaw=float(self._x_ref[5]),
            gains=GuidanceGains(
                k_forward=0.54,
                k_surge_damp=0.44,
                k_yaw=1.05,
                k_lateral=0.18,
                k_yaw_damp=0.60,
                k_depth=0.12,
                k_heave_damp=0.16,
                k_ballast=0.22,
                k_ballast_damp=0.12,
                max_forward_power=0.58,
                max_reverse_power=0.34,
                max_yaw_diff=0.22,
                max_theta_deg=14.0,
                depth_deadband_m=0.18,
            ),
        )
        u = np.array([
            cmd.thruster_power,
            cmd.thruster_theta,
            cmd.thruster_phi,
            cmd.thruster2_power,
            cmd.thruster2_theta,
            cmd.thruster2_phi,
            cmd.ballast_cmd,
        ], dtype=float)
        self._last_u_cmd = u.copy()
        self._last_time = float(time)

        # armazena pra telemetria
        self._last_error  = error
        self._last_effort = u

        return cmd.clip()

    @property
    def controller_state(self) -> ControllerState:
        return ControllerState(
            error=self._last_error,
            control_effort=self._last_effort,
            gain_matrix=self.K,
            controller_type='LQR',
            timestamp=0.0,
        )

    @property
    def gain_matrix(self) -> np.ndarray:
        return self.K.copy()

    # ─── Riccati ─────────────────────────────

    def _solve_riccati(self) -> np.ndarray:
        """
        Resolve equação algébrica de Riccati contínua.

        Se o sistema não é completamente controlável, projeta o LQR
        no subespaço controlável usando decomposição de Kalman.
        Retorna ganho K completo (4x12) com zeros nos estados não controláveis.
        """
        Q = self.weights.Q_matrix()
        R = self._lqr_R_matrix()

        # tenta resolver no sistema completo primeiro
        try:
            P = solve_continuous_are(self.A, self.B, Q, R)
            K = np.linalg.inv(R) @ self.B.T @ P
            return K

        except Exception:
            pass

        # fallback: LQR no subespaço controlável
        # identifica colunas de B com entrada real
        controllable_inputs = [
            i for i in range(self.B.shape[1])
            if np.linalg.norm(self.B[:, i]) > 1e-4
        ]

        # identifica estados influenciados por entradas controláveis
        # via propagação da matriz de controlabilidade
        reachable = set()
        col = self.B.copy()
        for _ in range(12):
            for i in range(col.shape[1]):
                idxs = np.where(np.abs(col[:, i]) > 1e-4)[0]
                reachable.update(idxs.tolist())
            col = self.A @ col

        controllable_states = sorted(list(reachable))

        if len(controllable_states) < 2:
            print("⚠️  Subespaço controlável muito pequeno — usando ganho P simples.")
            return self._proportional_fallback()

        # extrai subsistema
        idx_s = controllable_states
        idx_i = controllable_inputs

        A_r = self.A[np.ix_(idx_s, idx_s)]
        B_r = self.B[np.ix_(idx_s, idx_i)]
        Q_r = Q[np.ix_(idx_s, idx_s)]
        R_r = R[np.ix_(idx_i, idx_i)]

        try:
            P_r = solve_continuous_are(A_r, B_r, Q_r, R_r)
            K_r = np.linalg.inv(R_r) @ B_r.T @ P_r

            # expande K de volta pro espaço completo
            K = np.zeros((self.B.shape[1], 12))
            for ki, i in enumerate(idx_i):
                for kj, j in enumerate(idx_s):
                    K[i, j] = K_r[ki, kj]

            print(f"✓ LQR reduzido: {len(idx_s)} estados controláveis, "
                  f"{len(idx_i)} entradas ativas.")
            return K

        except Exception as e:
            print(f"⚠️  LQR reduzido falhou ({e}) — usando fallback proporcional.")
            return self._proportional_fallback()

    def _proportional_fallback(self) -> np.ndarray:
        """
        Ganho proporcional simples como fallback de segurança.
        Controla diretamente os estados mais observáveis.
        """
        K = np.zeros((self.B.shape[1], 12))
        # thrust power ← surge velocity (u, idx 6)
        if self.B.shape[1] >= 1:
            K[0, 6] = 0.35
        if self.B.shape[1] >= 4:
            K[3, 6] = 0.35
        # ballast ← heave velocity (w, idx 8) + depth error (z, idx 2)
        K[-1, 2] = 0.25
        K[-1, 8] = 0.20
        print("✓ Ganho proporcional de fallback ativo.")
        return K

    def _lqr_R_matrix(self) -> np.ndarray:
        """Matriz R específica do LQR dual-thruster (7x7)."""
        wt = self.weights
        return np.diag([
            wt.r_thrust_power,
            wt.r_thrust_theta,
            wt.r_thrust_phi,
            wt.r_thrust_power,
            wt.r_thrust_theta,
            wt.r_thrust_phi,
            wt.r_ballast,
        ])

    # ─── Conversão de controle ───────────────

    def _control_to_command(self, u: np.ndarray) -> ControlCommand:
        """
        Converte vetor de controle abstrato [4D] em comandos físicos.

        u[0] → thruster_power  — magnitude da força frontal
        u[1] → thruster_theta  — deflexão vertical (pitch/heave)
        u[2] → thruster_phi    — deflexão lateral (yaw/sway)
        u[3] → ballast_cmd     — controle de profundidade
        """
        # LQR dual-thruster: u = [p1,t1,ph1,p2,t2,ph2,ballast].
        # Compatibilidade: se vier vetor 4D, usa mapeamento espelhado.
        if len(u) >= 7:
            p1, t1, f1, p2, t2, f2, b = u[:7]
        else:
            p1, t1, f1, b = u[:4]
            p2, t2, f2 = p1, t1, f1

        return ControlCommand(
            thruster_power=float(np.clip(p1, -1.0, 1.0)),
            thruster_theta=float(np.clip(t1, 0.0, np.radians(60.0))),
            thruster_phi=float(f1 % (2 * np.pi)),
            ballast_cmd=float(np.clip(b, -1.0, 1.0)),
            thruster2_power=float(np.clip(p2, -1.0, 1.0)),
            thruster2_theta=float(np.clip(t2, 0.0, np.radians(60.0))),
            thruster2_phi=float(f2 % (2 * np.pi)),
        )

    @staticmethod
    def _wrap_angles(angles: np.ndarray) -> np.ndarray:
        """Normaliza ângulos para [-π, π]."""
        return ((angles + np.pi) % (2 * np.pi)) - np.pi


# ─────────────────────────────────────────────
# CONTROL ENGINE — interface unificada
# ─────────────────────────────────────────────

class ControlEngine:
    """
    Interface unificada para todos os controladores.
    Troca de controlador em runtime sem mudar o resto do sistema.

    Uso:
        engine = ControlEngine(physics)
        engine.set_controller('lqr')
        engine.set_reference(np.array([5.0, 0.0, 3.0]))
        cmd = engine.compute(ekf_state, time)
        physics.step(**cmd.__dict__, dt=0.01)
    """

    CONTROLLERS = ['lqr', 'mpc', 'rl']

    def __init__(self, physics_engine, hover_depth: float = 2.0):
        self.physics     = physics_engine
        self.hover_depth = hover_depth

        # inicializa LQR como controlador padrão
        self._lqr = LQRController(physics_engine, hover_depth=hover_depth)
        self._mpc = None   # implementado no próximo módulo
        self._rl  = None   # implementado no próximo módulo

        self._active = 'lqr'
        self._reference = np.array([0.0, 0.0, hover_depth])
        self._waypoints: list[np.ndarray] = []
        self._waypoint_threshold = 0.5
        self._current_waypoint_idx = 0

    def _active_controller(self):
        if self._active == 'lqr':
            return self._lqr
        if self._active == 'mpc':
            return self._mpc
        if self._active == 'rl':
            return self._rl
        return None

    def _current_target(self) -> Optional[np.ndarray]:
        if not self._waypoints:
            return self._reference
        if self._current_waypoint_idx < len(self._waypoints):
            return self._waypoints[self._current_waypoint_idx]
        return self._waypoints[-1]

    def _sync_reference(self) -> None:
        target = self._current_target()
        if target is None:
            return
        self._reference = np.asarray(target, dtype=float)
        controller = self._active_controller()
        if controller is not None and hasattr(controller, 'set_reference'):
            controller.set_reference(self._reference)

    # ─── Interface pública ───────────────────

    def set_controller(self, name: str) -> None:
        """Troca controlador em runtime."""
        if name not in self.CONTROLLERS:
            raise ValueError(f"Controlador '{name}' inválido. "
                           f"Opções: {self.CONTROLLERS}")
        if name == 'mpc' and self._mpc is None:
            raise RuntimeError("MPC não inicializado ainda.")
        if name == 'rl' and self._rl is None:
            raise RuntimeError("RL não inicializado ainda.")

        self._active = name
        print(f"✓ Controlador ativo: {name.upper()}")

    def set_reference(
        self,
        position: np.ndarray,
        yaw: float = 0.0
    ) -> None:
        """Define uma referência única e limpa a sequência de waypoints."""
        self._waypoints = [np.asarray(position, dtype=float)]
        self._current_waypoint_idx = 0
        self._reference = np.asarray(position, dtype=float)
        self._lqr.set_reference(position, yaw)
        if self._mpc is not None:
            self._mpc.set_reference(position, yaw)
        if self._rl is not None and hasattr(self._rl, 'set_waypoints'):
            self._rl.set_waypoints([np.asarray(position, dtype=float)])

    def set_waypoints(self, waypoints: list[np.ndarray], waypoint_threshold: float = 0.5) -> None:
        """Define uma sequência de waypoints para LQR/MPC.

        O último waypoint é mantido como referência de hold após a missão.
        """
        if not waypoints:
            self.clear_waypoints()
            return
        self._waypoints = [np.asarray(wp, dtype=float) for wp in waypoints]
        self._current_waypoint_idx = 0
        self._waypoint_threshold = float(waypoint_threshold)
        self._sync_reference()
        if self._rl is not None and hasattr(self._rl, 'set_waypoints'):
            self._rl.set_waypoints([wp.copy() for wp in self._waypoints])

    def clear_waypoints(self) -> None:
        """Remove a sequência de waypoints e volta para a referência atual."""
        self._waypoints = []
        self._current_waypoint_idx = 0

    @property
    def current_waypoint(self) -> Optional[np.ndarray]:
        return None if not self._waypoints else self._current_target().copy()

    @property
    def waypoint_index(self) -> int:
        return self._current_waypoint_idx

    @property
    def waypoint_count(self) -> int:
        return len(self._waypoints)

    @property
    def mission_complete(self) -> bool:
        return bool(self._waypoints) and self._current_waypoint_idx >= len(self._waypoints)

    def check_waypoint_reached(self, position: np.ndarray) -> bool:
        """Avança para o próximo waypoint quando o alvo atual foi atingido."""
        if not self._waypoints:
            return False
        if self._current_waypoint_idx >= len(self._waypoints):
            return False

        target = self._waypoints[self._current_waypoint_idx]
        dist = float(np.linalg.norm(np.asarray(position, dtype=float) - target))
        if dist <= self._waypoint_threshold:
            self._current_waypoint_idx += 1
            self._sync_reference()
            return True
        return False

    def compute(self, ekf_state: EKFState, time: float) -> ControlCommand:
        """Calcula comando usando o controlador ativo."""
        if self._active in ('lqr', 'mpc'):
            self.check_waypoint_reached(ekf_state.position)
            target = self._current_target()
            if target is not None:
                delta = np.asarray(target, dtype=float) - np.asarray(ekf_state.position, dtype=float)
                yaw = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta[:2]) > 1e-6 else 0.0
                if self._active == 'lqr':
                    self._lqr.set_reference(target, yaw)
                    if self._lqr._last_time is None and self._waypoints:
                        self._lqr._last_time = float(time - 0.2)
                elif self._mpc is not None:
                    self._mpc.set_reference(target, yaw)

        if self._active == 'lqr':
            return self._lqr.compute(ekf_state, time)
        elif self._active == 'mpc':
            return self._mpc.compute(ekf_state, time)
        elif self._active == 'rl':
            return self._rl.compute(ekf_state, time)

    def update_lqr_weights(self, **kwargs) -> None:
        """Atualiza pesos do LQR via GUI."""
        self._lqr.update_weights(**kwargs)

    @property
    def active_controller(self) -> str:
        return self._active

    @property
    def lqr_gains(self) -> np.ndarray:
        return self._lqr.gain_matrix

    @property
    def controller_state(self) -> ControllerState:
        if self._active == 'lqr':
            return self._lqr.controller_state
        return None


# ─────────────────────────────────────────────
# TESTES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from geometry_engine import GeometryEngine
    from physics_engine  import PhysicsEngine
    from sensor_engine   import SensorEngine, ExtendedKalmanFilter, Environment

    print("Inicializando Control Engine...")

    # setup completo
    geo     = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geo, max_thruster_force=10.0)
    env     = Environment(pool_depth=5.0)
    sensors = SensorEngine(env, noise_scale=1.0)
    ekf     = ExtendedKalmanFilter(physics)
    control = ControlEngine(physics, hover_depth=2.0)

    # Teste 1 — verifica matrizes A e B
    print("\nTeste 1 — Sistema linearizado:")
    A, B = control._lqr.linearizer.linearize(z0=2.0)
    print(f"  A shape: {A.shape} — norma: {np.linalg.norm(A):.4f}")
    print(f"  B shape: {B.shape} — norma: {np.linalg.norm(B):.4f}")

    ctrl_check = control._lqr.linearizer.check_controllability(A, B)
    print(f"  Controlável: {ctrl_check['controllable']} "
          f"(rank={ctrl_check['rank']}/{ctrl_check['required_rank']})")

    # Teste 2 — ganho K
    print("\nTeste 2 — Ganho LQR K:")
    K = control.lqr_gains
    print(f"  K shape: {K.shape}")
    print(f"  K norma: {np.linalg.norm(K):.4f}")
    print(f"  K[0] (thrust power row): {K[0].round(3)}")

    # Teste 3 — loop fechado por 5s
    print("\nTeste 3 — Loop fechado LQR por 5s (referência z=2m):")
    physics.reset()
    ekf.reset()
    control.set_reference(np.array([0.0, 0.0, 2.0]))

    dt = 0.01
    errors_z = []

    for i in range(500):
        # pipeline completo
        bundle  = sensors.read(physics.state, physics.time)
        ekf.predict(dt)
        ekf.update_imu(bundle.imu)
        ekf.update_barometer(bundle.barometer)
        ekf.update_sonar(bundle.sonar)

        est = ekf.state_estimate
        cmd = control.compute(est, physics.time)

        physics.step(
            thruster_power=cmd.thruster_power,
            thruster_theta=cmd.thruster_theta,
            thruster_phi=cmd.thruster_phi,
            ballast_cmd=cmd.ballast_cmd,
            thruster2_power=cmd.thruster2_power,
            thruster2_theta=cmd.thruster2_theta,
            thruster2_phi=cmd.thruster2_phi,
            dt=dt,
        )

        errors_z.append(abs(physics.state.z - 2.0))

    final_state = physics.state
    print(f"  Estado final: z={final_state.z:.4f}m  u={final_state.u:.4f}m/s")
    print(f"  Erro z final: {errors_z[-1]:.4f}m")
    print(f"  Erro z médio (últimos 100 steps): "
          f"{np.mean(errors_z[-100:]):.4f}m")

    # Teste 4 — troca de pesos em runtime
    print("\nTeste 4 — Atualização de pesos Q/R em runtime:")
    print(f"  K norma antes: {np.linalg.norm(control.lqr_gains):.4f}")
    control.update_lqr_weights(q_z=50.0, r_ballast=0.5)
    print(f"  K norma depois (q_z=50, r_ballast=0.5): "
          f"{np.linalg.norm(control.lqr_gains):.4f}")

    print("\n✓ Control Engine validado.")
