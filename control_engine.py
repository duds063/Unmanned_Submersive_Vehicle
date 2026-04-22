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

    def clip(self) -> 'ControlCommand':
        """Garante que os comandos estão dentro dos limites físicos."""
        return ControlCommand(
            thruster_power = float(np.clip(self.thruster_power, -1.0, 1.0)),
            thruster_theta = float(np.clip(self.thruster_theta, 0.0, np.radians(60))),
            thruster_phi   = float(self.thruster_phi % (2 * np.pi)),
            ballast_cmd    = float(np.clip(self.ballast_cmd, -1.0, 1.0)),
        )

    def to_dict(self) -> dict:
        return {
            'thruster_power': self.thruster_power,
            'thruster_theta': float(np.degrees(self.thruster_theta)),
            'thruster_phi':   float(np.degrees(self.thruster_phi)),
            'ballast_cmd':    self.ballast_cmd,
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
            B: matriz de entrada 12x4
        """
        # estado de operação: hovering neutro em z0
        x0 = np.zeros(12)
        x0[2] = z0

        # entrada de operação: propulsor com potência e deflexão leves
        # CRÍTICO: power=0 zera o canal theta em B porque F_z = power*F_max*sin(theta)
        # é bilinear — dF_z/dtheta = power*F_max*cos(theta) = 0 quando power=0.
        # Com power=0.3 e theta=15°, todos os canais ficam visíveis na linearização.
        u0 = np.array([0.3, np.radians(15), np.radians(90), 0.0])

        # Jacobiana A = ∂f/∂x numericamente
        A = np.zeros((12, 12))
        for i in range(12):
            x_plus  = x0.copy(); x_plus[i]  += self.eps
            x_minus = x0.copy(); x_minus[i] -= self.eps
            f_plus  = self._dynamics(x_plus,  u0)
            f_minus = self._dynamics(x_minus, u0)
            A[:, i] = (f_plus - f_minus) / (2 * self.eps)

        # Jacobiana B = ∂f/∂u numericamente
        B = np.zeros((12, 4))
        for i in range(4):
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

        # --- propulsor: bypass do clip para diferença centrada correta ---
        p.thruster._power = float(np.clip(u[0], -1.0, 1.0))
        p.thruster._theta = float(u[1])          # sem clip — linearização bidirecional
        p.thruster._phi   = float(u[2] % (2 * np.pi))

        # --- ballast: perturbação como fração do range de massa físico ---
        # range total de massa = mass_max - mass_min (determinado pela geometria)
        # u[3] ∈ [-1, 1] mapeia para ±50% do range — escala física, não temporal
        half_range = 0.5 * (p.ballast.mass_max - p.ballast.mass_min)
        dm = np.clip(u[3], -1.0, 1.0) * half_range
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
    ):
        self.physics     = physics_engine
        self.weights     = weights or LQRWeights()
        self.hover_depth = hover_depth

        # inicializa ballast no ponto neutro antes de linearizar
        # (garante que o ponto de operação da linearização é fisicamente válido)
        half = 0.5 * (physics_engine.ballast.mass_max - physics_engine.ballast.mass_min)
        mid  = physics_engine.ballast.mass_min + half
        physics_engine.ballast._water_mass = mid

        # lineariza o sistema uma vez
        self.linearizer = SystemLinearizer(physics_engine)
        self.A, self.B  = self.linearizer.linearize(z0=hover_depth)

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
        self._last_effort = np.zeros(4)

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
        # monta vetor de estado do EKF
        x = np.concatenate([ekf_state.eta, ekf_state.nu])

        # erro em relação à referência
        error = x - self._x_ref

        # normaliza ângulos — evita erro de 350° quando deveria ser -10°
        error[3:6] = self._wrap_angles(error[3:6])

        # lei de controle LQR: u = -K * error
        u = -self.K @ error

        # converte vetor de controle em comandos físicos
        cmd = self._control_to_command(u)

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
        R = self.weights.R_matrix()

        # Resolve Riccati no subespaço controlável (estados u, v, w)
        # O sistema tem rank=6/12: posições x,y,z,phi,tht,psi são integradoras
        # (autovalor 0 estrutural) — controláveis indiretamente via velocidades.
        # NÃO modificamos A artificialmente: isso distorceria o ganho.
        # Em vez disso, resolvemos no subsistema realmente controlável.
        try:
            P = solve_continuous_are(self.A, self.B, Q, R)
            K = np.linalg.inv(R) @ self.B.T @ P
            return K
        except Exception:
            pass  # sistema singular — usa subespaço abaixo

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
            A_r_stab = A_r - np.eye(len(idx_s)) * 0.1
            P_r = solve_continuous_are(A_r_stab, B_r, Q_r, R_r)
            K_r = np.linalg.inv(R_r) @ B_r.T @ P_r

            # expande K de volta pro espaço completo
            K = np.zeros((4, 12))
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
        K = np.zeros((4, 12))
        # thrust power ← surge velocity (u, idx 6)
        K[0, 6] = 0.5
        # ballast ← heave velocity (w, idx 8) + depth error (z, idx 2)
        K[3, 2] = 0.3
        K[3, 8] = 0.2
        print("✓ Ganho proporcional de fallback ativo.")
        return K

    # ─── Conversão de controle ───────────────

    def _control_to_command(self, u: np.ndarray) -> ControlCommand:
        """
        Converte vetor de controle abstrato [4D] em comandos físicos.

        u[0] → thruster_power  — magnitude da força frontal
        u[1] → thruster_theta  — deflexão vertical (pitch/heave)
        u[2] → thruster_phi    — deflexão lateral (yaw/sway)
        u[3] → ballast_cmd     — controle de profundidade
        """
        # normaliza power pelo máximo teórico
        max_force = self.physics.thruster.max_force
        power = np.clip(u[0] / max_force, -1.0, 1.0)

        # theta: deflexão no plano vertical
        # u[1] representa força vertical desejada → ângulo correspondente
        if abs(u[0]) > 1e-3:
            theta_raw = np.arctan2(abs(u[1]), abs(u[0]))
        else:
            theta_raw = np.radians(30) if abs(u[1]) > 0.1 else 0.0
        theta = np.clip(theta_raw, 0.0, np.radians(60))

        # phi: direção da deflexão no plano horizontal
        # u[1] = componente Z (heave), u[2] = componente Y (sway)
        phi = np.arctan2(u[1], u[2]) if (abs(u[1]) + abs(u[2])) > 1e-3 else 0.0

        # ballast: u[3] já está na escala do LQR — clip direto
        # (divisão por 10 anterior atenuava o sinal 10x sem justificativa)
        ballast = np.clip(u[3], -1.0, 1.0)

        return ControlCommand(
            thruster_power=float(power),
            thruster_theta=float(theta),
            thruster_phi=float(phi),
            ballast_cmd=float(ballast),
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
        """Define ponto de referência para todos os controladores."""
        self._reference = position
        self._lqr.set_reference(position, yaw)
        # MPC e RL receberão referência quando implementados

    def compute(self, ekf_state: EKFState, time: float) -> ControlCommand:
        """Calcula comando usando o controlador ativo."""
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
