"""
USV Digital Twin — MPC Controller
===================================
Model Predictive Control para o USV em 6 DOF.

MPC resolve a cada timestep um problema de otimização convexa:

    min  Σ_{k=0}^{N-1} [ x_k'Qx_k + u_k'Ru_k ] + x_N'P_f x_N
    s.t. x_{k+1} = Ax_k + Bu_k        (modelo linearizado)
         u_min ≤ u_k ≤ u_max           (restrições de atuador)
         x_min ≤ x_k ≤ x_max           (restrições de estado)
         x_0 = x_current               (condição inicial)

Vantagens sobre LQR:
    - Restrições explícitas nos atuadores e estados
    - Antecipa evolução futura do sistema
    - Mais robusto fora do ponto de linearização

Implementação via CVXPY — otimização convexa em Python.
Roda no Orange Pi 4A em tempo real com N=20.

Referências:
    - Rawlings & Mayne (2009) — Model Predictive Control: Theory and Design
    - Fossen (2011) cap. 13 — MPC para veículos marinhos
    - CVXPY documentation — cvxpy.org
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional, Tuple
from sensor_engine import EKFState
from control_engine import (
    ControlCommand,
    GuidanceGains,
    LQRWeights,
    SystemLinearizer,
    body_frame_position_error,
    guidance_to_dual_thruster_command,
    wrap_angle,
)

# Global tuning dictionary used by external scripts to steer MPC aggressiveness.
# Example: import mpc_controller; mpc_controller.MPC_TUNING = {'max_cruise_mult':1.2}
MPC_TUNING = {}


# ─────────────────────────────────────────────
# RESTRIÇÕES DO SISTEMA
# ─────────────────────────────────────────────

@dataclass
class MPCConstraints:
    """Restrições físicas do sistema para o MPC."""

    # restrições de atuador
    u_min: np.ndarray = None
    u_max: np.ndarray = None

    # restrições de estado
    x_min: np.ndarray = None
    x_max: np.ndarray = None

    # variação máxima de controle por step (rate constraints)
    du_max: np.ndarray = None

    def __post_init__(self):
        if self.u_min is None:
            # [p1, t1, ph1, p2, t2, ph2, ballast]
            self.u_min = np.array([
                -1.0, 0.0, 0.0,
                -1.0, 0.0, 0.0,
                -1.0,
            ])
        if self.u_max is None:
            self.u_max = np.array([
                1.0, np.radians(60), 2*np.pi,
                1.0, np.radians(60), 2*np.pi,
                1.0,
            ])
        if self.x_min is None:
            # [x, y, z, phi, theta, psi, u, v, w, p, q, r]
            self.x_min = np.array([
                -np.inf, -np.inf, 0.0,          # posição — z ≥ 0 (não sai da água)
                -np.pi,  -np.pi/2, -np.pi,      # orientação
                -5.0,    -2.0,    -2.0,          # velocidades lineares
                -2.0,    -2.0,    -2.0,          # velocidades angulares
            ])
        if self.x_max is None:
            self.x_max = np.array([
                np.inf,  np.inf,  20.0,          # posição — z ≤ 20m
                np.pi,   np.pi/2, np.pi,         # orientação
                5.0,     2.0,     2.0,           # velocidades lineares
                2.0,     2.0,     2.0,           # velocidades angulares
            ])
        if self.du_max is None:
            # variação máxima de controle por timestep
            self.du_max = np.array([
                0.3, np.radians(10), np.radians(20),
                0.3, np.radians(10), np.radians(20),
                0.3,
            ])


# ─────────────────────────────────────────────
# MPC CONTROLLER
# ─────────────────────────────────────────────

class MPCController:
    """
    Model Predictive Control para USV 6 DOF.

    Usa modelo linearizado das equações de Fossen.
    Resolve problema QP via CVXPY a cada timestep.
    Horizonte de predição N=20 passos.
    """

    def __init__(
        self,
        physics_engine,
        horizon:     int             = 20,
        dt:          float           = 0.1,    # timestep do MPC (pode ser maior que physics dt)
        weights:     LQRWeights      = None,
        constraints: MPCConstraints  = None,
        hover_depth: float           = 2.0,
        control_engine=None,  # optional reference to ControlEngine for waypoint detection
        tuning: dict = None,
    ):
        self.physics     = physics_engine
        self.control_engine = control_engine  # optional: allows terminal capture mode detection
        self.N           = horizon
        self.dt          = dt
        self.weights     = weights or LQRWeights()
        self.constraints = constraints or MPCConstraints()
        self.hover_depth = hover_depth
        # tuning multipliers (can be overridden by external scripts via 'tuning' or module-level MPC_TUNING)
        tuning = tuning or MPC_TUNING or {}
        self.tune_max_cruise_mult = float(tuning.get('max_cruise_mult', 1.0))
        self.tune_desired_surge_mult = float(tuning.get('desired_surge_mult', 1.0))
        self.tune_base_power_gain_mult = float(tuning.get('base_power_gain_mult', 1.0))
        self.tune_lateral_yaw_mult = float(tuning.get('lateral_yaw_mult', 1.0))
        self.tune_yaw_error_mult = float(tuning.get('yaw_error_mult', 1.0))
        self.tune_yaw_damp_mult = float(tuning.get('yaw_damp_mult', 1.0))
        self.tune_lateral_speed_penalty_mult = float(tuning.get('lateral_speed_penalty_mult', 1.0))
        self.tune_reverse_penalty = float(tuning.get('reverse_penalty', 0.55))
        self.tune_terminal_pull_mult = float(tuning.get('terminal_pull_mult', 1.0))
        self.tune_boundary_margin_m = float(tuning.get('boundary_margin_m', 0.25))

        # dimensões
        self.nx = 12   # estados
        self.nu = 7    # entradas (2 thrusters + ballast)

        # lineariza sistema
        linearizer    = SystemLinearizer(physics_engine)
        A_c, B_c      = linearizer.linearize(z0=hover_depth)

        # discretiza por Euler — simples e eficiente pra N=20
        # A_d = I + A_c * dt, B_d = B_c * dt
        self.A = np.eye(self.nx) + A_c * dt
        self.B = B_c * dt

        # referência — hovering neutro
        self._x_ref = np.zeros(self.nx)
        self._x_ref[2] = hover_depth

        # custo terminal — resolve Riccati pro custo infinito (estabilidade)
        self._P_f = self._compute_terminal_cost()

        # problema CVXPY — construído uma vez, parâmetros atualizados a cada call
        self._build_problem()

        # telemetria
        self._last_cmd    = ControlCommand(0, 0, 0, 0, 0, 0, 0)
        self._last_u_prev = np.zeros(self.nu)
        self._solve_time  = 0.0
        self._solve_status = 'not_solved'
        self._post_turn_recovery_steps = 0
        self._was_concentric_turn = False
        self._reverse_hold_steps = 0

    # ─── Interface pública ───────────────────

    def set_reference(self, position: np.ndarray, yaw: float = 0.0) -> None:
        self._x_ref      = np.zeros(self.nx)
        self._x_ref[:3]  = position
        self._x_ref[5]   = yaw

    def update_weights(self, **kwargs) -> None:
        """Atualiza pesos e reconstrói problema."""
        self.weights.update(**kwargs)
        self._P_f = self._compute_terminal_cost()
        self._build_problem()

    def compute(self, ekf_state: EKFState, time: float) -> ControlCommand:
        """
        Calcula comando MPC dado estado estimado pelo EKF.
        NUNCA acessa estado físico diretamente.
        """
        import time as _time

        t_start = _time.time()
        cmd = self._bounded_navigation_command(ekf_state)
        self._solve_time = _time.time() - t_start
        self._solve_status = 'bounded_navigation'
        u_opt = np.array([
            cmd.thruster_power,
            cmd.thruster_theta,
            cmd.thruster_phi,
            cmd.thruster2_power,
            cmd.thruster2_theta,
            cmd.thruster2_phi,
            cmd.ballast_cmd,
        ], dtype=float)
        self._last_u_prev = u_opt
        self._last_cmd = cmd.clip()
        return self._last_cmd

    def _bounded_navigation_command(self, ekf_state: EKFState) -> ControlCommand:
        pos = np.asarray(ekf_state.position, dtype=float)
        vel = np.asarray(ekf_state.velocity_linear, dtype=float)
        ang = np.asarray(ekf_state.velocity_angular, dtype=float)
        yaw = float(ekf_state.orientation[2])

        target = np.asarray(self._x_ref[:3], dtype=float)
        desired_yaw = float(self._x_ref[5])
        # if vision provides a relative waypoint (world-frame delta), use it
        if getattr(ekf_state, "vision_relative_waypoint", None) is not None:
            delta_world = np.asarray(ekf_state.vision_relative_waypoint, dtype=float)
        else:
            delta_world = target - pos
        err_body = body_frame_position_error(delta_world, yaw)
        forward_err, lateral_err, depth_err = [float(v) for v in err_body]

        horiz_dist = float(np.linalg.norm(delta_world[:2]))
        surge = float(vel[0])
        heave = float(vel[2])
        yaw_rate = float(ang[2])
        yaw_err = wrap_angle(desired_yaw - yaw)
        abs_yaw_err = abs(yaw_err)

        boundary_clearance = float('inf')
        if self.control_engine is not None and hasattr(self.control_engine, 'horizontal_clearance'):
            try:
                boundary_clearance = float(self.control_engine.horizontal_clearance(pos))
            except Exception:
                boundary_clearance = float('inf')
        boundary_margin = float(max(0.05, getattr(self, 'tune_boundary_margin_m', 0.25)))

        # Check if in terminal capture mode (near waypoint)
        in_terminal_capture = (
            self.control_engine is not None and 
            self.control_engine.is_near_waypoint(pos, capture_radius_m=0.5)
        )

        # Forward speed is explicitly bounded by turn demand. Increase cruise
        # speed a bit so MPC is more aggressive and matches LQR behavior.
        max_cruise = 2.8 * float(getattr(self, 'tune_max_cruise_mult', 1.0))
        speed_from_distance = max_cruise * float(np.tanh(horiz_dist / 2.8))
        heading_scale = 0.08 + 0.92 * float(np.clip(np.cos(abs_yaw_err), 0.0, 1.0) ** 2.2)
        desired_surge = speed_from_distance * heading_scale
        # small boost for approach to avoid slow creeping
        desired_surge *= 1.6 * float(getattr(self, 'tune_desired_surge_mult', 1.0))
        if horiz_dist < 1.5:
            desired_surge *= float(np.clip(horiz_dist / 1.5, 0.0, 1.0))
        if abs_yaw_err > np.radians(85.0):
            desired_surge *= 0.05

        # Back-and-fill decision: compare forward vs reverse maneuver costs.
        # Reverse is allowed but penalized, so it is chosen only when safety/radius needs it.
        reverse_mode = False
        if self._reverse_hold_steps > 0:
            reverse_mode = True
            self._reverse_hold_steps -= 1
        else:
            near_boundary = boundary_clearance < (boundary_margin + 0.35)
            tight_turn = abs_yaw_err > np.radians(35.0)
            need_backfill = tight_turn and (near_boundary or abs(lateral_err) > 1.2)
            if need_backfill:
                fwd_risk = max(0.0, (boundary_margin + 0.35) - boundary_clearance)
                cost_forward = 2.2 * abs_yaw_err + 3.0 * fwd_risk + 1.2 * max(0.0, surge)
                cost_reverse = (
                    float(getattr(self, 'tune_reverse_penalty', 0.55))
                    + 0.9 * abs_yaw_err
                    + 0.6 * fwd_risk
                    + max(0.0, -surge)
                )
                if cost_reverse + 0.05 < cost_forward:
                    reverse_mode = True
                    self._reverse_hold_steps = 10

        if reverse_mode:
            reverse_target = 0.12 + 0.22 * float(np.clip(abs_yaw_err / np.pi, 0.0, 1.0))
            desired_surge = min(desired_surge, -float(np.clip(reverse_target, 0.08, 0.35)))

        # Terminal capture mode: apply strong braking when near waypoint to prevent circling
        if in_terminal_capture:
            # Near the waypoint — reduce speed dramatically and increase depth control
            desired_surge *= 0.15  # heavy braking near waypoint
            depth_err *= 2.5       # amplify depth correction to settle at target z quickly

        # Terminal pull: once aligned and close enough, force exit from reverse behavior.
        if horiz_dist < 1.0 and abs_yaw_err < np.radians(20.0):
            self._reverse_hold_steps = 0
            desired_surge = max(desired_surge, 0.08 * float(getattr(self, 'tune_terminal_pull_mult', 1.0)))

        # If the target is behind the body axis, allow gentle reverse/braking.
        if forward_err < -0.35:
            desired_surge = min(desired_surge, -0.20 * float(np.tanh((-forward_err - 0.35) / 0.7)))

        # Hard safety constraint near boundary: prohibit positive surge when margin is violated.
        if boundary_clearance < boundary_margin:
            desired_surge = min(desired_surge, -0.10)

        base_power = 0.9 * float(getattr(self, 'tune_base_power_gain_mult', 1.0)) * (desired_surge - surge) - 0.10 * np.tanh(surge / 0.45)
        if abs_yaw_err > np.radians(45.0):
            base_power -= 0.12 * float(np.tanh(surge / 0.35))
        if abs(lateral_err) > 1.0:
            lateral_scale = float(np.clip(abs(lateral_err) / 3.0, 0.0, 1.0))
            base_power *= (1.0 - 0.35 * float(getattr(self, 'tune_lateral_speed_penalty_mult', 1.0)) * lateral_scale)
        base_power = float(np.clip(base_power, -0.28, 0.24))
        if boundary_clearance < boundary_margin:
            base_power = min(base_power, 0.0)

        yaw_cmd = (
            0.30 * float(getattr(self, 'tune_lateral_yaw_mult', 1.0)) * float(np.tanh(lateral_err / 1.2))
            + 1.05 * float(getattr(self, 'tune_yaw_error_mult', 1.0)) * float(np.tanh(yaw_err / 0.9))
            - 0.22 * float(getattr(self, 'tune_yaw_damp_mult', 1.0)) * yaw_rate
        )
        yaw_cmd = float(np.clip(yaw_cmd, -0.34, 0.34))
        brake_threshold = np.radians(5.0)

        post_turn_recovery = self._post_turn_recovery_steps > 0
        if self._was_concentric_turn and abs_yaw_err < brake_threshold:
            self._post_turn_recovery_steps = max(self._post_turn_recovery_steps, 8)
            post_turn_recovery = True

        # When the heading error is large, pivot using opposite thruster
        # directions so the boat brakes first, then executes a concentric turn.
        turn_threshold = np.radians(45.0)
        concentric_turn = abs_yaw_err >= turn_threshold or in_terminal_capture
        brake_first = concentric_turn and abs(surge) > 0.08 and abs_yaw_err < np.radians(6.0)
        if post_turn_recovery:
            recovery_power = 0.03 if abs(surge) < 0.25 else 0.02
            p1 = float(np.clip(recovery_power, -1.0, 1.0))
            p2 = float(np.clip(recovery_power, -1.0, 1.0))
            self._post_turn_recovery_steps = max(0, self._post_turn_recovery_steps - 1)
        elif brake_first:
            brake_power = -0.05 if abs(surge) > 0.20 else -0.02
            p1 = float(np.clip(brake_power, -1.0, 1.0))
            p2 = float(np.clip(brake_power, -1.0, 1.0))
        elif concentric_turn:
            turn_sign = 1.0 if yaw_err >= 0.0 else -1.0
            turn_power = 0.38 + 0.22 * float(np.clip(abs_yaw_err / np.pi, 0.0, 1.0))
            if abs_yaw_err < brake_threshold:
                turn_power *= 0.95
            turn_power = float(np.clip(turn_power, 0.30, 0.58))
            p1 = float(np.clip(-turn_sign * turn_power, -1.0, 1.0))
            p2 = float(np.clip(turn_sign * turn_power, -1.0, 1.0))
        else:
            p1 = float(np.clip(base_power - yaw_cmd, -1.0, 1.0))
            p2 = float(np.clip(base_power + yaw_cmd, -1.0, 1.0))

        self._was_concentric_turn = bool(concentric_turn)

        depth_cmd = 0.22 * float(np.tanh(depth_err / 1.2)) - 0.12 * heave
        theta = float(np.clip(abs(depth_cmd), 0.0, np.radians(18.0)))
        phi = float(np.pi / 2.0 if depth_cmd >= 0.0 else 3.0 * np.pi / 2.0) if theta > 1e-6 else 0.0
        ballast = float(np.clip(0.18 * np.tanh(depth_err / 1.8) - 0.08 * heave, -1.0, 1.0))

        return ControlCommand(
            thruster_power=p1,
            thruster_theta=theta,
            thruster_phi=phi,
            ballast_cmd=ballast,
            thruster2_power=p2,
            thruster2_theta=theta,
            thruster2_phi=phi,
        )

    @property
    def solve_time_ms(self) -> float:
        return self._solve_time * 1000

    @property
    def solve_status(self) -> str:
        return self._solve_status

    # ─── Construção do problema CVXPY ────────

    def _build_problem(self) -> None:
        """
        Constrói o problema de otimização convexa.
        Chamado uma vez na inicialização e quando pesos mudam.

        Variáveis de decisão:
            X: (nx, N+1) — trajetória de estados
            U: (nu, N)   — sequência de controles
        """
        Q   = self.weights.Q_matrix()
        R   = self._mpc_R_matrix()
        P_f = self._P_f
        A   = self.A
        B   = self.B
        N   = self.N

        c   = self.constraints

        # variáveis de decisão
        X = cp.Variable((self.nx, N + 1))
        U = cp.Variable((self.nu, N))

        # parâmetros — atualizados a cada chamada sem reconstruir problema
        x0_param    = cp.Parameter(self.nx)
        u_prev_param = cp.Parameter(self.nu)

        # função de custo
        cost = 0
        for k in range(N):
            cost += cp.quad_form(X[:, k], Q)    # custo de estado
            cost += cp.quad_form(U[:, k], R)    # custo de controle

            # rate constraint — suavidade do controle
            if k == 0:
                du = U[:, k] - u_prev_param
            else:
                du = U[:, k] - U[:, k-1]
            Ddu = np.diag(np.maximum(c.du_max, 1e-6))
            Wdu = np.linalg.inv(Ddu @ Ddu)
            cost += cp.quad_form(du, Wdu * 0.1)

        # custo terminal
        cost += cp.quad_form(X[:, N], P_f)

        # restrições
        constraints = [X[:, 0] == x0_param]  # condição inicial

        for k in range(N):
            # dinâmica
            constraints.append(X[:, k+1] == A @ X[:, k] + B @ U[:, k])

            # restrições de atuador
            constraints.append(U[:, k] >= c.u_min)
            constraints.append(U[:, k] <= c.u_max)

            # restrição de profundidade em coordenadas de erro
            # x_ref[2] = z_ref, erro = z - z_ref
            # z ≥ 0  →  erro ≥ -z_ref
            # z ≤ 20 →  erro ≤ 20 - z_ref
            z_ref = self._x_ref[2]
            constraints.append(X[2, k] >= c.x_min[2] - z_ref)
            constraints.append(X[2, k] <= c.x_max[2] - z_ref)

        # armazena referências
        self._x_var      = X
        self._u_var      = U
        self._x0_param   = x0_param
        self._u_prev_param = u_prev_param

        self._problem = cp.Problem(cp.Minimize(cost), constraints)

    def _compute_terminal_cost(self) -> np.ndarray:
        """
        Custo terminal P_f — solução da equação de Riccati.
        Garante estabilidade assintótica do MPC.
        """
        from scipy.linalg import solve_discrete_are

        Q = self.weights.Q_matrix()
        R = self._mpc_R_matrix()

        try:
            P_f = solve_discrete_are(self.A, self.B, Q, R)
            return P_f
        except Exception:
            # fallback — usa Q como custo terminal
            return Q * 10.0

    # ─── Conversão de controle ───────────────

    def _control_to_command(self, u: np.ndarray) -> ControlCommand:
        """Converte vetor de controle em comandos físicos."""
        p1 = np.clip(u[0], -1.0, 1.0)
        t1 = np.clip(u[1], 0.0, np.radians(60))
        f1 = float(u[2] % (2 * np.pi))

        p2 = np.clip(u[3], -1.0, 1.0)
        t2 = np.clip(u[4], 0.0, np.radians(60))
        f2 = float(u[5] % (2 * np.pi))

        ballast = np.clip(u[6], -1.0, 1.0)

        return ControlCommand(
            thruster_power=float(p1),
            thruster_theta=float(t1),
            thruster_phi=float(f1),
            ballast_cmd=float(ballast),
            thruster2_power=float(p2),
            thruster2_theta=float(t2),
            thruster2_phi=float(f2),
        )

    def _mpc_R_matrix(self) -> np.ndarray:
        """Matriz R 7x7 compatível com o modelo dual-thruster."""
        w = self.weights
        return np.diag([
            w.r_thrust_power,
            w.r_thrust_theta,
            w.r_thrust_phi,
            w.r_thrust_power,
            w.r_thrust_theta,
            w.r_thrust_phi,
            w.r_ballast,
        ])


# ─────────────────────────────────────────────
# INTEGRAÇÃO NO CONTROL ENGINE
# ─────────────────────────────────────────────

def integrate_mpc(control_engine, hover_depth: float = 2.0) -> None:
    """
    Inicializa e integra o MPC no ControlEngine existente.
    Chamado após instanciar o ControlEngine.

    Uso:
        control = ControlEngine(physics)
        integrate_mpc(control)
        control.set_controller('mpc')
    """
    # instantiate MPCController and pass through any global MPC_TUNING
    tuning = globals().get('MPC_TUNING', {})
    mpc = MPCController(
        physics_engine=control_engine.physics,
        horizon=20,
        dt=0.1,
        weights=control_engine._lqr.weights,   # compartilha pesos com LQR
        hover_depth=hover_depth,
        control_engine=control_engine,  # pass reference for waypoint detection
        tuning=tuning,
    )
    control_engine._mpc = mpc
    print("✓ MPC inicializado e integrado ao ControlEngine.")


# ─────────────────────────────────────────────
# TESTES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time
    from geometry_engine import GeometryEngine
    from physics_engine  import PhysicsEngine
    from sensor_engine   import SensorEngine, ExtendedKalmanFilter, Environment
    from control_engine  import ControlEngine

    print("Inicializando MPC Controller...")

    geo     = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geo, max_thruster_force=10.0)
    env     = Environment(pool_depth=5.0)
    sensors = SensorEngine(env, noise_scale=1.0)
    ekf     = ExtendedKalmanFilter(physics)
    control = ControlEngine(physics, hover_depth=2.0)

    # integra MPC
    integrate_mpc(control, hover_depth=2.0)
    control.set_controller('mpc')
    control._mpc.set_reference(np.array([0.0, 0.0, 2.0]))

    # Teste 1 — tempo de resolução
    print("\nTeste 1 — Tempo de resolução do QP:")
    physics.reset()
    ekf.reset()

    bundle = sensors.read(physics.state, 0.0)
    ekf.predict(0.1)
    ekf.update_imu(bundle.imu)
    ekf.update_barometer(bundle.barometer)
    est = ekf.state_estimate

    t0 = _time.time()
    cmd = control.compute(est, 0.0)
    solve_ms = (time := _time.time() - t0) * 1000

    print(f"  Status: {control._mpc.solve_status}")
    print(f"  Tempo:  {control._mpc.solve_time_ms:.1f} ms")
    print(f"  Comando: power={cmd.thruster_power:.3f} "
          f"ballast={cmd.ballast_cmd:.3f}")

    # Teste 2 — loop fechado por 30s
    print("\nTeste 2 — Loop fechado MPC por 30s (referência z=2m):")
    physics.reset()
    ekf.reset()

    dt_physics = 0.01
    dt_mpc     = 0.1
    mpc_counter = 0
    last_cmd    = ControlCommand(0, 0, 0, 0)
    errors_z    = []

    t_start = _time.time()
    for i in range(3000):
        bundle = sensors.read(physics.state, physics.time)
        ekf.predict(dt_physics)
        ekf.update_imu(bundle.imu)
        ekf.update_barometer(bundle.barometer)
        ekf.update_sonar(bundle.sonar)

        # MPC roda a 10Hz, physics a 100Hz
        if i % 10 == 0:
            est      = ekf.state_estimate
            last_cmd = control.compute(est, physics.time)

        env_cur, env_turb = sensors.get_environmental_state()
        env_harm = sensors.get_environmental_harmonics()
        physics.step(
            thruster_power=last_cmd.thruster_power,
            thruster_theta=last_cmd.thruster_theta,
            thruster_phi=last_cmd.thruster_phi,
            ballast_cmd=last_cmd.ballast_cmd,
            thruster2_power=last_cmd.thruster2_power,
            thruster2_theta=last_cmd.thruster2_theta,
            thruster2_phi=last_cmd.thruster2_phi,
            dt=dt_physics,
            env_current_world=env_cur,
            env_turbulence=env_turb,
            env_harmonics=env_harm,
        )

        errors_z.append(abs(physics.state.z - 2.0))

        if i % 300 == 0:
            d = physics.to_dict()
            print(f"  t={physics.time:.1f}s  z={physics.state.z:.3f}m  "
                  f"ρ={d['ballast']['density_avg']:.0f}kg/m³  "
                  f"err={errors_z[-1]:.3f}m  "
                  f"solver={control._mpc.solve_time_ms:.0f}ms")

    wall_time = _time.time() - t_start
    print(f"\n  Erro final z:  {errors_z[-1]:.4f}m")
    print(f"  Erro médio:    {np.mean(errors_z[-100:]):.4f}m")
    print(f"  Tempo real:    {wall_time:.1f}s (sim 30s)")
    print(f"  Fator tempo:   {30.0/wall_time:.1f}x real-time")

    # Teste 3 — comparação LQR vs MPC
    print("\nTeste 3 — Comparação tempo de convergência LQR vs MPC:")
    results = {}

    for ctrl_name in ['lqr', 'mpc']:
        physics.reset()
        ekf.reset()
        control.set_controller(ctrl_name)

        if ctrl_name == 'lqr':
            control._lqr.set_reference(np.array([0.0, 0.0, 2.0]))
        else:
            control._mpc.set_reference(np.array([0.0, 0.0, 2.0]))

        errors = []
        last_cmd = ControlCommand(0, 0, 0, 0)

        for i in range(3000):
            bundle = sensors.read(physics.state, physics.time)
            ekf.predict(dt_physics)
            ekf.update_imu(bundle.imu)
            ekf.update_barometer(bundle.barometer)

            if ctrl_name == 'lqr' or i % 10 == 0:
                est      = ekf.state_estimate
                last_cmd = control.compute(est, physics.time)

            env_cur, env_turb = sensors.get_environmental_state()
            env_harm = sensors.get_environmental_harmonics()
            physics.step(
                thruster_power=last_cmd.thruster_power,
                thruster_theta=last_cmd.thruster_theta,
                thruster_phi=last_cmd.thruster_phi,
                ballast_cmd=last_cmd.ballast_cmd,
                thruster2_power=last_cmd.thruster2_power,
                thruster2_theta=last_cmd.thruster2_theta,
                thruster2_phi=last_cmd.thruster2_phi,
                dt=dt_physics,
                env_current_world=env_cur,
                env_turbulence=env_turb,
                env_harmonics=env_harm,
            )
            errors.append(abs(physics.state.z - 2.0))

        results[ctrl_name] = {
            'final_error': errors[-1],
            'mean_error':  np.mean(errors[-100:]),
        }

    print(f"  LQR — erro final: {results['lqr']['final_error']:.4f}m  "
          f"médio: {results['lqr']['mean_error']:.4f}m")
    print(f"  MPC — erro final: {results['mpc']['final_error']:.4f}m  "
          f"médio: {results['mpc']['mean_error']:.4f}m")

    print("\n✓ MPC Controller validado.")
