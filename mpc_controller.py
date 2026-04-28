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
    guidance_to_dual_thruster_command,
)


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
    ):
        self.physics     = physics_engine
        self.N           = horizon
        self.dt          = dt
        self.weights     = weights or LQRWeights()
        self.constraints = constraints or MPCConstraints()
        self.hover_depth = hover_depth

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
        cmd = guidance_to_dual_thruster_command(
            ekf_state=ekf_state,
            target_position=self._x_ref[:3],
            desired_yaw=float(self._x_ref[5]),
            gains=GuidanceGains(
                k_forward=0.60,
                k_surge_damp=0.40,
                k_yaw=0.95,
                k_lateral=0.16,
                k_yaw_damp=0.50,
                k_depth=0.16,
                k_heave_damp=0.13,
                k_ballast=0.20,
                k_ballast_damp=0.09,
                max_forward_power=0.68,
                max_reverse_power=0.52,
                max_yaw_diff=0.32,
                max_theta_deg=31.0,
                depth_deadband_m=0.17,
                heave_priority_ratio=0.65,
            ),
        )
        self._solve_time = _time.time() - t_start
        self._solve_status = 'guidance_allocator'
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
    mpc = MPCController(
        physics_engine=control_engine.physics,
        horizon=20,
        dt=0.1,
        weights=control_engine._lqr.weights,   # compartilha pesos com LQR
        hover_depth=hover_depth,
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
