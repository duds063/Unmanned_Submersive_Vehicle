"""
USV Digital Twin — Módulo 6: Mission Engine
============================================
Gerencia episódios de treinamento com curriculum progressivo.

Curriculum:
    Fase 1 — ambiente vazio, waypoint único próximo, sem ruído
    Fase 2 — obstáculos estáticos, 3 waypoints, ruído parcial
    Fase 3 — obstáculos dinâmicos (random walk), 5 waypoints,
              ruído real + domain randomization em D(ν)

Condições de término de episódio:
    - Missão completa (todos waypoints atingidos)
    - Colisão (sonar hit < 0.3m)
    - Timeout (max_steps atingido)
    - Saiu dos limites do ambiente

Domain Randomization:
    - Coeficientes de arrasto D(ν) variam ±30% por episódio
    - Ruído de sensor varia entre episódios
    - Massa dos componentes varia ±10%
    - Fill rate do lastro varia ±20%
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque
from enum import Enum

from geometry_engine import GeometryEngine
from physics_engine  import PhysicsEngine, ComponentMasses
from sensor_engine   import SensorEngine, ExtendedKalmanFilter, Environment, Obstacle
from control_engine  import ControlEngine
from mpc_controller  import integrate_mpc
from rl_controller   import HRLController, integrate_rl


# ─────────────────────────────────────────────
# ENUMS E CONSTANTES
# ─────────────────────────────────────────────

class EpisodeTermination(Enum):
    MISSION_COMPLETE = "mission_complete"
    COLLISION        = "collision"
    TIMEOUT          = "timeout"
    OUT_OF_BOUNDS    = "out_of_bounds"
    RUNNING          = "running"


class CurriculumPhase(Enum):
    PHASE_1 = 1   # ambiente vazio, waypoint único, sem ruído
    PHASE_2 = 2   # obstáculos estáticos, 3 waypoints, ruído parcial
    PHASE_3 = 3   # obstáculos dinâmicos, 5 waypoints, ruído real


COLLISION_THRESHOLD = 0.3    # m — distância mínima ao obstáculo
MAX_STEPS = {
    CurriculumPhase.PHASE_1: 1000,
    CurriculumPhase.PHASE_2: 2000,
    CurriculumPhase.PHASE_3: 3000,
}
PHASE_ADVANCE_THRESHOLD = -2.0   # recompensa média mínima para avançar fase


# ─────────────────────────────────────────────
# OBSTÁCULO DINÂMICO — RANDOM WALK
# ─────────────────────────────────────────────

@dataclass
class DynamicObstacle:
    """
    Obstáculo com movimento de random walk browniano.
    Posição atualizada a cada step com perturbação gaussiana.
    """
    position:   np.ndarray
    radius:     float
    velocity:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    speed_max:  float      = 0.3    # m/s
    bounds_min: np.ndarray = field(default_factory=lambda: np.array([-20, -20, 0.5]))
    bounds_max: np.ndarray = field(default_factory=lambda: np.array([ 20,  20, 9.5]))

    def step(self, dt: float, rng: np.random.Generator) -> None:
        """Atualiza posição com random walk + reflexão nos limites."""
        # perturbação browniana
        noise = rng.normal(0, 0.1, 3)
        self.velocity += noise
        speed = np.linalg.norm(self.velocity)
        if speed > self.speed_max:
            self.velocity *= self.speed_max / speed

        self.position += self.velocity * dt

        # reflexão nos limites — inverte velocidade ao bater nas bordas
        for i in range(3):
            if self.position[i] < self.bounds_min[i]:
                self.position[i] = self.bounds_min[i]
                self.velocity[i] = abs(self.velocity[i])
            elif self.position[i] > self.bounds_max[i]:
                self.position[i] = self.bounds_max[i]
                self.velocity[i] = -abs(self.velocity[i])

    def to_obstacle(self) -> Obstacle:
        return Obstacle(position=self.position.copy(), radius=self.radius)


# ─────────────────────────────────────────────
# GERADOR DE EPISÓDIOS
# ─────────────────────────────────────────────

class EpisodeGenerator:
    """
    Gera configurações de episódio conforme fase do curriculum.
    Waypoints e obstáculos gerados aleatoriamente dentro dos limites.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate_waypoints(
        self,
        phase:    CurriculumPhase,
        start:    np.ndarray,
    ) -> List[np.ndarray]:
        """Gera waypoints progressivamente mais distantes e complexos."""

        if phase == CurriculumPhase.PHASE_1:
            # waypoint único a 2-4m do início
            offset = self.rng.uniform([1.5, -1.0, -0.5], [4.0, 1.0, 0.5])
            return [start + offset]

        elif phase == CurriculumPhase.PHASE_2:
            # 3 waypoints formando trajetória simples
            wps = []
            pos = start.copy()
            for _ in range(3):
                offset = self.rng.uniform([2.0, -2.0, -1.0], [5.0, 2.0, 1.0])
                pos = pos + offset
                pos[2] = np.clip(pos[2], 0.5, 4.5)   # mantém profundidade válida
                wps.append(pos.copy())
            return wps

        else:  # PHASE_3
            # 5 waypoints com trajetória complexa em 3D
            wps = []
            pos = start.copy()
            for _ in range(5):
                offset = self.rng.uniform([1.5, -3.0, -1.5], [4.0, 3.0, 1.5])
                pos = pos + offset
                pos[2] = np.clip(pos[2], 0.5, 4.5)
                wps.append(pos.copy())
            return wps

    def generate_static_obstacles(
        self,
        phase:     CurriculumPhase,
        waypoints: List[np.ndarray],
        start:     np.ndarray,
    ) -> List[Obstacle]:
        """Gera obstáculos estáticos que não bloqueiam completamente o caminho."""

        if phase == CurriculumPhase.PHASE_1:
            return []

        obstacles = []
        n_obs = 2 if phase == CurriculumPhase.PHASE_2 else 4

        for _ in range(n_obs):
            # posição aleatória no ambiente, longe dos waypoints
            for _ in range(10):   # 10 tentativas para posição válida
                pos = self.rng.uniform(
                    [start[0], -5.0, 0.5],
                    [start[0] + 8.0, 5.0, 4.5]
                )
                # verifica distância mínima de waypoints e início
                min_dist = min(
                    np.linalg.norm(pos - wp) for wp in waypoints + [start]
                )
                if min_dist > 1.5:
                    radius = self.rng.uniform(0.3, 0.7)
                    obstacles.append(Obstacle(position=pos, radius=radius))
                    break

        return obstacles

    def generate_dynamic_obstacles(
        self,
        phase:  CurriculumPhase,
        start:  np.ndarray,
    ) -> List[DynamicObstacle]:
        """Gera obstáculos dinâmicos apenas na fase 3."""

        if phase != CurriculumPhase.PHASE_3:
            return []

        n_dynamic = self.rng.integers(1, 4)
        obstacles  = []

        for _ in range(n_dynamic):
            pos = self.rng.uniform(
                [start[0] + 1.0, -4.0, 0.5],
                [start[0] + 7.0,  4.0, 4.5]
            )
            radius = self.rng.uniform(0.2, 0.5)
            vel    = self.rng.normal(0, 0.1, 3)
            obstacles.append(DynamicObstacle(
                position=pos,
                radius=radius,
                velocity=vel,
            ))

        return obstacles


# ─────────────────────────────────────────────
# DOMAIN RANDOMIZATION
# ─────────────────────────────────────────────

class DomainRandomizer:
    """
    Randomiza parâmetros físicos entre episódios.
    Fase 3 apenas — fases 1 e 2 usam parâmetros nominais.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def randomize(
        self,
        physics: PhysicsEngine,
        sensors: SensorEngine,
        phase:   CurriculumPhase,
    ) -> None:
        """Aplica randomização nos parâmetros do sistema."""

        if phase != CurriculumPhase.PHASE_3:
            # reseta para valores nominais
            self._reset_to_nominal(physics, sensors)
            return

        # ── arrasto D(ν) ±30% ────────────────
        coeff = physics.coeff
        scale = self.rng.uniform(0.7, 1.3)

        # modifica coeficientes de arrasto diretamente
        coeff.X_uu *= scale
        coeff.Y_vv *= self.rng.uniform(0.7, 1.3)
        coeff.Z_ww *= self.rng.uniform(0.7, 1.3)
        coeff.M_qq *= self.rng.uniform(0.7, 1.3)
        coeff.N_rr *= self.rng.uniform(0.7, 1.3)

        # ── massa dos componentes ±10% ────────
        comp = physics.components
        comp.battery_lipo  *= self.rng.uniform(0.9, 1.1)
        comp.thruster_motor *= self.rng.uniform(0.9, 1.1)

        # ── fill rate do lastro ±20% ──────────
        physics.ballast.fill_rate *= self.rng.uniform(0.8, 1.2)

        # ── ruído de sensor ──────────────────
        sensors.set_noise_scale(self.rng.uniform(0.5, 1.5))

    def _reset_to_nominal(
        self,
        physics: PhysicsEngine,
        sensors: SensorEngine,
    ) -> None:
        """Reseta parâmetros para valores nominais."""
        sensors.set_noise_scale(0.0 if True else 1.0)


# ─────────────────────────────────────────────
# EPISÓDIO
# ─────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Resultado de um episódio de treinamento."""
    termination:       EpisodeTermination
    total_steps:       int
    total_reward_n1:   float
    total_reward_n2:   float
    total_reward_n3:   float
    waypoints_reached: int
    total_waypoints:   int
    phase:             CurriculumPhase
    collision:         bool = False

    @property
    def success(self) -> bool:
        return self.termination == EpisodeTermination.MISSION_COMPLETE

    @property
    def completion_rate(self) -> float:
        if self.total_waypoints == 0:
            return 0.0
        return self.waypoints_reached / self.total_waypoints

    def __str__(self) -> str:
        return (f"[{self.phase.name}] "
                f"{self.termination.value} | "
                f"steps={self.total_steps} | "
                f"wp={self.waypoints_reached}/{self.total_waypoints} | "
                f"r_n3={self.total_reward_n3:.2f}")


# ─────────────────────────────────────────────
# MISSION ENGINE
# ─────────────────────────────────────────────

class MissionEngine:
    """
    Gerencia o loop de treinamento completo com curriculum.

    Uso:
        mission = MissionEngine(geo)
        mission.train(n_episodes=1000)
    """

    def __init__(
        self,
        geometry:     GeometryEngine,
        checkpoint_dir: str   = './checkpoints',
        seed:           int   = 42,
        pool_depth:     float = 5.0,
        pool_radius:    float = 20.0,
    ):
        self.geo            = geometry
        self.checkpoint_dir = checkpoint_dir
        self.pool_depth     = pool_depth
        self.pool_radius    = pool_radius
        self.rng            = np.random.default_rng(seed)

        # componentes do sistema
        self.physics  = PhysicsEngine(geometry, max_thruster_force=10.0)
        self.env      = Environment(pool_depth=pool_depth, pool_radius=pool_radius)
        self.sensors  = SensorEngine(self.env, noise_scale=0.0, seed=seed)
        self.ekf      = ExtendedKalmanFilter(self.physics)
        self.control  = ControlEngine(self.physics, hover_depth=pool_depth/2)

        # integra controladores
        integrate_mpc(self.control, hover_depth=pool_depth/2)
        self.hrl = integrate_rl(self.control, checkpoint_dir)

        # geradores auxiliares
        self.ep_gen    = EpisodeGenerator(self.rng)
        self.randomizer = DomainRandomizer(self.rng)
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # curriculum
        self._phase   = CurriculumPhase.PHASE_1
        self._phase_rewards: Dict[CurriculumPhase, deque] = {
            p: deque(maxlen=50) for p in CurriculumPhase
        }

        # histórico
        self.episode_history: List[EpisodeResult] = []
        self._episode_count  = 0
        self._total_steps    = 0

    # ─── Interface pública ───────────────────

    def train(
        self,
        n_episodes:     int   = 1000,
        dt:             float = 0.01,
        log_interval:   int   = 10,
        save_interval:  int   = 50,
    ) -> None:
        """
        Loop principal de treinamento com curriculum.

        Args:
            n_episodes:    número total de episódios
            dt:            timestep de física
            log_interval:  frequência de log em episódios
            save_interval: frequência de checkpoint
        """
        print(f"\n{'='*60}")
        print(f"  TREINAMENTO HRL — {n_episodes} episódios")
        print(f"  Fase inicial: {self._phase.name}")
        print(f"{'='*60}\n")

        # configura fase RL inicial
        self.hrl.set_phase(1)

        for ep in range(n_episodes):
            result = self._run_episode(dt=dt)
            self.episode_history.append(result)
            self._episode_count += 1

            # atualiza métricas de fase
            self._phase_rewards[self._phase].append(result.total_reward_n3)

            # verifica avanço de fase do curriculum
            self._check_curriculum_advance()

            # log
            if (ep + 1) % log_interval == 0:
                self._log_progress(ep + 1, n_episodes)

            # checkpoint
            if (ep + 1) % save_interval == 0:
                self.hrl.save_checkpoint(self._phase.value)

        print("\n✓ Treinamento completo.")
        self.hrl.save_checkpoint(self._phase.value)

    def run_inference(
        self,
        waypoints: List[np.ndarray],
        dt:        float = 0.01,
        max_steps: int   = 3000,
    ) -> EpisodeResult:
        """Roda episódio em modo inferência pura — sem treinamento."""
        self.hrl.set_phase(0)
        self.hrl.set_waypoints(waypoints)
        return self._run_episode(dt=dt, training=False, max_steps=max_steps)

    # ─── Loop de episódio ────────────────────

    def _run_episode(
        self,
        dt:        float = 0.01,
        training:  bool  = True,
        max_steps: int   = None,
    ) -> EpisodeResult:
        """Executa um episódio completo."""

        max_s = max_steps or MAX_STEPS[self._phase]

        # ── Setup do episódio ─────────────────
        start_pos   = np.array([0.0, 0.0, self.pool_depth / 2])
        waypoints   = self.ep_gen.generate_waypoints(self._phase, start_pos)
        static_obs  = self.ep_gen.generate_static_obstacles(
            self._phase, waypoints, start_pos
        )
        self.dynamic_obstacles = self.ep_gen.generate_dynamic_obstacles(
            self._phase, start_pos
        )

        # configura ambiente
        self._rebuild_environment(static_obs)

        # domain randomization
        self.randomizer.randomize(self.physics, self.sensors, self._phase)

        # configura ruído de sensor por fase
        noise_by_phase = {
            CurriculumPhase.PHASE_1: 0.0,
            CurriculumPhase.PHASE_2: 0.5,
            CurriculumPhase.PHASE_3: 1.0,
        }
        if self._phase != CurriculumPhase.PHASE_3:
            self.sensors.set_noise_scale(noise_by_phase[self._phase])

        # reseta sistemas com profundidade inicial correta
        from physics_engine import VehicleState
        initial_state = VehicleState(z=self.pool_depth / 2)
        self.physics.reset(initial_state)
        self.ekf.reset(np.concatenate([initial_state.eta, initial_state.nu]))
        self.hrl.set_waypoints(waypoints)

        # acumuladores
        total_r_n1 = 0.0
        total_r_n2 = 0.0
        total_r_n3 = 0.0
        step       = 0
        termination = EpisodeTermination.RUNNING
        last_cmd   = None
        mpc_counter = 0

        # ── Loop do episódio ──────────────────
        while step < max_s and termination == EpisodeTermination.RUNNING:

            # atualiza obstáculos dinâmicos
            for dyn_obs in self.dynamic_obstacles:
                dyn_obs.step(dt, self.rng)
            if self.dynamic_obstacles:
                self._update_dynamic_obstacles()

            # sensoriamento
            bundle = self.sensors.read(self.physics.state, self.physics.time)

            # EKF
            self.ekf.predict(dt)
            self.ekf.update_imu(bundle.imu)
            self.ekf.update_barometer(bundle.barometer)
            self.ekf.update_sonar(bundle.sonar)
            est = self.ekf.state_estimate

            # ação HRL
            cmd = self.hrl.compute(
                ekf_state=est,
                imu_reading=bundle.imu,
                sonar_readings=bundle.sonar,
                time=self.physics.time,
                training=training,
            )

            # física
            self.physics.step(
                thruster_power=cmd.thruster_power,
                thruster_theta=cmd.thruster_theta,
                thruster_phi=cmd.thruster_phi,
                ballast_cmd=cmd.ballast_cmd,
                dt=dt,
            )

            step += 1
            self._total_steps += 1

            # ── Condições de término ──────────
            pos = np.array([
                self.physics.state.x,
                self.physics.state.y,
                self.physics.state.z,
            ])

            # colisão
            if self._check_collision(bundle.sonar):
                termination = EpisodeTermination.COLLISION
                break

            # saiu dos limites
            if self._check_out_of_bounds(pos):
                termination = EpisodeTermination.OUT_OF_BOUNDS
                break

            # missão completa
            if self.hrl.n3.mission_complete:
                termination = EpisodeTermination.MISSION_COMPLETE
                break

        else:
            termination = EpisodeTermination.TIMEOUT

        # ── Atualização PPO ───────────────────
        if training:
            _, _, last_val = self.hrl.n1.network.act(
                self.hrl.n1.get_observation(bundle.imu, est)
            )
            metrics = self.hrl.update_networks({
                'n1': last_val,
                'n2': 0.0,
                'n3': 0.0,
            })

        return EpisodeResult(
            termination=termination,
            total_steps=step,
            total_reward_n1=total_r_n1,
            total_reward_n2=total_r_n2,
            total_reward_n3=total_r_n3,
            waypoints_reached=self.hrl.n3.current_wp_idx,
            total_waypoints=len(waypoints),
            phase=self._phase,
            collision=(termination == EpisodeTermination.COLLISION),
        )

    # ─── Curriculum ──────────────────────────

    def _check_curriculum_advance(self) -> None:
        """Verifica se deve avançar para próxima fase do curriculum."""
        rewards = self._phase_rewards[self._phase]

        if len(rewards) < 20:
            return

        mean_reward = np.mean(list(rewards))

        if mean_reward > PHASE_ADVANCE_THRESHOLD:
            if self._phase == CurriculumPhase.PHASE_1:
                self._advance_to_phase(CurriculumPhase.PHASE_2)
            elif self._phase == CurriculumPhase.PHASE_2:
                self._advance_to_phase(CurriculumPhase.PHASE_3)

    def _advance_to_phase(self, new_phase: CurriculumPhase) -> None:
        """Avança para nova fase — salva checkpoint e ajusta RL."""
        print(f"\n{'='*40}")
        print(f"  CURRICULUM: {self._phase.name} → {new_phase.name}")
        print(f"  Recompensa média: {np.mean(self._phase_rewards[self._phase]):.3f}")
        print(f"{'='*40}\n")

        self.hrl.save_checkpoint(self._phase.value)
        self._phase = new_phase

        # avança fase do RL junto com curriculum
        rl_phase = new_phase.value
        self.hrl.set_phase(rl_phase)

    # ─── Ambiente ────────────────────────────

    def _rebuild_environment(self, static_obstacles: List[Obstacle]) -> None:
        """Reconstrói ambiente com novos obstáculos."""
        self.env.obstacles.clear()

        # paredes do pool
        self.env._setup_boundaries(self.pool_depth, self.pool_radius)

        # obstáculos estáticos
        for obs in static_obstacles:
            self.env.obstacles.append(obs)

    def _update_dynamic_obstacles(self) -> None:
        """Atualiza posições dos obstáculos dinâmicos no ambiente."""
        # remove obstáculos dinâmicos antigos (mantém paredes + estáticos)
        n_static = len(self.env.obstacles) - len(self.dynamic_obstacles)
        self.env.obstacles = self.env.obstacles[:n_static]

        # adiciona posições atualizadas
        for dyn in self.dynamic_obstacles:
            self.env.obstacles.append(dyn.to_obstacle())

    # ─── Verificações ────────────────────────

    def _check_collision(self, sonar_readings) -> bool:
        """Verifica colisão por leituras do sonar."""
        for reading in sonar_readings:
            # só considera hits válidos com distância positiva
            if reading.hit and reading.distance > 0 and \
               reading.distance < COLLISION_THRESHOLD:
                return True
        return False

    def _check_out_of_bounds(self, position: np.ndarray) -> bool:
        """Verifica se saiu dos limites do ambiente."""
        return (
            position[2] < 0 or
            position[2] > self.pool_depth or
            np.sqrt(position[0]**2 + position[1]**2) > self.pool_radius
        )

    # ─── Log ─────────────────────────────────

    def _log_progress(self, episode: int, total: int) -> None:
        """Imprime progresso do treinamento."""
        recent = self.episode_history[-10:]

        success_rate = sum(1 for r in recent if r.success) / len(recent)
        collision_rate = sum(1 for r in recent if r.collision) / len(recent)
        mean_wp = np.mean([r.completion_rate for r in recent])
        mean_steps = np.mean([r.total_steps for r in recent])

        print(f"Ep {episode:4d}/{total} | "
              f"Fase: {self._phase.name} | "
              f"Sucesso: {success_rate:.0%} | "
              f"Colisão: {collision_rate:.0%} | "
              f"WP: {mean_wp:.0%} | "
              f"Steps: {mean_steps:.0f}")

        self.hrl.print_metrics()

    def summary(self) -> Dict:
        """Retorna resumo do treinamento."""
        if not self.episode_history:
            return {}

        return {
            'total_episodes':  self._episode_count,
            'total_steps':     self._total_steps,
            'current_phase':   self._phase.name,
            'success_rate':    np.mean([r.success for r in self.episode_history]),
            'collision_rate':  np.mean([r.collision for r in self.episode_history]),
            'mean_completion': np.mean([r.completion_rate for r in self.episode_history]),
        }


# ─────────────────────────────────────────────
# TESTES
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time

    print("Inicializando Mission Engine...")

    geo     = GeometryEngine(L=0.8, D=0.1)
    mission = MissionEngine(geo, checkpoint_dir='./checkpoints_test')

    # Teste 1 — gerador de episódios
    print("\nTeste 1 — Gerador de episódios:")
    rng = np.random.default_rng(42)
    gen = EpisodeGenerator(rng)
    start = np.array([0.0, 0.0, 2.5])

    for phase in CurriculumPhase:
        wps = gen.generate_waypoints(phase, start)
        obs = gen.generate_static_obstacles(phase, wps, start)
        dyn = gen.generate_dynamic_obstacles(phase, start)
        print(f"  {phase.name}: {len(wps)} waypoints, "
              f"{len(obs)} obstáculos estáticos, "
              f"{len(dyn)} dinâmicos")

    # Teste 2 — obstáculo dinâmico
    print("\nTeste 2 — Random walk de obstáculo dinâmico (10 steps):")
    dyn_obs = DynamicObstacle(
        position=np.array([3.0, 0.0, 2.0]),
        radius=0.5,
    )
    for i in range(10):
        dyn_obs.step(0.1, rng)
    print(f"  Posição inicial: [3.0, 0.0, 2.0]")
    print(f"  Posição final:   {dyn_obs.position.round(3)}")

    # Teste 3 — episódio curto fase 1
    print("\nTeste 3 — Episódio fase 1 (100 steps):")
    t0 = _time.time()
    result = mission._run_episode(dt=0.01, training=True,  max_steps=100)
    elapsed = _time.time() - t0
    print(f"  {result}")
    print(f"  Tempo: {elapsed:.2f}s ({100/elapsed:.0f} steps/s)")

    # Teste 4 — 5 episódios completos fase 1
    print("\nTeste 4 — 5 episódios fase 1:")
    t0 = _time.time()
    for i in range(5):
        result = mission._run_episode(dt=0.01, training=True)
        print(f"  Ep {i+1}: {result}")
    elapsed = _time.time() - t0
    print(f"  Tempo total: {elapsed:.1f}s")

    # Teste 5 — sumário
    print("\nTeste 5 — Sumário:")
    s = mission.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    print("\n✓ Mission Engine validado.")
