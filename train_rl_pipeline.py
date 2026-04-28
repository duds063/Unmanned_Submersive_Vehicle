"""
USV Digital Twin - RL Training Pipeline
======================================

Pipeline autonomo para treinar o HRL existente em fases sequenciais:
    Fase 1 -> N1 only
    Fase 2 -> N2 only
    Fase 3 -> N3 only

O script reusa a pilha de simulação já existente no repositório e
salva checkpoints após cada fase.

Uso:
    python train_rl_pipeline.py
    python train_rl_pipeline.py --fresh
    python train_rl_pipeline.py --phases 1 2 3 --phase-steps 4096
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from control_engine import ControlEngine
from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine, VehicleState
from rl_controller import HRLController, integrate_rl, SONAR_RANGE_MAX
from sensor_engine import Environment, ExtendedKalmanFilter, SensorEngine


def _default_waypoints(depth: float) -> List[np.ndarray]:
    return [
        np.array([4.0, 0.0, depth], dtype=float),
        np.array([6.0, 2.0, depth], dtype=float),
        np.array([8.0, -1.5, depth], dtype=float),
    ]


@dataclass
class PipelineConfig:
    phases: Tuple[int, ...] = (1, 2, 3)
    cycles: int = 1
    phase_steps: int = 4096
    episode_steps: int = 256
    dt: float = 0.01
    eval_steps: int = 2000
    hover_depth: float = 5.0
    pool_depth: float = 10.0
    pool_radius: float = 30.0
    noise_scale: float = 0.5
    enable_rayleigh: bool = False
    rayleigh_sigma: float = 0.03
    env_disturbance_scale: float = 1.0
    env_force_gain: float = 1.0
    env_turbulence_gain: float = 1.0
    env_wave_freq: float = 0.8
    env_spectral_enabled: bool = False
    wave_num_harmonics: int = 8
    wave_peak_freq: float = 0.8
    wave_amp_scale: float = 0.02
    wave_hs: float = 0.2
    seed: int = 42
    fresh: bool = False
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./training_runs"
    waypoint_depth: float = 5.0
    waypoint_threshold: float = 0.5
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)


def _to_jsonable(obj):
    """Converte estruturas Python/NumPy para tipos serializáveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def build_stack(config: PipelineConfig):
    np.random.seed(config.seed)

    geometry = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geometry, max_thruster_force=10.0)
    env = Environment(pool_depth=config.pool_depth, pool_radius=config.pool_radius)
    env.add_sphere(np.array([5.0, 0.0, 3.0], dtype=float), radius=1.0)
    sensors = SensorEngine(
        env,
        noise_scale=config.noise_scale,
        rayleigh_sigma=config.rayleigh_sigma,
        enable_rayleigh=config.enable_rayleigh,
        seed=config.seed,
        wave_hs=config.wave_hs,
    )
    sensors.set_environmental_disturbance(
        enabled=config.enable_rayleigh,
        scale=config.env_disturbance_scale,
        rayleigh_sigma=config.rayleigh_sigma,
        spectral=config.env_spectral_enabled,
        wave_num_harmonics=config.wave_num_harmonics,
        wave_peak_freq=config.wave_peak_freq,
        wave_amp_scale=config.wave_amp_scale,
        wave_hs=config.wave_hs,
    )
    ekf = ExtendedKalmanFilter(physics)

    control = ControlEngine(physics, hover_depth=config.hover_depth)
    hrl = integrate_rl(control, config.checkpoint_dir)
    hrl.n3.waypoint_threshold = float(config.waypoint_threshold)

    waypoints = [np.array(wp, dtype=float) for wp in config.waypoints]
    if not waypoints:
        waypoints = _default_waypoints(config.waypoint_depth)
    hrl.set_waypoints(waypoints)

    # configura parâmetros de acoplamento ambiental no physics (ganhos e frequências)
    physics.env_force_gain = float(config.env_force_gain)
    physics.env_turbulence_gain = float(config.env_turbulence_gain)
    physics.env_wave_freq = float(config.env_wave_freq)

    return geometry, physics, sensors, ekf, control, hrl, waypoints


def reset_episode(
    physics: PhysicsEngine,
    ekf: ExtendedKalmanFilter,
    hrl: HRLController,
    waypoints: Sequence[np.ndarray],
    hover_depth: float,
):
    physics.reset(VehicleState(z=hover_depth))

    initial_state = np.zeros(12, dtype=float)
    initial_state[2] = hover_depth
    ekf.reset(initial_state)

    hrl.set_waypoints(list(waypoints))


def _sonar_distances(sonar_readings) -> np.ndarray:
    return np.array(
        [reading.distance if reading.hit else SONAR_RANGE_MAX for reading in sonar_readings],
        dtype=float,
    )


def _active_agent(hrl: HRLController, phase: int):
    if phase == 1:
        return hrl.n1, 0.99, 0.95, "n1"
    if phase == 2:
        return hrl.n2, 0.99, 0.95, "n2"
    if phase == 3:
        return hrl.n3, 0.995, 0.95, "n3"
    raise ValueError(f"Unsupported phase: {phase}")


def _maybe_flush_agent(agent, gamma: float, lam: float, last_value: float):
    if getattr(agent, "frozen", False):
        return None
    if len(agent.buffer.rewards) == 0:
        return None

    agent.buffer.finalize(last_value, gamma=gamma, lam=lam)
    metrics = agent.updater.update(agent.buffer)
    agent.buffer.clear()
    return metrics


def step_hierarchy(
    hrl: HRLController,
    physics: PhysicsEngine,
    sensors: SensorEngine,
    ekf: ExtendedKalmanFilter,
    time_s: float,
    dt: float,
    training: bool,
    done: bool,
):
    bundle = sensors.read(physics.state, time_s)
    ekf.predict(dt)
    ekf.update_imu(bundle.imu)
    ekf.update_barometer(bundle.barometer)
    ekf.update_sonar(bundle.sonar)
    est = ekf.state_estimate

    cmd, info = hrl.compute(
        est,
        bundle.imu,
        bundle.sonar,
        dt,
        training=training,
        forced_done=done,
        return_info=True,
    )

    env_harm = sensors.get_environmental_harmonics()
    physics.step(
        thruster_power=cmd.thruster_power,
        thruster_theta=cmd.thruster_theta,
        thruster_phi=cmd.thruster_phi,
        ballast_cmd=cmd.ballast_cmd,
        thruster2_power=cmd.thruster2_power,
        thruster2_theta=cmd.thruster2_theta,
        thruster2_phi=cmd.thruster2_phi,
        dt=dt,
        env_current_world=sensors.get_environmental_state()[0],
        env_turbulence=sensors.get_environmental_state()[1],
        env_harmonics=env_harm,
    )

    return {
        "bundle": bundle,
        "state": est,
        "cmd": cmd,
        "values": info["values"],
        "rewards": info["rewards"],
    }


def run_phase(
    config: PipelineConfig,
    hrl: HRLController,
    physics: PhysicsEngine,
    sensors: SensorEngine,
    ekf: ExtendedKalmanFilter,
    waypoints: Sequence[np.ndarray],
    phase: int,
):
    agent, gamma, lam, agent_name = _active_agent(hrl, phase)
    hrl.set_phase(phase)

    total_steps = 0
    episode = 0
    phase_metrics: List[Dict] = []
    reward_sums = {"n1": 0.0, "n2": 0.0, "n3": 0.0}
    reward_counts = {"n1": 0, "n2": 0, "n3": 0}

    while total_steps < config.phase_steps:
        reset_episode(physics, ekf, hrl, waypoints, config.hover_depth)
        episode += 1

        for step in range(config.episode_steps):
            done = (step == config.episode_steps - 1) or hrl.n3.mission_complete
            result = step_hierarchy(
                hrl=hrl,
                physics=physics,
                sensors=sensors,
                ekf=ekf,
                time_s=physics.time,
                dt=config.dt,
                training=True,
                done=done,
            )

            total_steps += 1
            for key, value in result["rewards"].items():
                reward_sums[key] += float(value)
                reward_counts[key] += 1

            if total_steps % 32 == 0:
                last_value = float(result["values"][agent_name])
                maybe_metrics = agent.update(last_value)
                if maybe_metrics:
                    phase_metrics.append(maybe_metrics)

            if total_steps >= config.phase_steps:
                break

        flush_metrics = _maybe_flush_agent(agent, gamma, lam, float(result["values"][agent_name]))
        if flush_metrics:
            phase_metrics.append(flush_metrics)

    hrl.save_checkpoint(phase)

    phase_summary = {
        "phase": phase,
        "agent": agent_name,
        "steps": total_steps,
        "episodes": episode,
        "reward_mean": {
            key: (reward_sums[key] / max(1, reward_counts[key])) for key in reward_sums
        },
        "updates": phase_metrics,
    }
    return phase_summary


def evaluate(hrl: HRLController, physics: PhysicsEngine, sensors: SensorEngine, ekf: ExtendedKalmanFilter, waypoints: Sequence[np.ndarray], config: PipelineConfig):
    hrl.set_phase(0)
    reset_episode(physics, ekf, hrl, waypoints, config.hover_depth)

    reached = 0
    total_steps = 0
    min_distance_to_current_wp = float('inf')
    first_wp_step = None

    for _ in range(config.eval_steps):
        bundle = sensors.read(physics.state, physics.time)
        ekf.predict(config.dt)
        ekf.update_imu(bundle.imu)
        ekf.update_barometer(bundle.barometer)
        ekf.update_sonar(bundle.sonar)
        est = ekf.state_estimate

        current_wp = hrl.n3.current_waypoint
        if current_wp is not None:
            dist = float(np.linalg.norm(np.asarray(est.position) - np.asarray(current_wp)))
            min_distance_to_current_wp = min(min_distance_to_current_wp, dist)

        prev_idx = hrl.n3.current_wp_idx
        cmd = hrl.compute(est, bundle.imu, bundle.sonar, config.dt, training=False)
        env_harm = sensors.get_environmental_harmonics()
        physics.step(
            thruster_power=cmd.thruster_power,
            thruster_theta=cmd.thruster_theta,
            thruster_phi=cmd.thruster_phi,
            ballast_cmd=cmd.ballast_cmd,
            thruster2_power=cmd.thruster2_power,
            thruster2_theta=cmd.thruster2_theta,
            thruster2_phi=cmd.thruster2_phi,
            dt=config.dt,
            env_current_world=sensors.get_environmental_state()[0],
            env_turbulence=sensors.get_environmental_state()[1],
            env_harmonics=env_harm,
        )

        total_steps += 1
        if hrl.n3.current_wp_idx > prev_idx:
            reached += 1
            if first_wp_step is None:
                first_wp_step = total_steps
        if hrl.n3.mission_complete:
            break

    return {
        "steps": total_steps,
        "waypoints_reached": reached,
        "mission_complete": hrl.n3.mission_complete,
        "first_wp_step": first_wp_step,
        "min_distance_to_current_wp": None if min_distance_to_current_wp == float('inf') else min_distance_to_current_wp,
    }


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="USV RL training pipeline")
    parser.add_argument("--phases", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--cycles", type=int, default=1,
                        help="Quantas vezes repetir o bloco completo de fases.")
    parser.add_argument("--phase-steps", type=int, default=4096)
    parser.add_argument("--episode-steps", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--hover-depth", type=float, default=5.0)
    parser.add_argument("--pool-depth", type=float, default=10.0)
    parser.add_argument("--pool-radius", type=float, default=30.0)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--enable-rayleigh", action="store_true",
                        help="Ativa perturbação ambiental não gaussiana (Rayleigh).")
    parser.add_argument("--rayleigh-sigma", type=float, default=0.03,
                        help="Escala da distribuição de Rayleigh para a maresia.")
    parser.add_argument("--env-disturbance-scale", type=float, default=1.0,
                        help="Intensidade global da perturbação ambiental Rayleigh.")
    parser.add_argument("--env-force-gain", type=float, default=1.0,
                        help="Ganho global aplicado às forças ambientais sobre o casco.")
    parser.add_argument("--env-turbulence-gain", type=float, default=1.0,
                        help="Ganho específico para componentes de turbulência/ondas.")
    parser.add_argument("--env-wave-freq", type=float, default=0.8,
                        help="Frequência dominante (Hz) usada no modelo simplificado de ondas.")
    parser.add_argument("--env-spectral", action="store_true",
                        help="Ativa modelo espectral de ondas (superposição harmônica).")
    parser.add_argument("--wave-harmonics", type=int, default=8,
                        help="Número de harmônicos no modelo espectral.")
    parser.add_argument("--wave-peak-freq", type=float, default=0.8,
                        help="Frequência de pico (Hz) do espectro de ondas.")
    parser.add_argument("--wave-amp-scale", type=float, default=0.02,
                        help="Escala de amplitude para os harmônicos (m/s).")
    parser.add_argument("--wave-hs", type=float, default=0.2,
                        help="Significant Wave Height (m) — calibra espectro JONSWAP.")
    parser.add_argument("--waypoint-threshold", type=float, default=0.5,
                        help="Raio (m) para considerar waypoint atingido.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output-dir", type=str, default="./training_runs")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    return PipelineConfig(
        phases=tuple(args.phases),
        cycles=max(1, int(args.cycles)),
        phase_steps=args.phase_steps,
        episode_steps=args.episode_steps,
        dt=args.dt,
        eval_steps=args.eval_steps,
        hover_depth=args.hover_depth,
        pool_depth=args.pool_depth,
        pool_radius=args.pool_radius,
        noise_scale=args.noise_scale,
        enable_rayleigh=args.enable_rayleigh,
        rayleigh_sigma=args.rayleigh_sigma,
        env_disturbance_scale=args.env_disturbance_scale,
        env_force_gain=args.env_force_gain,
        env_turbulence_gain=args.env_turbulence_gain,
        env_wave_freq=args.env_wave_freq,
        env_spectral_enabled=args.env_spectral,
        wave_num_harmonics=args.wave_harmonics,
        wave_peak_freq=args.wave_peak_freq,
        wave_amp_scale=args.wave_amp_scale,
        wave_hs=args.wave_hs,
        waypoint_threshold=args.waypoint_threshold,
        seed=args.seed,
        fresh=args.fresh,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
    ), args.no_eval


def main() -> int:
    config, no_eval = parse_args()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(config.checkpoint_dir)
    if config.fresh and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    geometry, physics, sensors, ekf, control, hrl, waypoints = build_stack(config)

    report = {
        "config": asdict(config),
        "phases": [],
    }

    for cycle_idx in range(config.cycles):
        print(f"\n=== Training cycle {cycle_idx + 1}/{config.cycles} ===")
        for phase in config.phases:
            if phase not in (1, 2, 3):
                raise ValueError(f"Unsupported phase {phase}. Use 1, 2, or 3.")
            summary = run_phase(config, hrl, physics, sensors, ekf, waypoints, phase)
            summary["cycle"] = cycle_idx + 1
            report["phases"].append(summary)
            print(
                f"Cycle {cycle_idx + 1} | Phase {phase} complete: "
                f"{summary['steps']} steps"
            )

    evaluation = None
    if not no_eval:
        evaluation = evaluate(hrl, physics, sensors, ekf, waypoints, config)
        report["evaluation"] = evaluation
        print(
            f"Evaluation: {evaluation['waypoints_reached']} waypoints reached, "
            f"mission_complete={evaluation['mission_complete']}"
        )

    report_path = output_dir / "rl_training_report.json"
    report_json = _to_jsonable(report)
    report_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    print(f"Report saved to {report_path}")
    print(f"Checkpoints saved to {checkpoint_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())