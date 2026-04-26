import argparse
import json
import os
import time
import io
import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine, VehicleState
from sensor_engine import Environment, ExtendedKalmanFilter, Obstacle, SensorEngine
from control_engine import ControlEngine
from mpc_controller import integrate_mpc
from rl_controller import integrate_rl
from mission_engine import COLLISION_THRESHOLD, DynamicObstacle, EpisodeTermination


DEFAULT_MAX_STEPS = 2000


@dataclass
class BenchmarkScenario:
    waypoints: List[List[float]]
    static_obstacles: List[Dict]
    dynamic_obstacles: List[Dict]
    dt: float = 0.01
    max_steps: int = DEFAULT_MAX_STEPS
    trials: int = 3
    pool_depth: float = 10.0
    pool_radius: float = 30.0
    noise_scale: float = 0.5
    rayleigh_enabled: bool = False
    rayleigh_sigma: float = 0.03
    env_disturbance_scale: float = 0.0
    seed: int = 42


@dataclass
class BenchmarkRunResult:
    controller: str
    trial: int
    termination: str
    success: bool
    collision: bool
    out_of_bounds: bool
    completion_rate: float
    waypoints_reached: int
    total_waypoints: int
    steps: int
    sim_time_s: float
    mean_tracking_error_m: float
    rms_tracking_error_m: float
    final_position_error_m: float
    min_clearance_m: float
    path_length_m: float
    mean_speed_mps: float
    mean_compute_ms: float
    energy_score: float
    score: float


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


class ControllerBenchmark:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir

    def run(
        self,
        scenario: BenchmarkScenario,
        controllers: Optional[List[str]] = None,
        progress_callback=None,
    ) -> Dict:
        controller_list = controllers or ["lqr", "mpc", "rl"]
        runs: List[BenchmarkRunResult] = []
        total_runs = len(controller_list) * scenario.trials
        run_index = 0

        for trial in range(scenario.trials):
            trial_seed = scenario.seed + trial
            for controller in controller_list:
                run_index += 1
                if progress_callback is not None:
                    progress_callback({
                        "stage": "running",
                        "controller": controller,
                        "trial": trial + 1,
                        "trials": scenario.trials,
                        "progress": run_index / total_runs,
                    })
                runs.append(self._run_once(controller, scenario, trial_seed, trial + 1))

        grouped: Dict[str, List[BenchmarkRunResult]] = {name: [] for name in controller_list}
        for run in runs:
            grouped.setdefault(run.controller, []).append(run)

        summary = {
            "scenario": asdict(scenario),
            "generated_at_epoch_s": time.time(),
            "runs": [asdict(run) for run in runs],
            "controllers": {},
        }

        for controller in controller_list:
            controller_runs = grouped.get(controller, [])
            if not controller_runs:
                continue

            summary["controllers"][controller] = {
                "success_rate": _mean([1.0 if r.success else 0.0 for r in controller_runs]),
                "collision_rate": _mean([1.0 if r.collision else 0.0 for r in controller_runs]),
                "out_of_bounds_rate": _mean([1.0 if r.out_of_bounds else 0.0 for r in controller_runs]),
                "mean_completion_rate": _mean([r.completion_rate for r in controller_runs]),
                "mean_time_s": _mean([r.sim_time_s for r in controller_runs]),
                "std_time_s": _std([r.sim_time_s for r in controller_runs]),
                "mean_tracking_error_m": _mean([r.mean_tracking_error_m for r in controller_runs]),
                "rms_tracking_error_m": _mean([r.rms_tracking_error_m for r in controller_runs]),
                "mean_final_error_m": _mean([r.final_position_error_m for r in controller_runs]),
                "mean_clearance_m": _mean([r.min_clearance_m for r in controller_runs]),
                "mean_path_length_m": _mean([r.path_length_m for r in controller_runs]),
                "mean_speed_mps": _mean([r.mean_speed_mps for r in controller_runs]),
                "mean_compute_ms": _mean([r.mean_compute_ms for r in controller_runs]),
                "mean_energy_score": _mean([r.energy_score for r in controller_runs]),
                "score": _mean([r.score for r in controller_runs]),
            }

        ranking = sorted(
            (
                {"controller": name, **metrics}
                for name, metrics in summary["controllers"].items()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        summary["ranking"] = ranking
        return summary

    def _run_once(
        self,
        controller_name: str,
        scenario: BenchmarkScenario,
        seed: int,
        trial_number: int,
    ) -> BenchmarkRunResult:
        rng = np.random.default_rng(seed)
        geo = GeometryEngine(L=0.8, D=0.1)
        physics = PhysicsEngine(geo, max_thruster_force=10.0)
        env = Environment(pool_depth=scenario.pool_depth, pool_radius=scenario.pool_radius)
        sensors = SensorEngine(
            env,
            noise_scale=scenario.noise_scale,
            rayleigh_sigma=scenario.rayleigh_sigma,
            enable_rayleigh=scenario.rayleigh_enabled,
            seed=seed,
        )
        sensors.set_environmental_disturbance(
            enabled=scenario.rayleigh_enabled,
            scale=scenario.env_disturbance_scale,
            rayleigh_sigma=scenario.rayleigh_sigma,
        )
        ekf = ExtendedKalmanFilter(physics)
        with self._quiet_stdout():
            control = ControlEngine(physics, hover_depth=scenario.pool_depth / 2.0)
            integrate_mpc(control, hover_depth=scenario.pool_depth / 2.0)
            hrl = integrate_rl(control, self.checkpoint_dir)

        static_obstacles = [
            Obstacle(position=np.array(obs["position"], dtype=float), radius=float(obs["radius"]))
            for obs in scenario.static_obstacles
        ]
        dynamic_obstacles = [
            DynamicObstacle(
                position=np.array(obs["position"], dtype=float),
                radius=float(obs["radius"]),
                velocity=np.array(obs.get("velocity", [0.0, 0.0, 0.0]), dtype=float),
                speed_max=float(obs.get("speed_max", 0.3)),
                bounds_min=np.array(obs.get("bounds_min", [-20, -20, 0.5]), dtype=float),
                bounds_max=np.array(obs.get("bounds_max", [20, 20, scenario.pool_depth - 0.5]), dtype=float),
            )
            for obs in scenario.dynamic_obstacles
        ]
        env.obstacles.extend(static_obstacles)
        env.obstacles.extend([obs.to_obstacle() for obs in dynamic_obstacles])

        waypoints = [np.array(wp, dtype=float) for wp in scenario.waypoints]
        control.set_waypoints([wp.copy() for wp in waypoints])
        hrl.set_waypoints([wp.copy() for wp in waypoints])
        if controller_name in ("lqr", "mpc"):
            with self._quiet_stdout():
                control.set_controller(controller_name)

        initial_state = VehicleState(z=scenario.pool_depth / 2.0)
        physics.reset(initial_state)
        ekf.reset(np.concatenate([initial_state.eta, initial_state.nu]))

        steps = 0
        error_samples: List[float] = []
        speed_samples: List[float] = []
        compute_samples_ms: List[float] = []
        min_clearance = float("inf")
        energy_score = 0.0
        path_length = 0.0
        prev_pos = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
        termination = EpisodeTermination.RUNNING

        while steps < scenario.max_steps and termination == EpisodeTermination.RUNNING:
            for dyn_obs in dynamic_obstacles:
                dyn_obs.step(scenario.dt, rng)

            env.obstacles = env.obstacles[:2 + len(static_obstacles)]
            env.obstacles.extend([obs.to_obstacle() for obs in dynamic_obstacles])

            bundle = sensors.read(physics.state, physics.time)

            ekf.predict(scenario.dt)
            ekf.update_imu(bundle.imu)
            ekf.update_barometer(bundle.barometer)
            ekf.update_sonar(bundle.sonar)
            est = ekf.state_estimate

            target = self._current_target(controller_name, control, hrl, waypoints)
            if target is not None:
                error_samples.append(float(np.linalg.norm(est.position - target)))

            t0 = time.perf_counter()
            if controller_name == "rl":
                with self._quiet_stdout():
                    cmd = hrl.compute(est, bundle.imu, bundle.sonar, scenario.dt, training=False)
            else:
                with self._quiet_stdout():
                    cmd = control.compute(est, physics.time)
            compute_samples_ms.append((time.perf_counter() - t0) * 1000.0)

            physics.step(
                thruster_power=cmd.thruster_power,
                thruster_theta=cmd.thruster_theta,
                thruster_phi=cmd.thruster_phi,
                ballast_cmd=cmd.ballast_cmd,
                thruster2_power=cmd.thruster2_power,
                thruster2_theta=cmd.thruster2_theta,
                thruster2_phi=cmd.thruster2_phi,
                dt=scenario.dt,
            )
            steps += 1

            pos = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
            path_length += float(np.linalg.norm(pos - prev_pos))
            prev_pos = pos
            speed_samples.append(float(np.linalg.norm(physics.state.nu[:3])))

            thruster2_power = cmd.thruster_power if cmd.thruster2_power is None else cmd.thruster2_power
            energy_score += (
                abs(cmd.thruster_power)
                + abs(thruster2_power)
                + 0.35 * abs(cmd.ballast_cmd)
            ) * scenario.dt

            valid_distances = [reading.distance for reading in bundle.sonar if reading.hit and reading.distance > 0]
            if valid_distances:
                min_clearance = min(min_clearance, float(min(valid_distances)))

            if self._has_collision(bundle.sonar):
                termination = EpisodeTermination.COLLISION
            elif self._out_of_bounds(pos, scenario):
                termination = EpisodeTermination.OUT_OF_BOUNDS
            elif self._mission_complete(controller_name, control, hrl):
                termination = EpisodeTermination.MISSION_COMPLETE

        if termination == EpisodeTermination.RUNNING:
            termination = EpisodeTermination.TIMEOUT

        final_target = waypoints[-1] if waypoints else np.zeros(3)
        final_error = float(np.linalg.norm(prev_pos - final_target))
        completion_rate, reached = self._completion(controller_name, control, hrl, len(waypoints))
        min_clearance_value = float(min_clearance if np.isfinite(min_clearance) else scenario.pool_radius)
        score = self._score_run(
            termination=termination,
            completion_rate=completion_rate,
            final_error=final_error,
            rms_error=float(np.sqrt(np.mean(np.square(error_samples)))) if error_samples else final_error,
            energy_score=energy_score,
            min_clearance=min_clearance_value,
            sim_time=steps * scenario.dt,
        )

        return BenchmarkRunResult(
            controller=controller_name,
            trial=trial_number,
            termination=termination.value,
            success=(termination == EpisodeTermination.MISSION_COMPLETE),
            collision=(termination == EpisodeTermination.COLLISION),
            out_of_bounds=(termination == EpisodeTermination.OUT_OF_BOUNDS),
            completion_rate=completion_rate,
            waypoints_reached=reached,
            total_waypoints=len(waypoints),
            steps=steps,
            sim_time_s=steps * scenario.dt,
            mean_tracking_error_m=_mean(error_samples) if error_samples else final_error,
            rms_tracking_error_m=float(np.sqrt(np.mean(np.square(error_samples)))) if error_samples else final_error,
            final_position_error_m=final_error,
            min_clearance_m=min_clearance_value,
            path_length_m=path_length,
            mean_speed_mps=_mean(speed_samples),
            mean_compute_ms=_mean(compute_samples_ms),
            energy_score=energy_score,
            score=score,
        )

    @staticmethod
    def _current_target(controller_name, control, hrl, waypoints):
        if controller_name == "rl":
            current = hrl.n3.current_waypoint
        else:
            current = control.current_waypoint
        if current is not None:
            return np.asarray(current, dtype=float)
        return np.asarray(waypoints[-1], dtype=float) if waypoints else None

    @staticmethod
    def _mission_complete(controller_name, control, hrl) -> bool:
        return bool(hrl.n3.mission_complete if controller_name == "rl" else control.mission_complete)

    @staticmethod
    def _completion(controller_name, control, hrl, total_waypoints: int):
        reached = hrl.n3.current_wp_idx if controller_name == "rl" else control.waypoint_index
        if total_waypoints <= 0:
            return 0.0, reached
        return float(reached / total_waypoints), int(reached)

    @staticmethod
    def _has_collision(sonar_readings) -> bool:
        return any(
            reading.hit and reading.distance > 0 and reading.distance < COLLISION_THRESHOLD
            for reading in sonar_readings
        )

    @staticmethod
    def _out_of_bounds(position: np.ndarray, scenario: BenchmarkScenario) -> bool:
        return (
            position[2] < 0.0
            or position[2] > scenario.pool_depth
            or float(np.linalg.norm(position[:2])) > scenario.pool_radius
        )

    @staticmethod
    def _score_run(
        termination: EpisodeTermination,
        completion_rate: float,
        final_error: float,
        rms_error: float,
        energy_score: float,
        min_clearance: float,
        sim_time: float,
    ) -> float:
        score = 100.0 * completion_rate
        if termination == EpisodeTermination.MISSION_COMPLETE:
            score += 40.0
        if termination == EpisodeTermination.COLLISION:
            score -= 60.0
        if termination == EpisodeTermination.OUT_OF_BOUNDS:
            score -= 40.0
        score -= 6.0 * final_error
        score -= 4.0 * rms_error
        score -= 0.8 * energy_score
        score -= 0.15 * sim_time
        score += min(10.0, max(0.0, min_clearance - COLLISION_THRESHOLD) * 2.0)
        return float(score)

    @staticmethod
    def _quiet_stdout():
        return contextlib.redirect_stdout(io.StringIO())


def _parse_args():
    parser = argparse.ArgumentParser(description="Run USV controller benchmarks")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--output-dir", type=str, default="./training_runs")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--pool-depth", type=float, default=10.0)
    parser.add_argument("--pool-radius", type=float, default=30.0)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--enable-rayleigh", action="store_true")
    parser.add_argument("--rayleigh-sigma", type=float, default=0.03)
    parser.add_argument("--env-disturbance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controllers", nargs="*", default=["lqr", "mpc", "rl"])
    parser.add_argument(
        "--waypoint",
        action="append",
        nargs=3,
        metavar=("X", "Y", "Z"),
        type=float,
        help="Waypoint in NED coordinates. Can be provided multiple times.",
    )
    parser.add_argument("--waypoints-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default="benchmark_report.json")
    return parser.parse_args()


def _load_waypoints(args) -> List[List[float]]:
    if args.waypoints_file:
        path = Path(args.waypoints_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("waypoints-file must contain a JSON list")
        return [[float(coord) for coord in waypoint] for waypoint in data]
    if args.waypoint:
        return [[float(x), float(y), float(z)] for x, y, z in args.waypoint]
    return [[5.0, 0.0, 5.0]]


def main() -> int:
    args = _parse_args()
    scenario = BenchmarkScenario(
        waypoints=_load_waypoints(args),
        static_obstacles=[],
        dynamic_obstacles=[],
        dt=float(args.dt),
        max_steps=int(args.max_steps),
        trials=int(args.trials),
        pool_depth=float(args.pool_depth),
        pool_radius=float(args.pool_radius),
        noise_scale=float(args.noise_scale),
        rayleigh_enabled=bool(args.enable_rayleigh),
        rayleigh_sigma=float(args.rayleigh_sigma),
        env_disturbance_scale=float(args.env_disturbance_scale),
        seed=int(args.seed),
    )

    benchmark = ControllerBenchmark(args.checkpoint_dir)
    result = benchmark.run(scenario, controllers=list(args.controllers))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_file
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Benchmark saved to {output_path}")
    print(json.dumps(result.get("ranking", []), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
