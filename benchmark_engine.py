import argparse
import json
import os
import time
import io
import contextlib
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine, VehicleState
from sensor_engine import EKFState, Environment, ExtendedKalmanFilter, Obstacle, SensorEngine
from control_engine import ControlEngine
import control_engine as control_module
from mpc_controller import integrate_mpc
from rl_controller import integrate_rl
from mission_engine import COLLISION_THRESHOLD, DynamicObstacle, EpisodeTermination
from replay_exporter import ReplayExporter
from vehicle_profiles import VehicleProfile, load_vehicle_profile


DEFAULT_MAX_STEPS = 200000
DEFAULT_BENCHMARK_MODE = "mission"
BENCHMARK_MODES = ("mission", "stability", "docking")
DOCKING_MAX_FINAL_SPEED_MPS = 0.15
DOCKING_SETTLE_TIME_S = 0.1


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
    pool_shape: str = "circular"
    pool_length: Optional[float] = None
    pool_width: Optional[float] = None
    noise_scale: float = 0.5
    rayleigh_enabled: bool = True
    rayleigh_sigma: float = 0.03
    env_disturbance_scale: float = 0.5
    env_spectral_enabled: bool = False
    wave_hs: float = 0.0
    seed: int = 42
    benchmark_mode: str = DEFAULT_BENCHMARK_MODE
    hold_position: Optional[List[float]] = None
    position_tolerance_m: float = 0.20
    attitude_tolerance_deg: float = 12.0
    obstacle_collision_threshold_m: float = COLLISION_THRESHOLD
    boundary_collision_threshold_m: float = COLLISION_THRESHOLD
    use_truth_position_for_guidance: bool = False
    vehicle_profile_csv: Optional[str] = None
    surface_depth: Optional[float] = None
    planar_dof: bool = False
    lqr_tuning: Optional[Dict[str, float]] = None
    lqr_guidance_tuning: Optional[Dict[str, float]] = None
    mpc_tuning: Optional[Dict[str, float]] = None


@dataclass
class BenchmarkRunResult:
    benchmark_mode: str
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
    mean_attitude_error_deg: float
    rms_attitude_error_deg: float
    final_attitude_error_deg: float
    min_clearance_m: float
    path_length_m: float
    mean_speed_mps: float
    mean_compute_ms: float
    energy_score: float
    score: float
    replay_run_id: Optional[str] = None
    replay_frames_path: Optional[str] = None
    replay_meta_path: Optional[str] = None


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _wrap_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class ControllerBenchmark:
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        replay_dir: str = "./training_runs/replays",
        enable_replay_export: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.enable_replay_export = bool(enable_replay_export)
        self.replay_exporter = ReplayExporter(replay_dir) if self.enable_replay_export else None

    def run(
        self,
        scenario: BenchmarkScenario,
        controllers: Optional[List[str]] = None,
        progress_callback=None,
    ) -> Dict:
        scenario = self._normalize_scenario(scenario)
        controller_list = controllers or ["lqr", "mpc", "rl"]
        runs: List[BenchmarkRunResult] = []
        replay_runs: List[Dict] = []
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
                run_result, replay_manifest = self._run_once(controller, scenario, trial_seed, trial + 1)
                runs.append(run_result)
                if replay_manifest is not None:
                    replay_runs.append(replay_manifest)

        grouped: Dict[str, List[BenchmarkRunResult]] = {name: [] for name in controller_list}
        for run in runs:
            grouped.setdefault(run.controller, []).append(run)

        summary = {
            "scenario": asdict(scenario),
            "benchmark_mode": scenario.benchmark_mode,
            "generated_at_epoch_s": time.time(),
            "runs": [asdict(run) for run in runs],
            "controllers": {},
            "replays": replay_runs,
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
                "mean_attitude_error_deg": _mean([r.mean_attitude_error_deg for r in controller_runs]),
                "rms_attitude_error_deg": _mean([r.rms_attitude_error_deg for r in controller_runs]),
                "mean_final_attitude_error_deg": _mean([r.final_attitude_error_deg for r in controller_runs]),
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

    @staticmethod
    def _normalize_scenario(scenario: BenchmarkScenario) -> BenchmarkScenario:
        mode = str(getattr(scenario, "benchmark_mode", DEFAULT_BENCHMARK_MODE) or DEFAULT_BENCHMARK_MODE).lower()
        if mode not in BENCHMARK_MODES:
            raise ValueError(f"Unsupported benchmark_mode '{mode}'. Expected one of {BENCHMARK_MODES}.")

        shape = str(getattr(scenario, "pool_shape", "circular") or "circular").lower()
        if shape not in ("circular", "rectangle"):
            raise ValueError("Unsupported pool_shape. Expected 'circular' or 'rectangle'.")
        scenario.pool_shape = shape
        if scenario.pool_shape == "rectangle":
            if scenario.pool_length is None:
                scenario.pool_length = float(2.0 * scenario.pool_radius)
            if scenario.pool_width is None:
                scenario.pool_width = float(2.0 * scenario.pool_radius)
            scenario.pool_length = float(scenario.pool_length)
            scenario.pool_width = float(scenario.pool_width)
            if scenario.pool_length <= 0.0 or scenario.pool_width <= 0.0:
                raise ValueError("pool_length and pool_width must be > 0 for rectangle pool_shape.")

        scenario.benchmark_mode = mode
        if scenario.hold_position is None and mode == "docking":
            scenario.hold_position = [1.0, 0.0, scenario.pool_depth / 2.0]
        elif scenario.hold_position is None:
            target_depth = scenario.surface_depth if scenario.surface_depth is not None else scenario.pool_depth / 2.0
            scenario.hold_position = [0.0, 0.0, float(target_depth)]
        else:
            scenario.hold_position = [float(v) for v in scenario.hold_position]
        scenario.position_tolerance_m = float(scenario.position_tolerance_m)
        scenario.attitude_tolerance_deg = float(scenario.attitude_tolerance_deg)
        scenario.obstacle_collision_threshold_m = float(scenario.obstacle_collision_threshold_m)
        scenario.boundary_collision_threshold_m = float(scenario.boundary_collision_threshold_m)
        scenario.env_spectral_enabled = bool(scenario.env_spectral_enabled)
        scenario.wave_hs = float(max(0.0, scenario.wave_hs))
        if scenario.surface_depth is not None:
            scenario.surface_depth = float(scenario.surface_depth)
        scenario.planar_dof = bool(scenario.planar_dof)
        if scenario.lqr_tuning is not None:
            scenario.lqr_tuning = {str(k): float(v) for k, v in dict(scenario.lqr_tuning).items()}
        if scenario.lqr_guidance_tuning is not None:
            scenario.lqr_guidance_tuning = {str(k): float(v) for k, v in dict(scenario.lqr_guidance_tuning).items()}
        if scenario.mpc_tuning is not None:
            scenario.mpc_tuning = {str(k): float(v) for k, v in dict(scenario.mpc_tuning).items()}
        return scenario

    def _run_once(
        self,
        controller_name: str,
        scenario: BenchmarkScenario,
        seed: int,
        trial_number: int,
    ) -> tuple[BenchmarkRunResult, Optional[Dict]]:
        rng = np.random.default_rng(seed)
        profile = None
        if scenario.vehicle_profile_csv:
            profile = load_vehicle_profile(scenario.vehicle_profile_csv)
            geo = GeometryEngine(L=profile.length_m, D=profile.beam_m)
            physics = PhysicsEngine(
                geo,
                max_thruster_force=10.0,
                rigid_body_mass=profile.mass_kg,
                rigid_body_inertia=profile.inertia_kgm2,
                thruster_port_position=profile.thruster_port_position_m,
                thruster_starboard_position=profile.thruster_starboard_position_m,
                planar_dof=scenario.planar_dof,
            )
        else:
            geo = GeometryEngine(L=0.8, D=0.1)
            physics = PhysicsEngine(geo, max_thruster_force=10.0, planar_dof=scenario.planar_dof)
        env = Environment(pool_depth=scenario.pool_depth, pool_radius=scenario.pool_radius)
        sensors = SensorEngine(
            env,
            noise_scale=scenario.noise_scale,
            rayleigh_sigma=scenario.rayleigh_sigma,
            enable_rayleigh=scenario.rayleigh_enabled,
            seed=seed,
            wave_hs=scenario.wave_hs,
        )
        sensors.set_environmental_disturbance(
            enabled=scenario.rayleigh_enabled,
            scale=scenario.env_disturbance_scale,
            rayleigh_sigma=scenario.rayleigh_sigma,
            spectral=scenario.env_spectral_enabled,
            wave_hs=scenario.wave_hs,
        )
        ekf = ExtendedKalmanFilter(physics, pool_radius=scenario.pool_radius, pool_depth=scenario.pool_depth)

        # Apply optional LQR guidance tuning before ControlEngine instantiates LQR.
        try:
            guidance_tuning = {}
            module_guidance_tuning = getattr(control_module, 'LQR_GUIDANCE_TUNING', {})
            if module_guidance_tuning:
                guidance_tuning.update(dict(module_guidance_tuning))
            if scenario.lqr_guidance_tuning:
                guidance_tuning.update(dict(scenario.lqr_guidance_tuning))
            control_module.LQR_GUIDANCE_TUNING = guidance_tuning
        except Exception:
            pass

        with self._quiet_stdout():
            control = ControlEngine(physics, hover_depth=scenario.pool_depth / 2.0)
            integrate_mpc(control, hover_depth=scenario.pool_depth / 2.0)
            hrl = integrate_rl(control, self.checkpoint_dir)
        control.set_pool_bounds(
            pool_shape=scenario.pool_shape,
            pool_radius=scenario.pool_radius,
            pool_length=scenario.pool_length,
            pool_width=scenario.pool_width,
        )

        # Apply optional global/scenario MPC tuning (set mpc_controller.MPC_TUNING or scenario.mpc_tuning).
        try:
            mpc_tuning = {}
            module_mpc_tuning = getattr(__import__('mpc_controller'), 'MPC_TUNING', {})
            if module_mpc_tuning:
                mpc_tuning.update(dict(module_mpc_tuning))
            if scenario.mpc_tuning:
                mpc_tuning.update(dict(scenario.mpc_tuning))
            if mpc_tuning and getattr(control, '_mpc', None) is not None:
                control._mpc.tune_max_cruise_mult = float(mpc_tuning.get('max_cruise_mult', getattr(control._mpc, 'tune_max_cruise_mult', 1.0)))
                control._mpc.tune_desired_surge_mult = float(mpc_tuning.get('desired_surge_mult', getattr(control._mpc, 'tune_desired_surge_mult', 1.0)))
                control._mpc.tune_base_power_gain_mult = float(mpc_tuning.get('base_power_gain_mult', getattr(control._mpc, 'tune_base_power_gain_mult', 1.0)))
                control._mpc.tune_lateral_yaw_mult = float(mpc_tuning.get('lateral_yaw_mult', getattr(control._mpc, 'tune_lateral_yaw_mult', 1.0)))
                control._mpc.tune_yaw_error_mult = float(mpc_tuning.get('yaw_error_mult', getattr(control._mpc, 'tune_yaw_error_mult', 1.0)))
                control._mpc.tune_yaw_damp_mult = float(mpc_tuning.get('yaw_damp_mult', getattr(control._mpc, 'tune_yaw_damp_mult', 1.0)))
                control._mpc.tune_lateral_speed_penalty_mult = float(mpc_tuning.get('lateral_speed_penalty_mult', getattr(control._mpc, 'tune_lateral_speed_penalty_mult', 1.0)))
                control._mpc.tune_reverse_penalty = float(mpc_tuning.get('reverse_penalty', getattr(control._mpc, 'tune_reverse_penalty', 0.55)))
                control._mpc.tune_terminal_pull_mult = float(mpc_tuning.get('terminal_pull_mult', getattr(control._mpc, 'tune_terminal_pull_mult', 1.0)))
                control._mpc.tune_boundary_margin_m = float(mpc_tuning.get('boundary_margin_m', getattr(control._mpc, 'tune_boundary_margin_m', 0.25)))
        except Exception:
            pass

        # Apply optional global/scenario LQR tuning (set benchmark_engine.GLOBAL_TUNING or scenario.lqr_tuning).
        try:
            tuning = globals().get('GLOBAL_TUNING', {})
            if scenario.lqr_tuning:
                tuning = {**dict(tuning), **dict(scenario.lqr_tuning)}
            if tuning:
                wt = control._lqr.weights
                # apply simple multiplier keys if present
                if 'lqr_q_z_mult' in tuning:
                    wt.q_z *= float(tuning['lqr_q_z_mult'])
                if 'lqr_q_att_mult' in tuning:
                    wt.q_phi *= float(tuning['lqr_q_att_mult'])
                    wt.q_tht *= float(tuning['lqr_q_att_mult'])
                if 'lqr_q_vel_mult' in tuning:
                    for k in ('q_u','q_v','q_w','q_p','q_q','q_r'):
                        setattr(wt, k, float(getattr(wt, k)) * float(tuning['lqr_q_vel_mult']))
                if 'lqr_r_power_mult' in tuning:
                    wt.r_thrust_power *= float(tuning['lqr_r_power_mult'])
                # recompute LQR gains with adjusted weights
                try:
                    control._lqr.K = control._lqr._solve_riccati()
                except Exception:
                    pass
        except Exception:
            pass

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
        hold_position = np.array(scenario.hold_position or [0.0, 0.0, scenario.pool_depth / 2.0], dtype=float)
        start_depth = float(scenario.surface_depth if scenario.surface_depth is not None else scenario.pool_depth / 2.0)
        if scenario.benchmark_mode in ("stability", "docking"):
            control.set_reference(hold_position)
            hrl.set_waypoints([hold_position.copy()], waypoint_threshold=scenario.position_tolerance_m)
        else:
            control.set_waypoints([wp.copy() for wp in waypoints], waypoint_threshold=scenario.position_tolerance_m)
            hrl.set_waypoints([wp.copy() for wp in waypoints], waypoint_threshold=scenario.position_tolerance_m)
        if controller_name in ("lqr", "mpc"):
            with self._quiet_stdout():
                control.set_controller(controller_name)

        initial_state = VehicleState(z=start_depth)
        physics.reset(initial_state)
        ekf.reset(np.concatenate([initial_state.eta, initial_state.nu]))

        replay_writer = None
        if self.replay_exporter is not None:
            replay_writer = self.replay_exporter.start_run(
                benchmark_mode=scenario.benchmark_mode,
                controller=controller_name,
                trial=trial_number,
                seed=seed,
                scenario=asdict(scenario),
            )

        steps = 0
        error_samples: List[float] = []
        attitude_error_samples_deg: List[float] = []
        speed_samples: List[float] = []
        compute_samples_ms: List[float] = []
        min_clearance = float("inf")
        energy_score = 0.0
        path_length = 0.0
        stable_steps = 0
        prev_pos = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
        termination = EpisodeTermination.RUNNING
        docking_hold_steps = 0
        docking_hold_required = max(1, int(np.ceil(DOCKING_SETTLE_TIME_S / max(scenario.dt, 1e-6))))
        docking_initial_error = float(np.linalg.norm(prev_pos - hold_position))

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
            ekf.update_sonar_position(bundle.sonar)  # NEW: Constrain X/Y position using sonar
            # determine current target early so vision can observe it
            target = self._current_target(controller_name, control, hrl, waypoints, scenario)
            # Vision-style relative waypoint detection and EKF update
            if target is not None:
                vision_meas = sensors.detect_waypoint(physics.state, target, physics.time)
                ekf.update_vision(vision_meas, target)
            est = ekf.state_estimate
            est_for_control = est
            if scenario.use_truth_position_for_guidance:
                eta = est.eta.copy()
                eta[:3] = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
                est_for_control = EKFState(
                    eta=eta,
                    nu=est.nu.copy(),
                    P=est.P.copy(),
                    timestamp=est.timestamp,
                )

            desired_yaw = self._desired_yaw(scenario.benchmark_mode, target, prev_pos)
            if target is not None:
                error_samples.append(float(np.linalg.norm(prev_pos - target)))
            attitude_error_samples_deg.append(
                self._attitude_error_deg(
                    roll=physics.state.phi,
                    pitch=physics.state.tht,
                    yaw=physics.state.psi,
                    desired_yaw=desired_yaw,
                )
            )

            t0 = time.perf_counter()
            if controller_name == "rl":
                with self._quiet_stdout():
                    cmd = hrl.compute(
                        est_for_control,
                        bundle.imu,
                        bundle.sonar,
                        scenario.dt,
                        training=False,
                        navigation_position=np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float),
                    )
            else:
                with self._quiet_stdout():
                    cmd = control.compute(est_for_control, physics.time)
            compute_samples_ms.append((time.perf_counter() - t0) * 1000.0)

            env_cur, env_turb = sensors.get_environmental_state()
            env_harm = sensors.get_environmental_harmonics()
            physics.step(
                thruster_power=cmd.thruster_power,
                thruster_theta=cmd.thruster_theta,
                thruster_phi=cmd.thruster_phi,
                ballast_cmd=cmd.ballast_cmd,
                thruster2_power=cmd.thruster2_power,
                thruster2_theta=cmd.thruster2_theta,
                thruster2_phi=cmd.thruster2_phi,
                dt=scenario.dt,
                env_current_world=env_cur,
                env_turbulence=env_turb,
                env_harmonics=env_harm,
            )
            steps += 1

            pos = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
            path_length += float(np.linalg.norm(pos - prev_pos))
            prev_pos = pos
            speed_samples.append(float(np.linalg.norm(physics.state.nu[:3])))
            target_after_step = self._current_target(controller_name, control, hrl, waypoints, scenario)
            desired_yaw_after_step = self._desired_yaw(scenario.benchmark_mode, target_after_step, pos)
            if target_after_step is not None:
                error_samples[-1] = float(np.linalg.norm(pos - target_after_step))
            attitude_error_samples_deg[-1] = self._attitude_error_deg(
                roll=physics.state.phi,
                pitch=physics.state.tht,
                yaw=physics.state.psi,
                desired_yaw=desired_yaw_after_step,
            )

            thruster2_power = cmd.thruster_power if cmd.thruster2_power is None else cmd.thruster2_power
            energy_score += (
                abs(cmd.thruster_power)
                + abs(thruster2_power)
                + 0.35 * abs(cmd.ballast_cmd)
            ) * scenario.dt

            boundary_clearance, obstacle_clearance, clearance_now = self._clearances(pos, scenario, static_obstacles, dynamic_obstacles)
            min_clearance = min(min_clearance, clearance_now)

            if scenario.benchmark_mode == "stability":
                if self._stability_ready(
                    position_error_m=float(np.linalg.norm(pos - hold_position)),
                    attitude_error_deg=attitude_error_samples_deg[-1],
                    scenario=scenario,
                ):
                    stable_steps += 1

            if (
                boundary_clearance < scenario.boundary_collision_threshold_m
                or obstacle_clearance < scenario.obstacle_collision_threshold_m
            ):
                termination = EpisodeTermination.COLLISION
            elif self._out_of_bounds(pos, scenario):
                termination = EpisodeTermination.OUT_OF_BOUNDS
            elif scenario.benchmark_mode == "mission" and self._mission_complete(controller_name, control, hrl):
                termination = EpisodeTermination.MISSION_COMPLETE
            elif scenario.benchmark_mode == "docking":
                attitude_now = attitude_error_samples_deg[-1] if attitude_error_samples_deg else 0.0
                speed_now = speed_samples[-1] if speed_samples else 0.0
                dock_error_now = float(np.linalg.norm(pos - hold_position))
                if self._docking_ready(
                    position_error_m=dock_error_now,
                    attitude_error_deg=attitude_now,
                    speed_mps=speed_now,
                    scenario=scenario,
                ):
                    docking_hold_steps += 1
                else:
                    docking_hold_steps = 0

                if docking_hold_steps >= docking_hold_required:
                    termination = EpisodeTermination.MISSION_COMPLETE

            if replay_writer is not None:
                physics_data = physics.to_dict()
                tracking_error = float(error_samples[-1]) if error_samples else float("nan")
                attitude_error_deg = float(attitude_error_samples_deg[-1]) if attitude_error_samples_deg else 0.0
                replay_writer.write_frame({
                    "step": steps,
                    "time": float(physics.time),
                    "benchmark_mode": scenario.benchmark_mode,
                    "controller": controller_name,
                    "trial": trial_number,
                    "state_true": {
                        "position": [float(physics.state.x), float(physics.state.y), float(physics.state.z)],
                        "quaternion": [
                            float(physics.state.qw),
                            float(physics.state.qx),
                            float(physics.state.qy),
                            float(physics.state.qz),
                        ],
                        "euler": [float(physics.state.phi), float(physics.state.tht), float(physics.state.psi)],
                        "velocity_linear": [float(physics.state.u), float(physics.state.v), float(physics.state.w)],
                        "velocity_angular": [float(physics.state.p), float(physics.state.q), float(physics.state.r)],
                    },
                    "ekf_estimate": {
                        "eta": est.eta,
                        "nu": est.nu,
                        "position": est.position,
                        "orientation": est.orientation,
                        "velocity_linear": est.velocity_linear,
                        "velocity_angular": est.velocity_angular,
                    },
                    "command": cmd.to_dict(),
                    "sensors": bundle.to_dict(),
                    "environment": {
                        "pool_depth": float(scenario.pool_depth),
                        "pool_radius": float(scenario.pool_radius),
                        "pool_shape": str(scenario.pool_shape),
                        "pool_length": (None if scenario.pool_length is None else float(scenario.pool_length)),
                        "pool_width": (None if scenario.pool_width is None else float(scenario.pool_width)),
                        "rayleigh_enabled": bool(scenario.rayleigh_enabled),
                        "rayleigh_sigma": float(scenario.rayleigh_sigma),
                        "env_disturbance_scale": float(scenario.env_disturbance_scale),
                        "dynamic_obstacles": [
                            {
                                "position": obs.position,
                                "radius": float(obs.radius),
                                "velocity": obs.velocity,
                                "speed_max": float(obs.speed_max),
                            }
                            for obs in dynamic_obstacles
                        ],
                        "static_obstacles": [
                            {
                                "position": obs.position,
                                "radius": float(obs.radius),
                            }
                            for obs in static_obstacles
                        ],
                    },
                    "vectors": {
                        "velocity_body": physics.state.nu,
                        "thrust_total": physics_data["thruster"]["force_vector"],
                        "thrust_port": physics_data["thruster_pair"]["port"]["force_vector"],
                        "thrust_starboard": physics_data["thruster_pair"]["starboard"]["force_vector"],
                    },
                    "metrics": {
                        "tracking_error_m": tracking_error,
                        "attitude_error_deg": attitude_error_deg,
                        "min_clearance_m": float(min_clearance if np.isfinite(min_clearance) else scenario.pool_radius),
                        "energy_score": float(energy_score),
                        "path_length_m": float(path_length),
                        "mean_speed_mps": _mean(speed_samples),
                        "termination": termination.value,
                    },
                })

        rms_tracking_error = float(np.sqrt(np.mean(np.square(error_samples)))) if error_samples else 0.0
        mean_attitude_error_deg = _mean(attitude_error_samples_deg)
        rms_attitude_error_deg = float(np.sqrt(np.mean(np.square(attitude_error_samples_deg)))) if attitude_error_samples_deg else 0.0
        final_attitude_error_deg = attitude_error_samples_deg[-1] if attitude_error_samples_deg else 0.0
        final_target = hold_position if scenario.benchmark_mode in ("stability", "docking") else (waypoints[-1] if waypoints else hold_position)
        final_error = float(np.linalg.norm(prev_pos - final_target))
        if termination == EpisodeTermination.RUNNING:
            termination = EpisodeTermination.TIMEOUT

        final_speed = speed_samples[-1] if speed_samples else 0.0

        if scenario.benchmark_mode == "stability":
            completion_rate = float(stable_steps / max(1, steps))
            reached = 0
            success = (
                termination == EpisodeTermination.TIMEOUT
                and final_error <= scenario.position_tolerance_m
                and final_attitude_error_deg <= scenario.attitude_tolerance_deg
            )
            termination_label = "stability_complete" if success else termination.value
        elif scenario.benchmark_mode == "docking":
            success = termination == EpisodeTermination.MISSION_COMPLETE
            reached = 1 if success else 0
            if success:
                completion_rate = 1.0
            else:
                completion_rate = max(0.0, 1.0 - final_error / max(docking_initial_error, 1e-6))
            termination_label = "docking_complete" if success else termination.value
        else:
            completion_rate, reached = self._completion(controller_name, control, hrl, len(waypoints))
            success = termination == EpisodeTermination.MISSION_COMPLETE
            termination_label = termination.value

        min_clearance_value = float(min_clearance if np.isfinite(min_clearance) else scenario.pool_radius)
        score = self._score_run(
            benchmark_mode=scenario.benchmark_mode,
            termination=termination,
            success=success,
            completion_rate=completion_rate,
            final_error=final_error,
            rms_error=rms_tracking_error if error_samples else final_error,
            mean_attitude_error_deg=mean_attitude_error_deg,
            final_attitude_error_deg=final_attitude_error_deg,
            energy_score=energy_score,
            min_clearance=min_clearance_value,
            sim_time=steps * scenario.dt,
        )

        replay_manifest = None
        if replay_writer is not None:
            replay_manifest = replay_writer.close(
                summary={
                    "benchmark_mode": scenario.benchmark_mode,
                    "controller": controller_name,
                    "trial": trial_number,
                    "termination": termination_label,
                    "success": success,
                    "steps": steps,
                    "sim_time_s": steps * scenario.dt,
                    "score": score,
                }
            )

        run_result = BenchmarkRunResult(
            benchmark_mode=scenario.benchmark_mode,
            controller=controller_name,
            trial=trial_number,
            termination=termination_label,
            success=success,
            collision=(termination == EpisodeTermination.COLLISION),
            out_of_bounds=(termination == EpisodeTermination.OUT_OF_BOUNDS),
            completion_rate=completion_rate,
            waypoints_reached=reached,
            total_waypoints=(0 if scenario.benchmark_mode == "stability" else (1 if scenario.benchmark_mode == "docking" else len(waypoints))),
            steps=steps,
            sim_time_s=steps * scenario.dt,
            mean_tracking_error_m=_mean(error_samples) if error_samples else final_error,
            rms_tracking_error_m=rms_tracking_error if error_samples else final_error,
            final_position_error_m=final_error,
            mean_attitude_error_deg=mean_attitude_error_deg,
            rms_attitude_error_deg=rms_attitude_error_deg,
            final_attitude_error_deg=final_attitude_error_deg,
            min_clearance_m=min_clearance_value,
            path_length_m=path_length,
            mean_speed_mps=_mean(speed_samples),
            mean_compute_ms=_mean(compute_samples_ms),
            energy_score=energy_score,
            score=score,
            replay_run_id=(replay_manifest.get("run_id") if replay_manifest else None),
            replay_frames_path=(replay_manifest.get("frames_path") if replay_manifest else None),
            replay_meta_path=(replay_manifest.get("meta_path") if replay_manifest else None),
        )
        return run_result, replay_manifest

    @staticmethod
    def _current_target(controller_name, control, hrl, waypoints, scenario: BenchmarkScenario):
        if scenario.benchmark_mode in ("stability", "docking"):
            return np.asarray(scenario.hold_position, dtype=float)
        if controller_name == "rl":
            current = hrl.n3.current_waypoint
        else:
            current = control.current_waypoint
        if current is not None:
            return np.asarray(current, dtype=float)
        return np.asarray(waypoints[-1], dtype=float) if waypoints else None

    @staticmethod
    def _attitude_error_deg(
        *,
        roll: float,
        pitch: float,
        yaw: float,
        desired_yaw: Optional[float],
    ) -> float:
        components = [roll, pitch]
        if desired_yaw is not None:
            components.append(_wrap_angle_rad(yaw - desired_yaw))
        err = np.asarray(components, dtype=float)
        return float(np.degrees(np.linalg.norm(err)))

    @staticmethod
    def _desired_yaw(
        benchmark_mode: str,
        target: Optional[np.ndarray],
        position: np.ndarray,
    ) -> Optional[float]:
        if benchmark_mode in ("stability", "docking"):
            return None
        if target is None:
            return 0.0
        delta = np.asarray(target, dtype=float)[:2] - np.asarray(position, dtype=float)[:2]
        if np.linalg.norm(delta) <= 1e-9:
            return 0.0
        return float(np.arctan2(delta[1], delta[0]))

    @staticmethod
    def _mission_complete(controller_name, control, hrl) -> bool:
        return bool(hrl.n3.mission_complete if controller_name == "rl" else control.mission_complete)

    @staticmethod
    def _docking_ready(
        position_error_m: float,
        attitude_error_deg: float,
        speed_mps: float,
        scenario: BenchmarkScenario,
    ) -> bool:
        return (
            position_error_m <= scenario.position_tolerance_m
            and attitude_error_deg <= scenario.attitude_tolerance_deg
            and speed_mps <= DOCKING_MAX_FINAL_SPEED_MPS
        )

    @staticmethod
    def _stability_ready(
        position_error_m: float,
        attitude_error_deg: float,
        scenario: BenchmarkScenario,
    ) -> bool:
        return (
            position_error_m <= scenario.position_tolerance_m
            and attitude_error_deg <= scenario.attitude_tolerance_deg
        )

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
    def _clearances(
        position: np.ndarray,
        scenario: BenchmarkScenario,
        static_obstacles: List[Obstacle],
        dynamic_obstacles: List[DynamicObstacle],
    ) -> tuple[float, float, float]:
        pos = np.asarray(position, dtype=float)
        boundary_clearances = [ControllerBenchmark._horizontal_clearance(pos, scenario)]
        if not scenario.planar_dof:
            boundary_clearances.extend([
                pos[2],
                scenario.pool_depth - pos[2],
            ])

        obstacle_clearances = []
        for obs in static_obstacles:
            obstacle_clearances.append(float(np.linalg.norm(pos - obs.position) - obs.radius))
        for obs in dynamic_obstacles:
            obstacle_clearances.append(float(np.linalg.norm(pos - obs.position) - obs.radius))

        min_boundary = float(min(boundary_clearances)) if boundary_clearances else float("inf")
        min_obstacle = float(min(obstacle_clearances)) if obstacle_clearances else float("inf")
        min_total = float(min(min_boundary, min_obstacle))
        return min_boundary, min_obstacle, min_total

    @staticmethod
    def _out_of_bounds(position: np.ndarray, scenario: BenchmarkScenario) -> bool:
        horizontal_clearance = ControllerBenchmark._horizontal_clearance(np.asarray(position, dtype=float), scenario)
        if scenario.planar_dof:
            return horizontal_clearance < 0.0
        return (
            position[2] < 0.0
            or position[2] > scenario.pool_depth
            or horizontal_clearance < 0.0
        )

    @staticmethod
    def _horizontal_clearance(position: np.ndarray, scenario: BenchmarkScenario) -> float:
        if scenario.pool_shape == "rectangle":
            half_length = 0.5 * float(scenario.pool_length)
            half_width = 0.5 * float(scenario.pool_width)
            clearance_x = half_length - abs(float(position[0]))
            clearance_y = half_width - abs(float(position[1]))
            return float(min(clearance_x, clearance_y))
        return float(scenario.pool_radius - float(np.linalg.norm(position[:2])))

    @staticmethod
    def _score_run(
        benchmark_mode: str,
        termination: EpisodeTermination,
        success: bool,
        completion_rate: float,
        final_error: float,
        rms_error: float,
        mean_attitude_error_deg: float,
        final_attitude_error_deg: float,
        energy_score: float,
        min_clearance: float,
        sim_time: float,
    ) -> float:
        if benchmark_mode in ("stability", "docking"):
            score = 100.0
            if success:
                score += 20.0
            if termination == EpisodeTermination.COLLISION:
                score -= 120.0
            if termination == EpisodeTermination.OUT_OF_BOUNDS:
                score -= 80.0
            score -= 10.0 * final_error
            score -= 6.0 * rms_error
            score -= 0.7 * mean_attitude_error_deg
            score -= 0.5 * final_attitude_error_deg
            score -= 0.8 * energy_score
            score += max(0.0, min(10.0, sim_time * 0.1))
            return float(score)

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
    parser.add_argument("--scenario-file", type=str, default=None)
    parser.add_argument("--benchmark-mode", type=str, default=DEFAULT_BENCHMARK_MODE, choices=list(BENCHMARK_MODES))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--pool-depth", type=float, default=10.0)
    parser.add_argument("--pool-radius", type=float, default=30.0)
    parser.add_argument("--pool-shape", type=str, default="circular", choices=["circular", "rectangle"])
    parser.add_argument("--pool-length", type=float, default=None)
    parser.add_argument("--pool-width", type=float, default=None)
    parser.add_argument("--surface-depth", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--enable-rayleigh", action="store_true")
    parser.add_argument("--rayleigh-sigma", type=float, default=0.03)
    parser.add_argument("--env-disturbance-scale", type=float, default=0.0)
    parser.add_argument("--env-spectral-enabled", action="store_true")
    parser.add_argument("--wave-hs", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controllers", nargs="*", default=["lqr", "mpc", "rl"])
    parser.add_argument("--planar-dof", action="store_true")
    parser.add_argument("--vehicle-profile-csv", type=str, default=None)
    parser.add_argument("--hold-position", nargs=3, metavar=("X", "Y", "Z"), type=float, default=None)
    parser.add_argument("--position-tolerance-m", type=float, default=0.20)
    parser.add_argument("--attitude-tolerance-deg", type=float, default=12.0)
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
    parser.add_argument("--replay-dir", type=str, default="./training_runs/replays")
    parser.add_argument("--no-replay", action="store_true")
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
    if str(getattr(args, "benchmark_mode", DEFAULT_BENCHMARK_MODE)).lower() == "mission":
        return [[5.0, 0.0, 5.0]]
    return [[1.0, 0.0, 5.0]] if str(getattr(args, "benchmark_mode", DEFAULT_BENCHMARK_MODE)).lower() == "docking" else [[0.0, 0.0, 5.0]]


def _load_scenario_file(path: Path) -> BenchmarkScenario:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario-file must contain a JSON object")

    if "scenario" in data and isinstance(data["scenario"], dict):
        data = data["scenario"]

    valid_fields = {f.name for f in dataclass_fields(BenchmarkScenario)}
    scenario_kwargs = {key: value for key, value in data.items() if key in valid_fields}
    if "waypoints" not in scenario_kwargs:
        raise ValueError("scenario-file must define 'waypoints'")
    if scenario_kwargs.get("vehicle_profile_csv"):
        vehicle_profile_csv = Path(str(scenario_kwargs["vehicle_profile_csv"]))
        if not vehicle_profile_csv.is_absolute():
            vehicle_profile_csv = (path.parent / vehicle_profile_csv).resolve()
        scenario_kwargs["vehicle_profile_csv"] = str(vehicle_profile_csv)
    return BenchmarkScenario(**scenario_kwargs)


def _build_scenario_from_args(args) -> BenchmarkScenario:
    if args.scenario_file:
        return _load_scenario_file(Path(args.scenario_file))

    return BenchmarkScenario(
        waypoints=_load_waypoints(args),
        static_obstacles=[],
        dynamic_obstacles=[],
        dt=float(args.dt),
        max_steps=int(args.max_steps),
        trials=int(args.trials),
        pool_depth=float(args.pool_depth),
        pool_radius=float(args.pool_radius),
        pool_shape=str(args.pool_shape),
        pool_length=(float(args.pool_length) if args.pool_length is not None else None),
        pool_width=(float(args.pool_width) if args.pool_width is not None else None),
        noise_scale=float(args.noise_scale),
        rayleigh_enabled=bool(args.enable_rayleigh),
        rayleigh_sigma=float(args.rayleigh_sigma),
        env_disturbance_scale=float(args.env_disturbance_scale),
        env_spectral_enabled=bool(args.env_spectral_enabled),
        wave_hs=float(args.wave_hs),
        seed=int(args.seed),
        benchmark_mode=str(args.benchmark_mode),
        hold_position=([float(v) for v in args.hold_position] if args.hold_position else None),
        position_tolerance_m=float(args.position_tolerance_m),
        attitude_tolerance_deg=float(args.attitude_tolerance_deg),
        surface_depth=(float(args.surface_depth) if args.surface_depth is not None else None),
        planar_dof=bool(args.planar_dof),
        vehicle_profile_csv=(str(args.vehicle_profile_csv) if args.vehicle_profile_csv else None),
    )


def main() -> int:
    args = _parse_args()
    scenario = _build_scenario_from_args(args)

    benchmark = ControllerBenchmark(
        checkpoint_dir=args.checkpoint_dir,
        replay_dir=args.replay_dir,
        enable_replay_export=(not args.no_replay),
    )
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
