"""
USV Digital Twin - RL training for barrel roll
==============================================

Standalone PPO training loop focused on a barrel roll maneuver.

It reuses the numpy PPO implementation already present in rl_controller.py
and drives the physics engine directly, without the waypoint hierarchy.

Usage:
    python train_barrel_roll_rl.py
    python train_barrel_roll_rl.py --episodes 500 --fresh
    python train_barrel_roll_rl.py --no-eval
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None

from physics_engine import PhysicsEngine, VehicleState
from rl_controller import ActorCritic, PPOUpdater, RolloutBuffer, Adam
from vehicle_profiles import load_taluy_profile


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _to_jsonable(obj):
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
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class BarrelRollConfig:
    phases: Tuple[int, ...] = (1, 2, 3)
    cycles: int = 1
    phase_steps: int = 8192
    episodes: int = 600
    episode_steps: int = 320
    dt: float = 0.02
    seed: int = 42

    pool_depth: float = 10.0
    target_depth: float = 5.0
    target_speed: float = 0.8
    initial_speed: float = 0.6
    initial_depth_jitter: float = 0.15
    initial_attitude_jitter: float = 0.04

    target_turns: float = 1.0
    # Roll-first curriculum: learn to initiate the turn before tightening the target.
    curriculum_stages: Tuple[float, float, float] = (0.05, 0.20, 1.0)
    curriculum_breakpoints: Tuple[float, float] = (0.35, 0.70)

    max_thruster_force: float = 60.0
    max_tilt_deg: float = 60.0
    hover_depth_safety: float = 0.75
    pitch_safety_deg: float = 100.0
    # "per_thruster" preserves the legacy Taluy action layout. "paired"
    # exposes the compact port/starboard command space for experiments.
    action_layout: str = "per_thruster"
    vehicle_profile_source: str = "asset_zoo/vehicles/taluy/taluy_mjcf.xml"
    env_force_gain: float = 1.0
    env_turbulence_gain: float = 1.0
    env_wave_freq: float = 0.8
    enable_domain_randomization: bool = False

    roll_progress_weight: float = 300.0
    roll_track_weight: float = 0.4
    roll_velocity_weight: float = 0.10
    depth_weight: float = 0.05
    speed_weight: float = 0.03
    attitude_penalty: float = 0.008
    stability_penalty: float = 0.003
    action_penalty: float = 0.002
    reverse_roll_penalty_weight: float = 0.0
    terminal_progress_weight: float = 0.0
    completion_bonus: float = 80.0
    # episodic penalty / completion tuning
    episodic_penalty_mult: float = 3.0
    required_min_roll: float = 1.0

    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    value_lr: Optional[float] = None
    value_clip: float = 50.0

    # reward scaling/clipping to limit value magnitudes during training
    reward_scale: float = 1.0
    reward_clip: float = 2000.0

    save_every: int = 25
    eval_episodes: int = 8
    checkpoint_dir: str = "./checkpoints/barrel_roll"
    output_dir: str = "./training_runs"
    fresh: bool = False
    demo_warmstart_rollouts: int = 0
    demo_warmstart_segments: int = 4
    demo_warmstart_top_k: int = 4
    demo_warmstart_epochs: int = 24
    demo_warmstart_batch_size: int = 64
    demo_warmstart_actor_lr: float = 1e-3
    demo_warmstart_success_only: bool = True


@dataclass
class EpisodeSummary:
    episode: int
    stage_turns: float
    steps: int
    reward: float
    success: bool
    roll_turns: float
    final_depth_error: float
    final_pitch_deg: float
    update_metrics: Dict[str, float] = field(default_factory=dict)
    reward_components: Dict[str, float] = field(default_factory=dict)


class BarrelRollEnvironment:
    OBS_DIM = 15
    ACTION_DIM = 7  # default fallback; actual dim set per-instance in __init__

    def __init__(self, config: BarrelRollConfig, physics: Optional[PhysicsEngine] = None):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.action_layout = str(getattr(config, "action_layout", "paired")).strip().lower()

        if physics is None:
            profile = load_taluy_profile(config.vehicle_profile_source)
            self.profile = profile
            self.physics = PhysicsEngine.from_vehicle_profile(
                profile,
                max_thruster_force=config.max_thruster_force,
            )
        else:
            self.physics = physics
            self.profile = None

        # The physics interface currently exposes paired port/starboard control.
        # Keep the legacy per-thruster aggregated layout available for sweeps,
        # but default Taluy training to the compact paired action space.
        n_thrusters = len(getattr(self.physics, 'thrusters', []) or [None, None])
        if n_thrusters < 2:
            n_thrusters = 2
        self.n_thrusters = n_thrusters
        if self.action_layout == "paired":
            self.ACTION_DIM = 7
        elif self.action_layout == "per_thruster":
            self.ACTION_DIM = 3 * n_thrusters + 1
        else:
            raise ValueError(f"Unsupported action_layout: {self.action_layout}")

        self.target_roll = 2.0 * np.pi * float(config.target_turns)
        self.current_target_roll = self.target_roll
        self.roll_progress = 0.0
        self.prev_roll_angle = 0.0
        self.completion_awarded = False
        self.thruster_scale = 1.0

    def reset(self, stage_turns: float, training: bool = True) -> np.ndarray:
        self.current_target_roll = 2.0 * np.pi * float(stage_turns)
        self.roll_progress = 0.0
        self.completion_awarded = False

        depth = self.config.target_depth + self.rng.uniform(-self.config.initial_depth_jitter, self.config.initial_depth_jitter)
        roll = self.rng.uniform(-self.config.initial_attitude_jitter, self.config.initial_attitude_jitter)
        pitch = self.rng.uniform(-self.config.initial_attitude_jitter, self.config.initial_attitude_jitter)
        yaw = self.rng.uniform(-np.pi, np.pi)
        surge = self.config.initial_speed + self.rng.uniform(-0.08, 0.08)

        state = VehicleState(
            z=depth,
            phi=roll,
            tht=pitch,
            psi=yaw,
            u=surge,
        )
        self.physics.reset(state)
        self.prev_roll_angle = float(state.phi)

        if training:
            if self.config.enable_domain_randomization:
                self.thruster_scale = float(self.rng.uniform(0.9, 1.1))
                self.physics.env_force_gain = float(self.config.env_force_gain * self.rng.uniform(0.85, 1.15))
                self.physics.env_turbulence_gain = float(self.config.env_turbulence_gain * self.rng.uniform(0.85, 1.15))
                self.physics.env_wave_freq = float(self.config.env_wave_freq * self.rng.uniform(0.85, 1.15))
            else:
                self.thruster_scale = 1.0
                self.physics.env_force_gain = float(self.config.env_force_gain)
                self.physics.env_turbulence_gain = float(self.config.env_turbulence_gain)
                self.physics.env_wave_freq = float(self.config.env_wave_freq)
        else:
            self.thruster_scale = 1.0

        return self._observation()

    def _observation(self) -> np.ndarray:
        state = self.physics.state
        depth_error = state.z - self.config.target_depth
        roll_rate_target = self.current_target_roll / max(1.0, self.config.episode_steps * self.config.dt)

        return np.array([
            depth_error / max(0.25, self.config.hover_depth_safety),
            (state.u - self.config.target_speed) / max(0.2, self.config.target_speed),
            state.v / max(0.2, self.config.target_speed),
            state.w / max(0.2, self.config.target_speed),
            state.p / max(0.2, roll_rate_target),
            state.q / max(0.2, roll_rate_target),
            state.r / max(0.2, roll_rate_target),
            math.sin(state.phi),
            math.cos(state.phi),
            math.sin(state.tht),
            math.cos(state.tht),
            math.sin(state.psi),
            math.cos(state.psi),
            self.roll_progress / max(1e-6, self.current_target_roll),
            (self.current_target_roll - self.roll_progress) / max(1e-6, self.current_target_roll),
        ], dtype=float)

    def _action_to_step(self, action: np.ndarray) -> Dict[str, float]:
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        tilt_limit = np.radians(self.config.max_tilt_deg)
        power_scale = float(self.thruster_scale)

        # Legacy dual-thruster format (7 elements)
        if action.size == 7:
            return {
                "thruster_power": float(np.clip(power_scale * action[0], -1.0, 1.0)),
                "thruster_theta": float((action[1] + 1.0) * 0.5 * tilt_limit),
                "thruster_phi": float((action[2] + 1.0) * 0.5 * 2.0 * np.pi),
                "ballast_cmd": float(action[3]),
                "thruster2_power": float(np.clip(power_scale * action[4], -1.0, 1.0)),
                "thruster2_theta": float((action[5] + 1.0) * 0.5 * tilt_limit),
                "thruster2_phi": float((action[6] + 1.0) * 0.5 * 2.0 * np.pi),
            }

        # Per-thruster format: [p0,t0,phi0, p1,t1,phi1, ..., ballast]
        # Expect length = 3*N + 1
        if (action.size - 1) % 3 == 0 and action.size >= 4:
            n_in = (action.size - 1) // 3
            thrusters = getattr(self.physics, 'thrusters', None)
            if thrusters is None:
                # fallback: aggregate evenly into two channels
                half = max(1, n_in // 2)
                p_port = float(np.sum(action[0:3*half:3]))
                p_star = float(np.sum(action[3*half:3*n_in:3]))
                t1 = float(np.mean(action[1:3*half:3]) if half > 0 else 0.0)
                t2 = float(np.mean(action[3*half+1:3*n_in:3]) if (n_in-half) > 0 else 0.0)
                f1 = float(np.mean((action[2:3*half:3] + 1.0) * 0.5 * 2.0 * np.pi) if half > 0 else 0.0)
                f2 = float(np.mean((action[3*half+2:3*n_in:3] + 1.0) * 0.5 * 2.0 * np.pi) if (n_in-half) > 0 else 0.0)
            else:
                # map thruster indices to port/star by position_body.y sign
                port_idxs = [i for i, th in enumerate(thrusters) if float(th.position_body[1]) > 0.0]
                star_idxs = [i for i, th in enumerate(thrusters) if float(th.position_body[1]) <= 0.0]
                if not port_idxs:
                    port_idxs = [0]
                if not star_idxs:
                    star_idxs = [min(len(thrusters)-1, 1)]

                # accumulate per-side
                p_port = 0.0
                p_star = 0.0
                t1_vals = []
                t2_vals = []
                f1_vals = []
                f2_vals = []
                for i in range(len(thrusters)):
                    if i < n_in:
                        p = float(action[3*i + 0])
                        t = float((action[3*i + 1] + 1.0) * 0.5 * tilt_limit)
                        f = float((action[3*i + 2] + 1.0) * 0.5 * 2.0 * np.pi)
                        if i in port_idxs:
                            p_port += p
                            t1_vals.append(t)
                            f1_vals.append(f)
                        else:
                            p_star += p
                            t2_vals.append(t)
                            f2_vals.append(f)

                t1 = float(np.mean(t1_vals) if t1_vals else 0.0)
                t2 = float(np.mean(t2_vals) if t2_vals else 0.0)
                f1 = float(np.mean(f1_vals) if f1_vals else 0.0)
                f2 = float(np.mean(f2_vals) if f2_vals else 0.0)

            return {
                "thruster_power": float(np.clip(power_scale * p_port, -1.0, 1.0)),
                "thruster_theta": float(np.clip(t1, 0.0, tilt_limit)),
                "thruster_phi": float(f1),
                "ballast_cmd": float(action[-1]),
                "thruster2_power": float(np.clip(power_scale * p_star, -1.0, 1.0)),
                "thruster2_theta": float(np.clip(t2, 0.0, tilt_limit)),
                "thruster2_phi": float(f2),
            }

        # fallback to zeros
        return {
            "thruster_power": 0.0,
            "thruster_theta": 0.0,
            "thruster_phi": 0.0,
            "ballast_cmd": 0.0,
            "thruster2_power": 0.0,
            "thruster2_theta": 0.0,
            "thruster2_phi": 0.0,
        }

    def _current_stage_turns(self, episode_index: int, total_episodes: int) -> float:
        first_break = max(1, int(math.ceil(total_episodes * self.config.curriculum_breakpoints[0])))
        second_break = max(first_break + 1, int(math.ceil(total_episodes * self.config.curriculum_breakpoints[1])))
        if episode_index < first_break:
            return float(self.config.curriculum_stages[0])
        if episode_index < second_break:
            return float(self.config.curriculum_stages[1])
        return float(self.config.curriculum_stages[2])

    def step(self, action: np.ndarray, step_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        prev_roll = self.prev_roll_angle
        command = self._action_to_step(action)
        self.physics.step(
            dt=self.config.dt,
            env_current_world=np.zeros(3, dtype=float),
            env_turbulence=0.0,
            env_harmonics=None,
            **command,
        )

        state = self.physics.state
        roll_delta = _wrap_angle(state.phi - prev_roll)
        self.roll_progress += roll_delta
        self.prev_roll_angle = float(state.phi)

        elapsed = (step_index + 1) * self.config.dt
        desired_roll = min(self.current_target_roll, self.current_target_roll * elapsed / max(self.config.episode_steps * self.config.dt, 1e-6))
        roll_error = desired_roll - self.roll_progress
        depth_error = state.z - self.config.target_depth
        speed_error = state.u - self.config.target_speed

        roll_rate_target = self.current_target_roll / max(self.config.episode_steps * self.config.dt, 1e-6)
        target_tracking = math.exp(-abs(roll_error) / max(0.35, 0.1 * self.current_target_roll))
        depth_reward = math.exp(-((depth_error / max(self.config.hover_depth_safety, 1e-6)) ** 2))
        speed_reward = math.exp(-((speed_error / max(0.25, self.config.target_speed)) ** 2))
        phase_ratio = min(1.0, self.current_target_roll / max(self.target_roll, 1e-6))
        auxiliary_scale = 0.35 + 0.65 * phase_ratio

        # compute component-level values for logging and diagnostic
        # Reward forward progress but explicitly penalize backtracking so the
        # policy cannot farm shaping with forward/back oscillations.
        roll_progress_reward = max(0.0, roll_delta) / max(1e-6, self.current_target_roll)
        reverse_roll_penalty = max(0.0, -roll_delta) / max(1e-6, self.current_target_roll)
        roll_rate_target = self.current_target_roll / max(self.config.episode_steps * self.config.dt, 1e-6)
        # reward positive roll rate in the desired direction, penalize reversals/oscillation
        roll_velocity_raw = float(state.p)
        roll_velocity_reward = float(max(0.0, roll_velocity_raw / max(1e-6, roll_rate_target)))

        attitude_penalty = (state.tht ** 2 + 0.5 * state.psi ** 2)
        stability_penalty = 0.10 * state.q ** 2 + 0.10 * state.r ** 2 + 0.05 * state.v ** 2 + 0.05 * state.w ** 2
        action_penalty = float(np.sum(np.square(np.clip(action, -1.0, 1.0))))

        # components
        roll_progress_component = float(self.config.roll_progress_weight * roll_progress_reward)
        roll_track_component = float(self.config.roll_track_weight * target_tracking)
        roll_velocity_component = float(self.config.roll_velocity_weight * roll_velocity_reward)
        reverse_roll_component = float(-self.config.reverse_roll_penalty_weight * reverse_roll_penalty)
        depth_component = float(self.config.depth_weight * auxiliary_scale * depth_reward)
        speed_component = float(self.config.speed_weight * auxiliary_scale * speed_reward)
        attitude_component = float(-self.config.attitude_penalty * auxiliary_scale * attitude_penalty)
        stability_component = float(-self.config.stability_penalty * auxiliary_scale * stability_penalty)
        action_component = float(-self.config.action_penalty * action_penalty)

        # dense shaping reward is scaled independently from terminal signals
        shaping_reward = (
            roll_progress_component
            + roll_track_component
            + roll_velocity_component
            + reverse_roll_component
            + depth_component
            + speed_component
            + attitude_component
            + stability_component
            + action_component
        )

        reward = float(shaping_reward)
        reward = float(reward * float(self.config.reward_scale))
        if float(self.config.reward_clip) > 0.0:
            reward = float(np.clip(reward, -float(self.config.reward_clip), float(self.config.reward_clip)))

        success = False
        if not self.completion_awarded and self.roll_progress >= self.current_target_roll:
            if abs(depth_error) <= self.config.hover_depth_safety and abs(state.tht) <= np.radians(30.0):
                reward += self.config.completion_bonus
                self.completion_awarded = True
                success = True

        # EPISODIC PENALTY: applied only on final step of episode
        episodic_penalty = 0.0
        terminal_progress_component = 0.0
        if (step_index + 1) == self.config.episode_steps:
            completed_ratio = float(np.clip(self.roll_progress / max(1e-6, self.current_target_roll), 0.0, 1.0))
            terminal_progress_component = float(self.config.terminal_progress_weight * (completed_ratio - 1.0))
            reward = reward + terminal_progress_component

            required_min_roll = self.current_target_roll * float(self.config.required_min_roll)
            if self.roll_progress < required_min_roll:
                insufficient_roll = required_min_roll - self.roll_progress
                terminal_scale = 0.25 + 0.75 * min(1.0, self.current_target_roll / max(self.target_roll, 1e-6))
                episodic_penalty = float(insufficient_roll * float(self.config.episodic_penalty_mult) * terminal_scale)
                reward = reward - episodic_penalty

            # keep terminal penalties visible even when shaping is scaled down
            if float(self.config.reward_clip) > 0.0:
                reward = float(np.clip(reward, -float(self.config.reward_clip), float(self.config.reward_clip)))

        pitch_limit = np.radians(self.config.pitch_safety_deg)
        done = False
        if abs(depth_error) > 3.0:
            done = True
        if abs(state.tht) > pitch_limit:
            done = True
        if not np.isfinite(self.roll_progress):
            done = True

        info = {
            "roll_progress": float(self.roll_progress),
            "roll_turns": float(self.roll_progress / (2.0 * np.pi)),
            "desired_roll": float(desired_roll),
            "roll_rate_target": float(roll_rate_target),
            "depth_error": float(depth_error),
            "speed_error": float(speed_error),
            "success": float(success),
        }

        info["reward_components"] = {
            "roll_progress": roll_progress_component,
            "roll_track": roll_track_component,
            "roll_velocity": roll_velocity_component,
            "reverse_roll": reverse_roll_component,
            "depth": depth_component,
            "speed": speed_component,
            "attitude": attitude_component,
            "stability": stability_component,
            "action": action_component,
            "terminal_progress": terminal_progress_component,
            "episodic_penalty": episodic_penalty,
            "total": float(reward),
        }

        return self._observation(), float(reward), done, info


class BarrelRollTrainer:
    def __init__(self, config: BarrelRollConfig):
        self.config = config
        np.random.seed(config.seed)

        self.env = BarrelRollEnvironment(config)
        # bound critic outputs using config.value_clip to avoid extreme value predictions
        self.network = ActorCritic(
            obs_dim=self.env.OBS_DIM,
            action_dim=self.env.ACTION_DIM,
            hidden=[64, 64],
            value_scale=float(config.value_clip) if getattr(config, 'value_clip', None) is not None else None,
        )
        self.updater = PPOUpdater(
            self.network,
            lr=config.lr,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.25,
            n_epochs=3,
            batch_size=64,
            value_lr=config.value_lr,
            value_clip=config.value_clip,
        )

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.latest_prefix = str(self.checkpoint_dir / "barrel_roll_latest")
        self.best_prefix = str(self.checkpoint_dir / "barrel_roll_best")
        self.report_path = self.output_dir / "barrel_roll_training_report.json"
        self.curve_dir = self.output_dir / "barrel_roll_curves"
        self.curve_dir.mkdir(parents=True, exist_ok=True)
        self.curve_report_path = self.curve_dir / "barrel_roll_curves.json"
        self.curve_plot_path = self.curve_dir / "barrel_roll_curves.png"

    def _save(self, prefix: str) -> None:
        self.network.save(prefix)

    def _load_if_requested(self) -> None:
        backbone_path = Path(self.latest_prefix + "_backbone.pkl")
        if backbone_path.exists():
            self.network.load(self.latest_prefix)

    def _select_stage_turns(self, episode_index: int) -> float:
        return self.env._current_stage_turns(episode_index, self.config.episodes)

    def _policy_mean_action(self, obs: np.ndarray) -> np.ndarray:
        mean, _, _ = self.network.forward(obs, normalize=True, update_obs_stats=False)
        return np.asarray(mean, dtype=float)

    @staticmethod
    def _episode_score(summary: EpisodeSummary) -> Tuple[float, float, float]:
        return (
            1.0 if summary.success else 0.0,
            float(summary.roll_turns),
            float(summary.reward),
        )

    @staticmethod
    def _aggregate_score(metrics: Dict[str, float]) -> Tuple[float, float, float]:
        return (
            float(metrics.get("success_rate", 0.0)),
            float(metrics.get("roll_turns_mean", 0.0)),
            float(metrics.get("reward_mean", 0.0)),
        )

    def _warmstart_report_path(self) -> Path:
        return self.output_dir / "barrel_roll_warmstart_report.json"

    def collect_demo_trajectories(
        self,
        rollouts: int,
        segments: int,
        top_k: int,
        success_only: bool = True,
    ) -> Dict[str, object]:
        demo_env = BarrelRollEnvironment(self.config)
        demo_env.rng = np.random.default_rng(self.config.seed + 101)
        rng = np.random.default_rng(self.config.seed + 2025)
        segments = max(1, int(segments))
        seg_len = max(1, int(math.ceil(self.config.episode_steps / segments)))

        candidates: List[Dict[str, object]] = []
        success_count = 0

        for rollout_idx in range(max(1, int(rollouts))):
            obs = demo_env.reset(float(self.config.target_turns), training=False)
            seq = rng.uniform(-1.0, 1.0, size=(segments, demo_env.ACTION_DIM))
            obs_trace: List[np.ndarray] = []
            action_trace: List[np.ndarray] = []
            total_reward = 0.0
            success = False
            last_info: Dict[str, float] = {"roll_turns": 0.0, "depth_error": 0.0}

            for step in range(self.config.episode_steps):
                action = np.asarray(seq[min(step // seg_len, segments - 1)], dtype=float)
                obs_trace.append(np.asarray(obs, dtype=float))
                action_trace.append(action.copy())
                obs, reward, done, info = demo_env.step(action, step)
                total_reward += float(reward)
                success = success or bool(info.get("success", 0.0))
                last_info = info
                if done:
                    break

            if success:
                success_count += 1

            candidates.append(
                {
                    "rollout": rollout_idx + 1,
                    "success": bool(success),
                    "roll_turns": float(last_info.get("roll_turns", 0.0)),
                    "reward": float(total_reward),
                    "depth_error_abs": abs(float(last_info.get("depth_error", 0.0))),
                    "pitch_deg_abs": abs(float(np.degrees(demo_env.physics.state.tht))),
                    "steps": len(obs_trace),
                    "observations": np.asarray(obs_trace, dtype=float),
                    "actions": np.asarray(action_trace, dtype=float),
                }
            )

        candidates.sort(
            key=lambda item: (
                1.0 if bool(item["success"]) else 0.0,
                float(item["roll_turns"]),
                float(item["reward"]),
            ),
            reverse=True,
        )
        if success_only and success_count > 0:
            selected = [item for item in candidates if bool(item["success"])][: max(1, int(top_k))]
        else:
            selected = candidates[: max(1, int(top_k))]

        if selected:
            demo_obs = np.concatenate([item["observations"] for item in selected], axis=0)
            demo_actions = np.concatenate([item["actions"] for item in selected], axis=0)
        else:
            demo_obs = np.zeros((0, self.env.OBS_DIM), dtype=float)
            demo_actions = np.zeros((0, self.env.ACTION_DIM), dtype=float)

        selected_summary = [
            {
                "rollout": int(item["rollout"]),
                "success": bool(item["success"]),
                "roll_turns": float(item["roll_turns"]),
                "reward": float(item["reward"]),
                "depth_error_abs": float(item["depth_error_abs"]),
                "pitch_deg_abs": float(item["pitch_deg_abs"]),
                "steps": int(item["steps"]),
            }
            for item in selected
        ]

        return {
            "rollouts": int(rollouts),
            "segments": int(segments),
            "success_count": int(success_count),
            "best_roll_turns": float(candidates[0]["roll_turns"]) if candidates else 0.0,
            "selected_count": int(len(selected)),
            "selected_success_count": int(sum(1 for item in selected if bool(item["success"]))),
            "selected": selected_summary,
            "observations": demo_obs,
            "actions": demo_actions,
        }

    def warmstart_actor_from_demos(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        epochs: int,
        batch_size: int,
        actor_lr: float,
    ) -> Dict[str, float]:
        obs_arr = np.asarray(observations, dtype=float)
        act_arr = np.asarray(actions, dtype=float)
        if obs_arr.size == 0 or act_arr.size == 0:
            return {"samples": 0, "epochs": 0, "loss_initial": float("nan"), "loss_final": float("nan")}

        for obs in obs_arr:
            self.network.normalize_obs(np.asarray(obs, dtype=float), update_stats=True)

        optimizer = Adam(
            self.network.backbone.parameters() + self.network.actor_head.parameters(),
            lr=float(actor_lr),
        )

        def _loss_for_batch(obs_b: np.ndarray, act_b: np.ndarray) -> float:
            norm_obs = np.asarray([self.network.normalize_obs(obs, update_stats=False) for obs in obs_b], dtype=float)
            h = self.network.backbone.forward(norm_obs)
            mean = np.tanh(self.network.actor_head.forward(h))
            return float(0.5 * np.mean(np.square(mean - act_b)))

        loss_initial = _loss_for_batch(obs_arr, act_arr)
        losses: List[float] = []

        n_samples = len(obs_arr)
        batch_size = max(1, int(batch_size))
        epochs = max(1, int(epochs))
        for _ in range(epochs):
            perm = np.random.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                obs_b = obs_arr[idx]
                act_b = act_arr[idx]
                norm_obs = np.asarray([self.network.normalize_obs(obs, update_stats=False) for obs in obs_b], dtype=float)

                h = self.network.backbone.forward(norm_obs)
                mean = np.tanh(self.network.actor_head.forward(h))
                diff = mean - act_b
                loss = float(0.5 * np.mean(np.square(diff)))
                losses.append(loss)

                grad_mean = diff / max(1.0, float(diff.shape[0] * diff.shape[1]))
                grad_pre_tanh = grad_mean * (1.0 - mean ** 2)

                self.network.zero_grad()
                grad_backbone = self.network.actor_head.backward(grad_pre_tanh)
                self.network.backbone.backward(grad_backbone)
                optimizer.step(max_grad_norm=0.5)
                optimizer.zero_grad()

        loss_final = _loss_for_batch(obs_arr, act_arr)
        return {
            "samples": int(n_samples),
            "epochs": int(epochs),
            "loss_initial": float(loss_initial),
            "loss_final": float(loss_final),
            "loss_mean": float(np.mean(losses)) if losses else float("nan"),
        }

    def demo_warmstart(self) -> Dict[str, object]:
        if int(getattr(self.config, "demo_warmstart_rollouts", 0)) <= 0:
            return {}

        search = self.collect_demo_trajectories(
            rollouts=int(self.config.demo_warmstart_rollouts),
            segments=int(self.config.demo_warmstart_segments),
            top_k=int(self.config.demo_warmstart_top_k),
            success_only=bool(self.config.demo_warmstart_success_only),
        )
        fit = self.warmstart_actor_from_demos(
            observations=np.asarray(search.pop("observations"), dtype=float),
            actions=np.asarray(search.pop("actions"), dtype=float),
            epochs=int(self.config.demo_warmstart_epochs),
            batch_size=int(self.config.demo_warmstart_batch_size),
            actor_lr=float(self.config.demo_warmstart_actor_lr),
        )
        evaluation = self.evaluate(self.config.eval_episodes)
        report = {
            "search": search,
            "fit": fit,
            "evaluation": evaluation,
        }
        self._warmstart_report_path().write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")
        return report

    def run_episode(self, episode_index: int, training: bool = True) -> EpisodeSummary:
        stage_turns = self._select_stage_turns(episode_index) if training else float(self.config.target_turns)
        obs = self.env.reset(stage_turns=stage_turns, training=training)
        buffer = RolloutBuffer(capacity=self.config.episode_steps)

        episode_reward = 0.0
        success = False
        info = {
            "roll_turns": 0.0,
            "depth_error": 0.0,
            "success": 0.0,
        }
        # accumulator for per-step reward components
        comp_acc: Dict[str, float] = {}

        for step_index in range(self.config.episode_steps):
            if training:
                action, log_prob, value = self.network.act(obs, update_obs_stats=True)
            else:
                action = self._policy_mean_action(obs)
                log_prob = 0.0
                _, _, value = self.network.forward(obs, normalize=True, update_obs_stats=False)

            next_obs, reward, done, info = self.env.step(action, step_index)
            if training:
                # include info dict from environment for diagnostic correlation
                buffer.add(obs, action, log_prob, reward, value, done, info)
            obs = next_obs
            episode_reward += reward
            # accumulate reward components if provided by env
            comps = info.get("reward_components") if isinstance(info, dict) else None
            if isinstance(comps, dict):
                for k, v in comps.items():
                    comp_acc[k] = comp_acc.get(k, 0.0) + float(v)
            success = bool(info.get("success", 0.0)) or success

            if done:
                break

        if training and len(buffer.rewards) > 0:
            last_value = 0.0 if success else float(self.network.forward(obs, normalize=True, update_obs_stats=False)[2])
            buffer.finalize(last_value, gamma=self.config.gamma, lam=self.config.lam)
            update_metrics = self.updater.update(buffer)
        else:
            update_metrics = {}

        summary = EpisodeSummary(
            episode=episode_index + 1,
            stage_turns=stage_turns,
            steps=step_index + 1,
            reward=float(episode_reward),
            success=bool(success),
            roll_turns=float(info.get("roll_turns", 0.0)),
            final_depth_error=float(info.get("depth_error", 0.0)),
            final_pitch_deg=float(np.degrees(self.env.physics.state.tht)),
            update_metrics=update_metrics,
            reward_components=comp_acc,
        )
        return summary

    def evaluate(self, episodes: int) -> Dict[str, float]:
        totals = {
            "reward": 0.0,
            "success": 0.0,
            "roll_turns": 0.0,
            "depth_error_abs": 0.0,
            "pitch_deg_abs": 0.0,
        }

        for episode_index in range(episodes):
            summary = self.run_episode(episode_index, training=False)
            totals["reward"] += summary.reward
            totals["success"] += 1.0 if summary.success else 0.0
            totals["roll_turns"] += summary.roll_turns
            totals["depth_error_abs"] += abs(summary.final_depth_error)
            totals["pitch_deg_abs"] += abs(summary.final_pitch_deg)

        return {
            "episodes": int(episodes),
            "reward_mean": totals["reward"] / max(1, episodes),
            "success_rate": totals["success"] / max(1, episodes),
            "roll_turns_mean": totals["roll_turns"] / max(1, episodes),
            "depth_error_abs_mean": totals["depth_error_abs"] / max(1, episodes),
            "pitch_deg_abs_mean": totals["pitch_deg_abs"] / max(1, episodes),
        }

    @staticmethod
    def _moving_average(values: List[float], window: int = 10) -> List[float]:
        if not values:
            return []
        window = max(1, int(window))
        result: List[float] = []
        for idx in range(len(values)):
            start = max(0, idx + 1 - window)
            result.append(float(np.mean(values[start:idx + 1])))
        return result

    def _save_training_curves(self, episode_history: List[Dict[str, object]], evaluation: Dict[str, float] | None) -> None:
        episodes = [int(item["episode"]) for item in episode_history]
        rewards = [float(item["reward"]) for item in episode_history]
        roll_turns = [float(item["roll_turns"]) for item in episode_history]
        depth_error = [abs(float(item["final_depth_error"])) for item in episode_history]
        pitch_deg = [abs(float(item["final_pitch_deg"])) for item in episode_history]
        success = [1.0 if bool(item["success"]) else 0.0 for item in episode_history]
        losses = [float(item["update_metrics"].get("loss", float("nan"))) if item["update_metrics"] else float("nan") for item in episode_history]
        policy_losses = [float(item["update_metrics"].get("policy_loss", float("nan"))) if item["update_metrics"] else float("nan") for item in episode_history]
        value_losses = [float(item["update_metrics"].get("value_loss", float("nan"))) if item["update_metrics"] else float("nan") for item in episode_history]
        entropies = [float(item["update_metrics"].get("entropy", float("nan"))) if item["update_metrics"] else float("nan") for item in episode_history]

        curve_data = {
            "episodes": episodes,
            "reward": rewards,
            "reward_ma10": self._moving_average(rewards, 10),
            "roll_turns": roll_turns,
            "roll_turns_ma10": self._moving_average(roll_turns, 10),
            "depth_error_abs": depth_error,
            "depth_error_abs_ma10": self._moving_average(depth_error, 10),
            "pitch_deg_abs": pitch_deg,
            "pitch_deg_abs_ma10": self._moving_average(pitch_deg, 10),
            "success": success,
            "success_ma10": self._moving_average(success, 10),
            "loss": losses,
            "policy_loss": policy_losses,
            "value_loss": value_losses,
            "entropy": entropies,
            "evaluation": evaluation,
        }

        # extract per-episode reward components if available
        comp_keys = [
            "roll_progress",
            "roll_track",
            "roll_velocity",
            "reverse_roll",
            "depth",
            "speed",
            "attitude",
            "stability",
            "action",
            "terminal_progress",
            "episodic_penalty",
            "total",
        ]
        comps: Dict[str, List[float]] = {k: [] for k in comp_keys}
        for item in episode_history:
            rc = item.get("reward_components") or {}
            for k in comp_keys:
                comps[k].append(float(rc.get(k, 0.0)))

        curve_data["reward_components"] = comps

        self.curve_report_path.write_text(json.dumps(_to_jsonable(curve_data), indent=2), encoding="utf-8")

        if plt is None:
            return

        fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
        axes = axes.ravel()

        axes[0].plot(episodes, rewards, alpha=0.35, label="reward")
        axes[0].plot(episodes, curve_data["reward_ma10"], linewidth=2, label="reward ma10")
        axes[0].set_title("Reward")
        axes[0].legend()

        axes[1].plot(episodes, roll_turns, alpha=0.35, label="roll turns")
        axes[1].plot(episodes, curve_data["roll_turns_ma10"], linewidth=2, label="roll turns ma10")
        axes[1].set_title("Roll progress")
        axes[1].legend()

        axes[2].plot(episodes, depth_error, alpha=0.35, label="depth error abs")
        axes[2].plot(episodes, curve_data["depth_error_abs_ma10"], linewidth=2, label="depth error abs ma10")
        axes[2].set_title("Depth error")
        axes[2].legend()

        axes[3].plot(episodes, pitch_deg, alpha=0.35, label="pitch abs deg")
        axes[3].plot(episodes, curve_data["pitch_deg_abs_ma10"], linewidth=2, label="pitch abs deg ma10")
        axes[3].set_title("Pitch magnitude")
        axes[3].legend()

        axes[4].plot(episodes, success, alpha=0.35, label="success")
        axes[4].plot(episodes, curve_data["success_ma10"], linewidth=2, label="success ma10")
        axes[4].set_title("Success rate")
        axes[4].set_ylim(-0.05, 1.05)
        axes[4].legend()

        axes[5].plot(episodes, losses, alpha=0.35, label="loss")
        axes[5].plot(episodes, policy_losses, alpha=0.35, label="policy loss")
        axes[5].plot(episodes, value_losses, alpha=0.35, label="value loss")
        axes[5].plot(episodes, entropies, alpha=0.35, label="entropy")
        axes[5].set_title("PPO metrics")
        axes[5].legend()

        if evaluation is not None:
            fig.suptitle(
                f"Barrel Roll Training Curves | eval_success={evaluation['success_rate']:.2f} | "
                f"eval_reward={evaluation['reward_mean']:.2f}",
                fontsize=14,
            )
        else:
            fig.suptitle("Barrel Roll Training Curves", fontsize=14)

        fig.savefig(self.curve_plot_path, dpi=160)
        plt.close(fig)

    def train(self, no_eval: bool = False) -> Dict[str, object]:
        print("\n" + "=" * 64)
        print(f"  RL TRAINING - BARREL ROLL ({self.config.episodes} episodes)")
        print("=" * 64 + "\n")

        best_score = (-float("inf"), -float("inf"), -float("inf"))
        episode_history: List[Dict[str, object]] = []

        for episode_index in range(self.config.episodes):
            summary = self.run_episode(episode_index, training=True)
            episode_history.append(asdict(summary))

            print(
                f"EP {summary.episode:04d} | stage={summary.stage_turns:.2f} turns | "
                f"steps={summary.steps:03d} | reward={summary.reward:8.3f} | "
                f"roll={summary.roll_turns:6.3f} turns | success={summary.success}"
            )

            if (episode_index + 1) % self.config.save_every == 0:
                self._save(self.latest_prefix)

            summary_score = self._episode_score(summary)
            if summary_score > best_score:
                best_score = summary_score
                self._save(self.best_prefix)

        self._save(self.latest_prefix)

        evaluation = None
        if not no_eval:
            evaluation = self.evaluate(self.config.eval_episodes)
            print(
                f"\nEvaluation | success_rate={evaluation['success_rate']:.2f} | "
                f"reward_mean={evaluation['reward_mean']:.3f} | "
                f"roll_turns_mean={evaluation['roll_turns_mean']:.3f}"
            )

            evaluation_score = self._aggregate_score(evaluation)
            if evaluation_score > best_score:
                best_score = evaluation_score
                self._save(self.best_prefix)

        report = {
            "config": asdict(self.config),
            "episodes": episode_history,
            "evaluation": evaluation,
        }
        self.report_path.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")
        self._save_training_curves(episode_history, evaluation)
        print(f"\nReport saved to {self.report_path}")
        print(f"Curve log saved to {self.curve_report_path}")
        if plt is not None:
            print(f"Curve plot saved to {self.curve_plot_path}")
        print(f"Checkpoints saved to {self.checkpoint_dir.resolve()}")
        return report


def build_stack(config: BarrelRollConfig):
    np.random.seed(config.seed)
    trainer = BarrelRollTrainer(config)
    return trainer.env.physics.geo, trainer.env.physics, trainer.env, trainer.network, trainer.updater, trainer


def _active_phase_turns(config: BarrelRollConfig, phase: int) -> Tuple[float, str]:
    if phase == 1:
        return float(config.curriculum_stages[0]), "phase_1"
    if phase == 2:
        return float(config.curriculum_stages[1]), "phase_2"
    if phase == 3:
        return float(config.curriculum_stages[2]), "phase_3"
    raise ValueError(f"Unsupported phase: {phase}")


def _maybe_flush_buffer(updater: PPOUpdater, buffer: RolloutBuffer, gamma: float, lam: float, last_value: float):
    if len(buffer.rewards) == 0:
        return None
    buffer.finalize(last_value, gamma=gamma, lam=lam)
    metrics = updater.update(buffer)
    buffer.clear()
    return metrics


def run_phase(
    config: BarrelRollConfig,
    hrl: BarrelRollTrainer,
    phase: int,
):
    stage_turns, phase_name = _active_phase_turns(config, phase)

    total_steps = 0
    episode = 0
    phase_metrics: List[Dict] = []
    episode_history: List[Dict[str, object]] = []
    reward_sum = 0.0
    reward_count = 0
    success_count = 0
    roll_sum = 0.0
    depth_error_sum = 0.0
    pitch_deg_sum = 0.0

    while total_steps < config.phase_steps:
        obs = hrl.env.reset(stage_turns, training=True)
        buffer = RolloutBuffer(capacity=config.episode_steps)
        episode += 1
        episode_reward = 0.0
        last_info: Dict[str, float] = {"roll_turns": 0.0, "depth_error": 0.0, "success": 0.0}
        success = False
        comp_acc: Dict[str, float] = {}

        for step in range(config.episode_steps):
            action, log_prob, value = hrl.network.act(obs, update_obs_stats=True)
            next_obs, reward, done, info = hrl.env.step(action, step)
            # include per-step info for diagnostics
            buffer.add(obs, action, log_prob, reward, value, done, info)

            obs = next_obs
            episode_reward += float(reward)
            last_info = info
            # accumulate reward components if present
            comps = info.get("reward_components") if isinstance(info, dict) else None
            if isinstance(comps, dict):
                for k, v in comps.items():
                    comp_acc[k] = comp_acc.get(k, 0.0) + float(v)
            success = success or bool(info.get("success", 0.0))

            total_steps += 1
            if done or total_steps >= config.phase_steps:
                break

        last_value = 0.0 if success else float(hrl.network.forward(obs, normalize=True, update_obs_stats=False)[2])
        maybe_metrics = _maybe_flush_buffer(hrl.updater, buffer, config.gamma, config.lam, last_value)
        if maybe_metrics:
            phase_metrics.append(maybe_metrics)

        reward_sum += episode_reward
        reward_count += 1
        success_count += 1 if success else 0
        roll_sum += float(last_info.get("roll_turns", 0.0))
        depth_error_sum += abs(float(last_info.get("depth_error", 0.0)))
        pitch_deg_sum += abs(float(np.degrees(hrl.env.physics.state.tht)))

        episode_history.append(
            {
                "phase": phase,
                "phase_name": phase_name,
                "episode": episode,
                "steps": step + 1,
                "reward": float(episode_reward),
                "success": bool(success),
                "roll_turns": float(last_info.get("roll_turns", 0.0)),
                "final_depth_error": float(last_info.get("depth_error", 0.0)),
                "final_pitch_deg": float(np.degrees(hrl.env.physics.state.tht)),
                "update_metrics": maybe_metrics or {},
                "reward_components": comp_acc,
            }
        )

    latest_prefix = str(Path(config.checkpoint_dir) / "barrel_roll_latest")
    phase_prefix = str(Path(config.checkpoint_dir) / f"barrel_roll_phase_{phase}")
    hrl._save(phase_prefix)
    hrl._save(latest_prefix)

    phase_summary = {
        "phase": phase,
        "phase_name": phase_name,
        "stage_turns": stage_turns,
        "steps": total_steps,
        "episodes": episode,
        "reward_mean": reward_sum / max(1, reward_count),
        "success_rate": success_count / max(1, reward_count),
        "roll_turns_mean": roll_sum / max(1, reward_count),
        "depth_error_abs_mean": depth_error_sum / max(1, reward_count),
        "pitch_deg_abs_mean": pitch_deg_sum / max(1, reward_count),
        "updates": phase_metrics,
    }
    return phase_summary, episode_history


def evaluate(hrl: BarrelRollTrainer, config: BarrelRollConfig):
    totals = {
        "reward": 0.0,
        "success": 0.0,
        "roll_turns": 0.0,
        "depth_error_abs": 0.0,
        "pitch_deg_abs": 0.0,
    }

    for episode_index in range(config.eval_episodes):
        obs = hrl.env.reset(float(config.target_turns), training=False)
        episode_reward = 0.0
        last_info: Dict[str, float] = {"roll_turns": 0.0, "depth_error": 0.0, "success": 0.0}
        success = False

        for step in range(config.episode_steps):
            mean, _, _ = hrl.network.forward(obs, normalize=True, update_obs_stats=False)
            next_obs, reward, done, info = hrl.env.step(mean, step)
            obs = next_obs
            episode_reward += float(reward)
            last_info = info
            success = success or bool(info.get("success", 0.0))
            if done:
                break

        totals["reward"] += episode_reward
        totals["success"] += 1.0 if success else 0.0
        totals["roll_turns"] += float(last_info.get("roll_turns", 0.0))
        totals["depth_error_abs"] += abs(float(last_info.get("depth_error", 0.0)))
        totals["pitch_deg_abs"] += abs(float(np.degrees(hrl.env.physics.state.tht)))

    return {
        "episodes": int(config.eval_episodes),
        "reward_mean": totals["reward"] / max(1, config.eval_episodes),
        "success_rate": totals["success"] / max(1, config.eval_episodes),
        "roll_turns_mean": totals["roll_turns"] / max(1, config.eval_episodes),
        "depth_error_abs_mean": totals["depth_error_abs"] / max(1, config.eval_episodes),
        "pitch_deg_abs_mean": totals["pitch_deg_abs"] / max(1, config.eval_episodes),
    }


def _save_training_curves(hrl: BarrelRollTrainer, episode_history: List[Dict[str, object]], evaluation: Dict[str, float] | None) -> None:
    hrl._save_training_curves(episode_history, evaluation)


def _checkpoint_score(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("roll_turns_mean", 0.0)),
        float(metrics.get("reward_mean", 0.0)),
    )


def parse_args() -> Tuple[BarrelRollConfig, bool]:
    parser = argparse.ArgumentParser(description="USV barrel roll RL training")
    parser.add_argument("--phases", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--cycles", type=int, default=1,
                        help="Quantas vezes repetir o bloco completo de fases.")
    parser.add_argument("--phase-steps", type=int, default=8192)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episode-steps", type=int, default=320)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-depth", type=float, default=5.0)
    parser.add_argument("--target-speed", type=float, default=0.8)
    parser.add_argument("--target-turns", type=float, default=1.0)
    parser.add_argument("--max-thruster-force", type=float, default=60.0)
    parser.add_argument(
        "--action-layout",
        type=str,
        default="per_thruster",
        choices=["paired", "per_thruster"],
        help="Control parameterization for Barrel Roll training",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/barrel_roll")
    parser.add_argument("--output-dir", type=str, default="./training_runs")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--demo-warmstart-rollouts", type=int, default=0, help="Random piecewise-constant rollouts to mine for imitation warmstart")
    parser.add_argument("--demo-warmstart-segments", type=int, default=4, help="Number of constant-action segments per demo rollout")
    parser.add_argument("--demo-warmstart-top-k", type=int, default=4, help="Number of top demo rollouts used for behavior cloning")
    parser.add_argument("--demo-warmstart-epochs", type=int, default=24, help="Behavior cloning epochs for actor warmstart")
    parser.add_argument("--demo-warmstart-batch-size", type=int, default=64, help="Behavior cloning batch size")
    parser.add_argument("--demo-warmstart-actor-lr", type=float, default=1e-3, help="Behavior cloning learning rate for actor warmstart")
    parser.add_argument("--demo-warmstart-allow-failures", action="store_true", help="Allow top non-success demos when no success rollouts are found")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate for PPO updater")
    parser.add_argument("--value-lr", type=float, default=None, help="Separate learning rate for value updates (optional)")
    parser.add_argument("--value-clip", type=float, default=50.0, help="Clip critic values before value loss (<=0 disables)")
    parser.add_argument("--roll-progress-weight", type=float, default=None, help="Override roll_progress_weight in config")
    parser.add_argument("--roll-track-weight", type=float, default=None, help="Override roll_track_weight in config")
    parser.add_argument("--roll-velocity-weight", type=float, default=None, help="Override roll_velocity_weight in config")
    parser.add_argument("--reward-clip", type=float, default=0.0, help="Clip per-step reward to +-value (0 = disabled)")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Scale per-step reward by this factor")
    parser.add_argument("--reverse-roll-penalty-weight", type=float, default=None, help="Penalty weight for reverse roll progress")
    parser.add_argument("--terminal-progress-weight", type=float, default=None, help="Terminal penalty weight for incomplete rolls")
    parser.add_argument("--completion-bonus", type=float, default=None, help="Override completion bonus reward")
    parser.add_argument("--episodic-penalty-mult", type=float, default=None, help="Multiplier for episodic insufficient-roll penalty")
    parser.add_argument("--required-min-roll", type=float, default=None, help="Fraction of target roll required to avoid episodic penalty (0-1)")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    config = BarrelRollConfig(
        phases=tuple(args.phases),
        cycles=max(1, int(args.cycles)),
        phase_steps=args.phase_steps,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        dt=args.dt,
        seed=args.seed,
        target_depth=args.target_depth,
        target_speed=args.target_speed,
        target_turns=args.target_turns,
        max_thruster_force=args.max_thruster_force,
        action_layout=args.action_layout,
        lr=args.lr,
        value_lr=args.value_lr,
        value_clip=args.value_clip,
        reward_clip=args.reward_clip,
        reward_scale=args.reward_scale,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        save_every=args.save_every,
        eval_episodes=args.eval_episodes,
        fresh=args.fresh,
        demo_warmstart_rollouts=args.demo_warmstart_rollouts,
        demo_warmstart_segments=args.demo_warmstart_segments,
        demo_warmstart_top_k=args.demo_warmstart_top_k,
        demo_warmstart_epochs=args.demo_warmstart_epochs,
        demo_warmstart_batch_size=args.demo_warmstart_batch_size,
        demo_warmstart_actor_lr=args.demo_warmstart_actor_lr,
        demo_warmstart_success_only=not bool(args.demo_warmstart_allow_failures),
    )
    # apply optional CLI overrides for reward weights if provided
    if getattr(args, 'roll_progress_weight', None) is not None:
        config.roll_progress_weight = float(args.roll_progress_weight)
    if getattr(args, 'roll_track_weight', None) is not None:
        config.roll_track_weight = float(args.roll_track_weight)
    if getattr(args, 'roll_velocity_weight', None) is not None:
        config.roll_velocity_weight = float(args.roll_velocity_weight)
    # apply scale/clip
    config.reward_clip = float(args.reward_clip)
    config.reward_scale = float(args.reward_scale)
    if getattr(args, "reverse_roll_penalty_weight", None) is not None:
        config.reverse_roll_penalty_weight = float(args.reverse_roll_penalty_weight)
    if getattr(args, "terminal_progress_weight", None) is not None:
        config.terminal_progress_weight = float(args.terminal_progress_weight)
    if getattr(args, 'completion_bonus', None) is not None:
        config.completion_bonus = float(args.completion_bonus)
    if getattr(args, 'episodic_penalty_mult', None) is not None:
        config.episodic_penalty_mult = float(args.episodic_penalty_mult)
    if getattr(args, 'required_min_roll', None) is not None:
        config.required_min_roll = float(args.required_min_roll)
    return config, args.no_eval


def main() -> int:
    config, no_eval = parse_args()

    checkpoint_dir = Path(config.checkpoint_dir)
    if config.fresh and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    geometry, physics, env, network, updater, trainer = build_stack(config)
    if not config.fresh:
        trainer._load_if_requested()
    elif int(getattr(config, "demo_warmstart_rollouts", 0)) > 0:
        warmstart = trainer.demo_warmstart()
        search = warmstart.get("search", {})
        evaluation = warmstart.get("evaluation", {})
        print(
            "Warmstart | "
            f"success_demos={search.get('selected_success_count', 0)}/{search.get('selected_count', 0)} | "
            f"best_demo_roll={search.get('best_roll_turns', 0.0):.3f} | "
            f"eval_success={evaluation.get('success_rate', 0.0):.2f} | "
            f"eval_roll={evaluation.get('roll_turns_mean', 0.0):.3f}"
        )

    report = {
        "config": asdict(config),
        "phases": [],
    }
    episode_history: List[Dict[str, object]] = []
    best_phase_score = (-float("inf"), -float("inf"), -float("inf"))

    for cycle_idx in range(config.cycles):
        print(f"\n=== Training cycle {cycle_idx + 1}/{config.cycles} ===")
        for phase in config.phases:
            if phase not in (1, 2, 3):
                raise ValueError(f"Unsupported phase {phase}. Use 1, 2, or 3.")
            summary, phase_history = run_phase(config, trainer, phase)
            summary["cycle"] = cycle_idx + 1
            report["phases"].append(summary)
            episode_history.extend(phase_history)
            print(
                f"Cycle {cycle_idx + 1} | Phase {phase} complete: "
                f"{summary['steps']} steps | reward_mean={summary['reward_mean']:.3f} | "
                f"roll_turns_mean={summary['roll_turns_mean']:.3f} | "
                f"success_rate={summary['success_rate']:.3f}"
            )

            summary_score = _checkpoint_score(summary)
            if summary_score > best_phase_score:
                best_phase_score = summary_score
                trainer._save(trainer.best_prefix)

    evaluation = None
    if not no_eval:
        evaluation = evaluate(trainer, config)
        report["evaluation"] = evaluation
        print(
            f"Evaluation: success_rate={evaluation['success_rate']:.2f}, "
            f"reward_mean={evaluation['reward_mean']:.3f}, "
            f"roll_turns_mean={evaluation['roll_turns_mean']:.3f}"
        )
        evaluation_score = _checkpoint_score(evaluation)
        if evaluation_score > best_phase_score:
            best_phase_score = evaluation_score
            trainer._save(trainer.best_prefix)

    report_path = Path(config.output_dir) / "barrel_roll_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_to_jsonable(report), indent=2), encoding="utf-8")

    _save_training_curves(trainer, episode_history, evaluation)
    trainer._save(trainer.latest_prefix)

    print(f"Report saved to {report_path}")
    print(f"Curve log saved to {trainer.curve_report_path}")
    if plt is not None:
        print(f"Curve plot saved to {trainer.curve_plot_path}")
    print(f"Checkpoints saved to {Path(config.checkpoint_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
