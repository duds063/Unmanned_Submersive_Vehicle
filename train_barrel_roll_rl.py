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

from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine, VehicleState
from rl_controller import ActorCritic, PPOUpdater, RolloutBuffer


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
    phase_steps: int = 4096
    episodes: int = 300
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
    curriculum_stages: Tuple[float, float, float] = (0.25, 0.5, 1.0)
    curriculum_breakpoints: Tuple[float, float] = (0.30, 0.70)

    max_thruster_force: float = 12.0
    max_tilt_deg: float = 60.0
    hover_depth_safety: float = 0.75
    pitch_safety_deg: float = 100.0
    env_force_gain: float = 1.0
    env_turbulence_gain: float = 1.0
    env_wave_freq: float = 0.8
    enable_domain_randomization: bool = False

    roll_progress_weight: float = 4.0
    roll_track_weight: float = 2.5
    depth_weight: float = 1.5
    speed_weight: float = 0.8
    attitude_penalty: float = 0.15
    stability_penalty: float = 0.05
    action_penalty: float = 0.01
    completion_bonus: float = 20.0

    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4

    save_every: int = 25
    eval_episodes: int = 8
    checkpoint_dir: str = "./checkpoints/barrel_roll"
    output_dir: str = "./training_runs"
    fresh: bool = False


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


class BarrelRollEnvironment:
    OBS_DIM = 15
    ACTION_DIM = 7

    def __init__(self, config: BarrelRollConfig, physics: Optional[PhysicsEngine] = None):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        if physics is None:
            geometry = GeometryEngine(L=0.8, D=0.1)
            self.physics = PhysicsEngine(geometry, max_thruster_force=config.max_thruster_force)
        else:
            self.physics = physics

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

        return {
            "thruster_power": float(np.clip(power_scale * action[0], -1.0, 1.0)),
            "thruster_theta": float((action[1] + 1.0) * 0.5 * tilt_limit),
            "thruster_phi": float((action[2] + 1.0) * 0.5 * 2.0 * np.pi),
            "ballast_cmd": float(action[3]),
            "thruster2_power": float(np.clip(power_scale * action[4], -1.0, 1.0)),
            "thruster2_theta": float((action[5] + 1.0) * 0.5 * tilt_limit),
            "thruster2_phi": float((action[6] + 1.0) * 0.5 * 2.0 * np.pi),
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

        roll_progress_reward = roll_delta / max(1e-6, self.current_target_roll)
        attitude_penalty = (state.tht ** 2 + 0.5 * state.psi ** 2)
        stability_penalty = 0.25 * state.q ** 2 + 0.25 * state.r ** 2 + 0.10 * state.v ** 2 + 0.10 * state.w ** 2
        action_penalty = float(np.sum(np.square(np.clip(action, -1.0, 1.0))))

        reward = (
            self.config.roll_progress_weight * roll_progress_reward
            + self.config.roll_track_weight * target_tracking
            + self.config.depth_weight * depth_reward
            + self.config.speed_weight * speed_reward
            - self.config.attitude_penalty * attitude_penalty
            - self.config.stability_penalty * stability_penalty
            - self.config.action_penalty * action_penalty
        )

        success = False
        if not self.completion_awarded and self.roll_progress >= self.current_target_roll:
            if abs(depth_error) <= self.config.hover_depth_safety and abs(state.tht) <= np.radians(30.0):
                reward += self.config.completion_bonus
                self.completion_awarded = True
                success = True

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

        return self._observation(), float(reward), done, info


class BarrelRollTrainer:
    def __init__(self, config: BarrelRollConfig):
        self.config = config
        np.random.seed(config.seed)

        self.env = BarrelRollEnvironment(config)
        self.network = ActorCritic(obs_dim=self.env.OBS_DIM, action_dim=self.env.ACTION_DIM, hidden=[64, 64])
        self.updater = PPOUpdater(
            self.network,
            lr=config.lr,
            clip_eps=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            n_epochs=10,
            batch_size=64,
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

        for step_index in range(self.config.episode_steps):
            if training:
                action, log_prob, value = self.network.act(obs, update_obs_stats=True)
            else:
                action = self._policy_mean_action(obs)
                log_prob = 0.0
                _, _, value = self.network.forward(obs, normalize=True, update_obs_stats=False)

            next_obs, reward, done, info = self.env.step(action, step_index)
            if training:
                buffer.add(obs, action, log_prob, reward, value, done)
            obs = next_obs
            episode_reward += reward
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

        best_reward = -float("inf")
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

            if summary.reward > best_reward:
                best_reward = summary.reward
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

            if evaluation["reward_mean"] > best_reward:
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

        for step in range(config.episode_steps):
            action, log_prob, value = hrl.network.act(obs, update_obs_stats=True)
            next_obs, reward, done, info = hrl.env.step(action, step)
            buffer.add(obs, action, log_prob, reward, value, done)

            obs = next_obs
            episode_reward += float(reward)
            last_info = info
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


def parse_args() -> Tuple[BarrelRollConfig, bool]:
    parser = argparse.ArgumentParser(description="USV barrel roll RL training")
    parser.add_argument("--phases", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--cycles", type=int, default=1,
                        help="Quantas vezes repetir o bloco completo de fases.")
    parser.add_argument("--phase-steps", type=int, default=4096)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--episode-steps", type=int, default=320)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-depth", type=float, default=5.0)
    parser.add_argument("--target-speed", type=float, default=0.8)
    parser.add_argument("--target-turns", type=float, default=1.0)
    parser.add_argument("--max-thruster-force", type=float, default=12.0)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/barrel_roll")
    parser.add_argument("--output-dir", type=str, default="./training_runs")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=8)
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
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        save_every=args.save_every,
        eval_episodes=args.eval_episodes,
        fresh=args.fresh,
    )
    return config, args.no_eval


def main() -> int:
    config, no_eval = parse_args()

    checkpoint_dir = Path(config.checkpoint_dir)
    if config.fresh and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    geometry, physics, env, network, updater, trainer = build_stack(config)
    if not config.fresh:
        trainer._load_if_requested()

    report = {
        "config": asdict(config),
        "phases": [],
    }
    episode_history: List[Dict[str, object]] = []
    best_phase_reward = -float("inf")

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
                f"{summary['steps']} steps | reward_mean={summary['reward_mean']:.3f}"
            )

            if summary["reward_mean"] > best_phase_reward:
                best_phase_reward = summary["reward_mean"]
                trainer._save(trainer.best_prefix)

    evaluation = None
    if not no_eval:
        evaluation = evaluate(trainer, config)
        report["evaluation"] = evaluation
        print(
            f"Evaluation: success_rate={evaluation['success_rate']:.2f}, "
            f"reward_mean={evaluation['reward_mean']:.3f}"
        )
        if evaluation["reward_mean"] > best_phase_reward:
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