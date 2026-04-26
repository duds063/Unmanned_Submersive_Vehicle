import bisect
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ReplayTrial:
    run_id: str
    frames_path: str
    meta_path: str
    controller: str
    trial: int
    benchmark_mode: str
    frame_count: int
    duration_s: float


class VisualizationPlayer:
    def __init__(self, replay_dir: str = "./training_runs/replays"):
        self.replay_dir = replay_dir
        self._lock = threading.Lock()
        self._catalog: Dict[str, ReplayTrial] = {}
        self._frames_cache: Dict[str, List[dict]] = {}
        self._times_cache: Dict[str, List[float]] = {}

        self.primary_run_id: Optional[str] = None
        self.selected_run_ids: List[str] = []
        self.playing = False
        self.speed = 1.0
        self.playhead_s = 0.0

        self.refresh_catalog()

    def refresh_catalog(self) -> Dict[str, ReplayTrial]:
        catalog: Dict[str, ReplayTrial] = {}
        if not os.path.isdir(self.replay_dir):
            self._catalog = {}
            return self._catalog

        for name in os.listdir(self.replay_dir):
            if not name.endswith(".meta.json"):
                continue
            meta_path = os.path.join(self.replay_dir, name)
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            run_id = str(meta.get("run_id") or "")
            frame_count = int(meta.get("frame_count") or 0)
            metadata = meta.get("metadata") or {}
            summary = meta.get("summary") or {}
            if not run_id:
                continue

            frames_path = str(meta.get("frames_path") or os.path.join(self.replay_dir, f"{run_id}.jsonl"))
            if not os.path.isabs(frames_path):
                frames_path = os.path.join(self.replay_dir, os.path.basename(frames_path))
            if not os.path.exists(frames_path):
                frames_path = os.path.join(self.replay_dir, f"{run_id}.jsonl")
            if not os.path.exists(frames_path):
                continue

            duration_s = float(summary.get("sim_time_s") or 0.0)
            benchmark_mode = str(metadata.get("benchmark_mode") or summary.get("benchmark_mode") or "mission")
            controller = str(metadata.get("controller") or summary.get("controller") or "unknown")
            trial = int(metadata.get("trial") or summary.get("trial") or 0)

            catalog[run_id] = ReplayTrial(
                run_id=run_id,
                frames_path=frames_path,
                meta_path=meta_path,
                controller=controller,
                trial=trial,
                benchmark_mode=benchmark_mode,
                frame_count=frame_count,
                duration_s=duration_s,
            )

        self._catalog = dict(sorted(catalog.items(), key=lambda item: item[0]))
        return self._catalog

    def list_trials(self) -> List[dict]:
        with self._lock:
            return self._list_trials_nolock()

    def _list_trials_nolock(self) -> List[dict]:
        items = list(self._catalog.values())
        return [
            {
                "run_id": t.run_id,
                "controller": t.controller,
                "trial": t.trial,
                "benchmark_mode": t.benchmark_mode,
                "frame_count": t.frame_count,
                "duration_s": t.duration_s,
            }
            for t in items
        ]

    def load_primary(self, run_id: str) -> bool:
        with self._lock:
            if run_id not in self._catalog:
                return False
            self._ensure_loaded(run_id)
            self.primary_run_id = run_id
            if run_id not in self.selected_run_ids:
                self.selected_run_ids = [run_id]
            self.playhead_s = 0.0
            return True

    def select_trials(self, run_ids: List[str]) -> List[str]:
        with self._lock:
            valid = [rid for rid in run_ids if rid in self._catalog]
            if not valid and self.primary_run_id:
                valid = [self.primary_run_id]
            for rid in valid:
                self._ensure_loaded(rid)
            self.selected_run_ids = valid
            if self.primary_run_id not in self.selected_run_ids and self.selected_run_ids:
                self.primary_run_id = self.selected_run_ids[0]
            return self.selected_run_ids[:]

    def play(self) -> None:
        with self._lock:
            self.playing = True

    def pause(self) -> None:
        with self._lock:
            self.playing = False

    def toggle_pause(self) -> bool:
        with self._lock:
            self.playing = not self.playing
            return self.playing

    def set_speed(self, speed: float) -> float:
        with self._lock:
            self.speed = max(0.1, min(10.0, float(speed)))
            return self.speed

    def seek_ratio(self, ratio: float) -> float:
        with self._lock:
            duration = self._primary_duration()
            if duration <= 0.0:
                self.playhead_s = 0.0
            else:
                self.playhead_s = max(0.0, min(duration, float(ratio) * duration))
            return self.playhead_s

    def seek_time(self, target_s: float) -> float:
        with self._lock:
            duration = self._primary_duration()
            self.playhead_s = max(0.0, min(duration, float(target_s)))
            return self.playhead_s

    def reset(self) -> None:
        with self._lock:
            self.playhead_s = 0.0

    def tick(self, wall_dt_s: float) -> None:
        with self._lock:
            if not self.playing or not self.primary_run_id:
                return
            duration = self._primary_duration()
            if duration <= 0.0:
                return
            self.playhead_s += float(wall_dt_s) * self.speed
            if self.playhead_s >= duration:
                self.playhead_s = duration
                self.playing = False

    def current_state(self) -> Optional[dict]:
        with self._lock:
            if not self.primary_run_id:
                return None

            primary_frame = self._frame_at_time(self.primary_run_id, self.playhead_s)
            if primary_frame is None:
                return None

            trajectory = self._trajectory_window(self.primary_run_id, self.playhead_s, max_points=500)
            envelope = self._envelope_at_time(self.playhead_s)
            errors = self._ekf_error(primary_frame)
            return self._build_payload(primary_frame, trajectory, envelope, errors)

    def status(self) -> dict:
        with self._lock:
            duration = self._primary_duration()
            return {
                "playing": self.playing,
                "speed": self.speed,
                "playhead_s": self.playhead_s,
                "duration_s": duration,
                "primary_run_id": self.primary_run_id,
                "selected_run_ids": self.selected_run_ids[:],
                "trials": self._list_trials_nolock(),
            }

    def _ensure_loaded(self, run_id: str) -> None:
        if run_id in self._frames_cache:
            return
        trial = self._catalog[run_id]
        frames: List[dict] = []
        times: List[float] = []
        with open(trial.frames_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frame = json.loads(line)
                except Exception:
                    continue
                frame_time = float(frame.get("time") or 0.0)
                frames.append(frame)
                times.append(frame_time)
        self._frames_cache[run_id] = frames
        self._times_cache[run_id] = times

    def _primary_duration(self) -> float:
        if not self.primary_run_id:
            return 0.0
        self._ensure_loaded(self.primary_run_id)
        times = self._times_cache.get(self.primary_run_id, [])
        return float(times[-1]) if times else 0.0

    def _frame_at_time(self, run_id: str, target_s: float) -> Optional[dict]:
        self._ensure_loaded(run_id)
        frames = self._frames_cache.get(run_id, [])
        times = self._times_cache.get(run_id, [])
        if not frames:
            return None
        idx = bisect.bisect_left(times, float(target_s))
        if idx <= 0:
            return frames[0]
        if idx >= len(frames):
            return frames[-1]
        prev_t = times[idx - 1]
        next_t = times[idx]
        if abs(target_s - prev_t) <= abs(next_t - target_s):
            return frames[idx - 1]
        return frames[idx]

    def _trajectory_window(self, run_id: str, target_s: float, max_points: int = 500) -> List[List[float]]:
        self._ensure_loaded(run_id)
        frames = self._frames_cache.get(run_id, [])
        times = self._times_cache.get(run_id, [])
        if not frames:
            return []
        idx = bisect.bisect_right(times, float(target_s))
        window = frames[max(0, idx - max_points):idx]
        return [self._position_from_frame(frame) for frame in window]

    def _envelope_at_time(self, target_s: float) -> Optional[dict]:
        if len(self.selected_run_ids) <= 1:
            return None
        positions = []
        for run_id in self.selected_run_ids:
            frame = self._frame_at_time(run_id, target_s)
            if frame is None:
                continue
            positions.append(self._position_from_frame(frame))
        if len(positions) <= 1:
            return None
        arr = np.asarray(positions, dtype=float)
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "sample_count": int(arr.shape[0]),
        }

    @staticmethod
    def _position_from_frame(frame: dict) -> List[float]:
        pos = frame.get("state_true", {}).get("position") or [0.0, 0.0, 0.0]
        return [float(pos[0]), float(pos[1]), float(pos[2])]

    @staticmethod
    def _ekf_error(frame: dict) -> dict:
        truth = np.asarray(frame.get("state_true", {}).get("position") or [0.0, 0.0, 0.0], dtype=float)
        est = np.asarray(frame.get("ekf_estimate", {}).get("position") or [0.0, 0.0, 0.0], dtype=float)
        return {
            "ekf_position_error_m": float(np.linalg.norm(est - truth)),
            "ekf_position_error_vector": (est - truth).tolist(),
        }

    def _build_payload(self, frame: dict, trajectory: List[List[float]], envelope: Optional[dict], errors: dict) -> dict:
        truth = frame.get("state_true", {})
        sensors = frame.get("sensors", {})
        command = frame.get("command", {})
        vectors = frame.get("vectors", {})
        env = frame.get("environment", {})
        ekf = frame.get("ekf_estimate", {})

        sonar = []
        for item in sensors.get("sonar", []) or []:
            sonar.append({
                "direction": item.get("direction", [0.0, 0.0, 0.0]),
                "distance": float(item.get("distance", -1.0)),
                "hit": bool(item.get("distance", -1.0) > 0.0),
                "confidence": float(item.get("confidence", 0.0)),
            })

        velocity_linear = truth.get("velocity_linear") or [0.0, 0.0, 0.0]
        velocity_angular = truth.get("velocity_angular") or [0.0, 0.0, 0.0]

        return {
            "time": float(frame.get("time") or 0.0),
            "position": truth.get("position") or [0.0, 0.0, 0.0],
            "quaternion": truth.get("quaternion") or [1.0, 0.0, 0.0, 0.0],
            "velocity_linear": velocity_linear,
            "velocity_angular": velocity_angular,
            "euler": truth.get("euler") or [0.0, 0.0, 0.0],
            "ballast": {
                "density_avg": 1000.0,
            },
            "thruster": {
                "power": float(command.get("thruster_power") or 0.0),
                "theta_deg": float(command.get("thruster_theta") or 0.0),
                "phi_deg": float(command.get("thruster_phi") or 0.0),
                "force_vector": vectors.get("thrust_total") or [0.0, 0.0, 0.0],
            },
            "sonar": sonar,
            "ekf": {
                "position": ekf.get("position") or [0.0, 0.0, 0.0],
                "orientation": ekf.get("orientation") or [0.0, 0.0, 0.0],
            },
            "trajectory": trajectory,
            "waypoints": ((frame.get("metadata") or {}).get("scenario") or {}).get("waypoints", []),
            "reference_frame": "NED",
            "environmental_noise": {
                "rayleigh_enabled": bool(env.get("rayleigh_enabled", False)),
                "rayleigh_sigma": float(env.get("rayleigh_sigma", 0.0)),
                "env_disturbance_scale": float(env.get("env_disturbance_scale", 0.0)),
            },
            "dynamic_obstacles": [
                {
                    "position": item.get("position", [0.0, 0.0, 0.0]),
                    "radius": float(item.get("radius", 0.5)),
                }
                for item in (env.get("dynamic_obstacles") or [])
            ],
            "controller": str(frame.get("controller") or "unknown"),
            "controller_waypoint": None,
            "controller_waypoint_index": 0,
            "controller_waypoint_count": 0,
            "controller_mission_complete": bool(frame.get("metrics", {}).get("termination") == "mission_complete"),
            "command": command,
            "replay": {
                "run_id": self.primary_run_id,
                "selected_run_ids": self.selected_run_ids,
                "playing": self.playing,
                "speed": self.speed,
                "playhead_s": self.playhead_s,
                "duration_s": self._primary_duration(),
            },
            "errors": errors,
            "vectors": {
                "velocity_body": velocity_linear,
                "thrust_total": vectors.get("thrust_total") or [0.0, 0.0, 0.0],
                "thrust_port": vectors.get("thrust_port") or [0.0, 0.0, 0.0],
                "thrust_starboard": vectors.get("thrust_starboard") or [0.0, 0.0, 0.0],
            },
            "comparison_envelope": envelope,
        }


class PlayerLoop:
    def __init__(self, player: VisualizationPlayer, emit_callback, hz: float = 60.0):
        self.player = player
        self.emit_callback = emit_callback
        self.hz = hz
        self._running = False
        self._thread = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        interval = 1.0 / max(1.0, self.hz)
        last = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            dt = now - last
            last = now
            self.player.tick(dt)
            payload = self.player.current_state()
            if payload is not None:
                self.emit_callback(payload)
            sleep_time = interval - (time.perf_counter() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self._running = False
