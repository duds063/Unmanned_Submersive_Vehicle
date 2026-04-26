import json
import os
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np


class ReplayRunWriter:
    def __init__(
        self,
        replay_dir: str,
        run_id: str,
        metadata: Dict[str, Any],
    ):
        os.makedirs(replay_dir, exist_ok=True)
        self.replay_dir = replay_dir
        self.run_id = run_id
        self.frames_path = os.path.join(replay_dir, f"{run_id}.jsonl")
        self.meta_path = os.path.join(replay_dir, f"{run_id}.meta.json")
        self._fh = open(self.frames_path, "w", encoding="utf-8")
        self._frame_count = 0
        self._started_at = time.time()
        self._meta = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_epoch_s": self._started_at,
            "metadata": self._to_jsonable(metadata),
        }

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def write_frame(self, frame: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(self._to_jsonable(frame), separators=(",", ":"), ensure_ascii=False))
        self._fh.write("\n")
        self._frame_count += 1

    def close(self, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()

        finished_at = time.time()
        self._meta["finished_at_epoch_s"] = finished_at
        self._meta["elapsed_wall_s"] = float(finished_at - self._started_at)
        self._meta["frame_count"] = int(self._frame_count)
        self._meta["summary"] = self._to_jsonable(summary or {})

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

        return {
            "run_id": self.run_id,
            "frames_path": self.frames_path,
            "meta_path": self.meta_path,
            "frame_count": self._frame_count,
            "summary": self._meta["summary"],
            "metadata": self._meta["metadata"],
        }

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if is_dataclass(value):
            return cls._to_jsonable(asdict(value))
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(v) for v in value]
        return value


class ReplayExporter:
    def __init__(self, replay_dir: str = "./training_runs/replays"):
        self.replay_dir = replay_dir
        os.makedirs(self.replay_dir, exist_ok=True)

    def start_run(self, *, benchmark_mode: str, controller: str, trial: int, seed: int, scenario: Any) -> ReplayRunWriter:
        timestamp = int(time.time() * 1000)
        run_id = f"{benchmark_mode}_{controller}_trial{trial}_seed{seed}_{timestamp}"
        metadata = {
            "benchmark_mode": benchmark_mode,
            "controller": controller,
            "trial": trial,
            "seed": seed,
            "scenario": scenario,
        }
        return ReplayRunWriter(self.replay_dir, run_id, metadata)
