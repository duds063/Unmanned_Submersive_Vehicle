from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    command = [
        sys.executable,
        str(repo_root / "train_barrel_roll_rl.py"),
        "--fresh",
        "--cycles",
        "1",
        "--episodes",
        "120",
        "--phase-steps",
        "2048",
        "--episode-steps",
        "320",
        "--eval-episodes",
        "8",
    ]
    completed = subprocess.run(command, cwd=repo_root)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())