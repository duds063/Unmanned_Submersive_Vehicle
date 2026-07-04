#!/usr/bin/env python3
"""Run the three curriculum phases sequentially, emphasizing phase 3.

This script runs `train_barrel_roll_rl.py` three times in sequence:
  1) Phase 1 (short)
  2) Phase 2 (medium)
  3) Phase 3 (long, emphasized)

Defaults are conservative; you can pass additional CLI args which will be forwarded
to each `train_barrel_roll_rl.py` invocation (useful to override `--value-lr`,
`--value-clip`, `--episodes`, etc.). The first call uses `--fresh` to reset
checkpoints/output; subsequent calls continue from the saved latest.
"""
import sys
import subprocess
from pathlib import Path


def call_train(extra_args, phases, phase_steps, episodes, fresh=False):
    cmd = [sys.executable, "train_barrel_roll_rl.py"]
    cmd += ["--phases"] + [str(p) for p in phases]
    cmd += ["--phase-steps", str(phase_steps)]
    cmd += ["--episodes", str(episodes)]
    if fresh:
        cmd.append("--fresh")
    # forward extra args
    if extra_args:
        cmd += extra_args
    print("\nRunning:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"train_barrel_roll_rl.py failed with exit code {rc}")


def main():
    # defaults (tunable)
    # small -> medium -> large: emphasis on phase 3
    phase1_steps = 4096
    phase2_steps = 4096
    phase3_steps = 16384

    phase1_episodes = 60
    phase2_episodes = 120
    phase3_episodes = 360

    # forward any extra args given on the command line
    extra = sys.argv[1:]

    # ensure output/checkpoints directory exists (train will create it anyway)
    Path("checkpoints").mkdir(exist_ok=True)
    Path("training_runs").mkdir(exist_ok=True)

    # Phase 1: short, fresh
    call_train(extra, phases=[1], phase_steps=phase1_steps, episodes=phase1_episodes, fresh=True)

    # Phase 2: medium, continue
    call_train(extra, phases=[2], phase_steps=phase2_steps, episodes=phase2_episodes, fresh=False)

    # Phase 3: long, emphasized
    call_train(extra, phases=[3], phase_steps=phase3_steps, episodes=phase3_episodes, fresh=False)


if __name__ == "__main__":
    main()
