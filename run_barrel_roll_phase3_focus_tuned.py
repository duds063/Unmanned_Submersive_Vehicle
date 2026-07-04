#!/usr/bin/env python3
"""Phase3-focused runner with tunable critic hyperparams and phase budgets.

Usage example:
  python run_barrel_roll_phase3_focus_tuned.py \
    --value-lr 5e-06 --value-clip 5.0 --phase3-steps 32768 --phase3-episodes 720
"""
import argparse
import subprocess
import sys
from pathlib import Path


def call_train(phases, phase_steps, episodes, extra_args, fresh=False):
    cmd = [sys.executable, "train_barrel_roll_rl.py"]
    cmd += ["--phases"] + [str(p) for p in phases]
    cmd += ["--phase-steps", str(phase_steps)]
    cmd += ["--episodes", str(episodes)]
    if fresh:
        cmd.append("--fresh")
    if extra_args:
        cmd += extra_args
    print("\nRunning:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"train_barrel_roll_rl.py failed with exit code {rc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-lr", type=float, default=None)
    parser.add_argument("--value-clip", type=float, default=None)
    parser.add_argument("--phase1-steps", type=int, default=4096)
    parser.add_argument("--phase2-steps", type=int, default=4096)
    parser.add_argument("--phase3-steps", type=int, default=16384)
    parser.add_argument("--phase1-episodes", type=int, default=60)
    parser.add_argument("--phase2-episodes", type=int, default=120)
    parser.add_argument("--phase3-episodes", type=int, default=360)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args forwarded to train_barrel_roll_rl.py")
    args = parser.parse_args()

    extra_forward = []
    if args.value_lr is not None:
        extra_forward += ["--value-lr", str(args.value_lr)]
    if args.value_clip is not None:
        extra_forward += ["--value-clip", str(args.value_clip)]
    if args.extra:
        # user can pass additional explicit args after --extra
        extra_forward += args.extra

    Path("checkpoints").mkdir(exist_ok=True)
    Path("training_runs").mkdir(exist_ok=True)

    # Phase 1
    call_train([1], args.phase1_steps, args.phase1_episodes, extra_forward, fresh=args.fresh)
    # Phase 2
    call_train([2], args.phase2_steps, args.phase2_episodes, extra_forward, fresh=False)
    # Phase 3 (emphasized)
    call_train([3], args.phase3_steps, args.phase3_episodes, extra_forward, fresh=False)


if __name__ == "__main__":
    main()
