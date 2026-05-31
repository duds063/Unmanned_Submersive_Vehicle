#!/usr/bin/env python3
"""Run a long, tuned barrel-roll campaign locally with periodic summaries.

This script launches the phase3-focused tuned runner and periodically prints
summary metrics read from training_runs/barrel_roll_training_report.json.

Example:
  python run_barrel_roll_local_long.py --value-lr 5e-06 --value-clip 5.0

Defaults mirror the tuned campaign we've been using.
"""
import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import statistics


def read_report(path: Path):
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def summarize_report(data):
    out_lines = []
    if data is None:
        return ["(no report yet)"]
    phases = data.get("phases", [])
    if phases:
        last = phases[-1]
        updates = last.get("updates", [])
        vals = [u.get("value_loss") for u in updates if u.get("value_loss") is not None]
        out_lines.append(f"Phase: {last.get('phase_name')} steps={last.get('steps')} episodes={last.get('episodes')}")
        out_lines.append(f"  reward_mean={last.get('reward_mean'):.3f} roll_turns_mean={last.get('roll_turns_mean'):.3f} success_rate={last.get('success_rate'):.3f}")
        if vals:
            out_lines.append(f"  value_loss_mean={statistics.mean(vals):.3f} value_loss_max={max(vals):.3f}")
    eval = data.get("evaluation")
    if eval:
        out_lines.append(f"Evaluation: reward_mean={eval.get('reward_mean'):.3f} success_rate={eval.get('success_rate'):.3f} roll_turns_mean={eval.get('roll_turns_mean'):.3f}")
    return out_lines


def build_command(args):
    cmd = [sys.executable, "run_barrel_roll_phase3_focus_tuned.py"]
    if args.value_lr is not None:
        cmd += ["--value-lr", str(args.value_lr)]
    if args.value_clip is not None:
        cmd += ["--value-clip", str(args.value_clip)]
    if args.phase1_steps is not None:
        cmd += ["--phase1-steps", str(args.phase1_steps)]
    if args.phase2_steps is not None:
        cmd += ["--phase2-steps", str(args.phase2_steps)]
    if args.phase3_steps is not None:
        cmd += ["--phase3-steps", str(args.phase3_steps)]
    if args.phase1_episodes is not None:
        cmd += ["--phase1-episodes", str(args.phase1_episodes)]
    if args.phase2_episodes is not None:
        cmd += ["--phase2-episodes", str(args.phase2_episodes)]
    if args.phase3_episodes is not None:
        cmd += ["--phase3-episodes", str(args.phase3_episodes)]
    if args.fresh:
        cmd.append("--fresh")
    # forward required-min-roll via --extra
    if args.required_min_roll is not None:
        cmd += ["--extra", "--required-min-roll", str(args.required_min_roll)]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run long tuned barrel roll with periodic summaries")
    parser.add_argument("--value-lr", type=float, default=5e-06)
    parser.add_argument("--value-clip", type=float, default=5.0)
    parser.add_argument("--phase1-steps", type=int, default=4096)
    parser.add_argument("--phase2-steps", type=int, default=4096)
    parser.add_argument("--phase3-steps", type=int, default=65536)
    parser.add_argument("--phase1-episodes", type=int, default=60)
    parser.add_argument("--phase2-episodes", type=int, default=120)
    parser.add_argument("--phase3-episodes", type=int, default=1440)
    parser.add_argument("--required-min-roll", type=float, default=0.75)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--interval-min", type=int, default=60, help="Minutes between summary prints")
    args = parser.parse_args()

    report_path = Path("training_runs") / "barrel_roll_training_report.json"
    cmd = build_command(args)
    print("Starting long run:", " ".join(cmd))

    proc = subprocess.Popen(cmd)

    try:
        interval = max(1, args.interval_min) * 60
        while True:
            # print timestamped summary
            data = read_report(report_path)
            lines = summarize_report(data)
            print("[", datetime.now().isoformat(timespec='minutes'), "] Summary:")
            for l in lines:
                print(l)
            # check if process finished
            rc = proc.poll()
            if rc is not None:
                print(f"Process exited with code {rc}")
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Interrupted by user, terminating subprocess...")
        proc.terminate()
        proc.wait()

    # final dump
    final = read_report(report_path)
    print("Final report summary:")
    for l in summarize_report(final):
        print(l)


if __name__ == "__main__":
    main()
