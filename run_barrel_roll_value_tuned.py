#!/usr/bin/env python3
"""Runner: tuned critic hyperparams for barrel roll training.

Usage: python run_barrel_roll_value_tuned.py
You can override args by editing the `args` list below or passing additional
command-line arguments which will be forwarded to `train_barrel_roll_rl.py`.
"""
import sys
import subprocess

def main():
    base = [sys.executable, "train_barrel_roll_rl.py"]
    args = [
        "--fresh",
        "--cycles", "1",
        "--episodes", "360",
        "--phase-steps", "4096",
        "--episode-steps", "320",
        "--eval-episodes", "16",
        "--value-lr", "1e-5",
        "--value-clip", "10.0"
    ]
    # Forward any extra args from the command line
    extra = sys.argv[1:]
    cmd = base + args + extra
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"train_barrel_roll_rl.py exited with code {rc}")
    sys.exit(rc)

if __name__ == "__main__":
    main()
