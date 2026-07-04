#!/usr/bin/env python3
"""Diagnostic runner: short training + extract value_loss timeseries.

Usage: python run_barrel_roll_value_diagnostic.py
Creates: training_runs/value_loss_timeseries.json
"""
import sys
import subprocess
import json
import statistics
from pathlib import Path


def run_training():
    py = sys.executable
    cmd = [py, "train_barrel_roll_rl.py",
           "--fresh",
           "--cycles", "1",
           "--episodes", "60",
           "--phase-steps", "1024",
           "--episode-steps", "160",
           "--eval-episodes", "4",
           "--value-lr", "5e-06",
           "--value-clip", "5.0"
    ]
    print("Running training:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"train_barrel_roll_rl.py failed with code {rc}")


def extract_timeseries(report_path: Path, out_path: Path):
    with report_path.open() as f:
        r = json.load(f)
    phases = r.get("phases", [])
    out = {}
    for entry in phases:
        name = entry.get("phase_name")
        cycle = entry.get("cycle")
        updates = entry.get("updates", [])
        vals = [u.get("value_loss") for u in updates if u.get("value_loss") is not None]
        out_key = f"{name}_cycle_{cycle}"
        out[out_key] = {
            "phase_name": name,
            "cycle": cycle,
            "count": len(vals),
            "value_loss_mean": statistics.mean(vals) if vals else None,
            "value_loss_max": max(vals) if vals else None,
            "value_loss_series": vals
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    base = Path("training_runs")
    report = base / "barrel_roll_training_report.json"
    out = base / "value_loss_timeseries.json"
    run_training()
    print("Training finished, extracting timeseries...")
    data = extract_timeseries(report, out)
    # Print short summary
    for k, v in data.items():
        print(f"{k}: count={v['count']}, mean={v['value_loss_mean']}, max={v['value_loss_max']}")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
