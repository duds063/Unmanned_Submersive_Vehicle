import argparse
import json
import os
import sys
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geometry_engine import GeometryEngine
from mission_engine import MissionEngine


def _json_dict_or_empty(value: Optional[object]) -> Dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("Manual config must be a JSON object.")


def load_manual_config(file_path: Optional[str], inline_json: Optional[str]) -> Dict[str, object]:
    config: Dict[str, object] = {}

    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            file_payload = json.load(f)
        config.update(_json_dict_or_empty(file_payload))

    if inline_json:
        inline_payload = json.loads(inline_json)
        config.update(_json_dict_or_empty(inline_payload))

    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MissionEngine in independent manual mode using a selected input source."
    )
    parser.add_argument(
        "--manual-source-mode",
        required=True,
        choices=["keyboard", "joystick", "udp", "replay", "expression"],
        help="Manual input adapter to use.",
    )
    parser.add_argument(
        "--manual-config-file",
        type=str,
        default=None,
        help="Path to JSON file with manual source configuration.",
    )
    parser.add_argument(
        "--manual-config-json",
        type=str,
        default=None,
        help="Inline JSON object merged on top of file config.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep.")
    parser.add_argument("--max-steps", type=int, default=200, help="Episode length limit.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run.")
    parser.add_argument("--seed", type=int, default=42, help="Simulation RNG seed.")
    parser.add_argument("--pool-depth", type=float, default=5.0, help="Pool depth in meters.")
    parser.add_argument("--pool-radius", type=float, default=20.0, help="Pool radius in meters.")
    parser.add_argument("--vehicle-length", type=float, default=0.8, help="Vehicle length (GeometryEngine L).")
    parser.add_argument("--vehicle-diameter", type=float, default=0.1, help="Vehicle diameter (GeometryEngine D).")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory used by MissionEngine dependencies.",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    config = load_manual_config(args.manual_config_file, args.manual_config_json)

    geo = GeometryEngine(L=float(args.vehicle_length), D=float(args.vehicle_diameter))
    mission = MissionEngine(
        geo,
        checkpoint_dir=str(args.checkpoint_dir),
        seed=int(args.seed),
        pool_depth=float(args.pool_depth),
        pool_radius=float(args.pool_radius),
        control_mode="manual",
        manual_source_mode=str(args.manual_source_mode),
        manual_source_config=config,
    )

    results = []
    for _ in range(max(1, int(args.episodes))):
        result = mission._run_episode(
            dt=float(args.dt),
            training=False,
            max_steps=int(args.max_steps),
        )
        results.append(
            {
                "termination": result.termination.value,
                "total_steps": int(result.total_steps),
                "phase": result.phase.name,
                "collision": bool(result.collision),
            }
        )

    summary = {
        "control_mode": "manual",
        "manual_source_mode": str(args.manual_source_mode),
        "episodes": int(args.episodes),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
