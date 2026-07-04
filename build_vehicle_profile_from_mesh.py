"""Build a VehicleProfile-compatible JSON artifact from an STL/OBJ mesh.

Usage:
    python tools/build_vehicle_profile_from_mesh.py --mesh hull.stl --out training_runs/profiles/hull_profile.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mesh_hydro_pipeline import MeshHydroPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate hydrodynamic profile from 3D mesh")
    parser.add_argument("--mesh", required=True, help="Input mesh path (.stl or .obj)")
    parser.add_argument("--out", required=True, help="Output profile JSON path")
    parser.add_argument("--target-faces", type=int, default=800, help="Target face count after decimation")
    parser.add_argument("--units-scale", type=float, default=1.0, help="Scale factor to convert mesh units to meters")
    parser.add_argument("--name", default=None, help="Optional profile name")
    parser.add_argument("--rho", type=float, default=1025.0, help="Fluid density kg/m^3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = MeshHydroPipeline(fluid_density_kg_m3=args.rho)
    result = pipeline.process(
        mesh_path=args.mesh,
        target_faces=args.target_faces,
        units_scale_to_m=args.units_scale,
        name=args.name,
    )

    payload = result.to_vehicle_profile_payload()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Generated profile: {out_path}")
    print(f"Mesh: {result.source_mesh_path}")
    print(f"Faces: {result.original_faces} -> {result.decimated_faces}")
    print(f"Dimensions [m]: L={result.length_m:.3f}, B={result.beam_total_m:.3f}, T={result.draft_m:.3f}")
    print(f"Displaced volume [m^3]: {result.volume_m3:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
