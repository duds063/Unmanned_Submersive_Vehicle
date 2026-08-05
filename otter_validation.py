"""Otter validation: run the digital-twin physics engine parameterized as a Fossen Otter
and compare it, variable by variable, against the independent reference model in
``tools/otter_reference.py`` (campaign task T0.1).

The engine is driven with the *same* prescribed generalized force ``tau = [X, 0, N]`` as the
reference (see :func:`tools.otter_reference.maneuver_wrench`), translated into equal/differential
port & starboard surge thrust. The metric is the per-variable **normalized RMS**:

    nRMS_i = RMS(sim_i - ref_i) / range(ref_i)

CLI:

    python otter_validation.py            # prints the nRMS summary and writes a JSON report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent      # otter/
ROOT = HERE.parent                          # repo root (physics_engine, vehicle_profiles)
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import otter_reference as otr  # noqa: E402
from physics_engine import PhysicsEngine, VehicleState  # noqa: E402
from vehicle_profiles import load_otter_profile  # noqa: E402


def run_engine_as_otter(maneuver: str = otr.DEFAULT_MANEUVER, dt: float = otr.DT) -> dict[str, np.ndarray]:
    """Drive the physics engine (configured as an Otter, planar DOF) over a named maneuver.

    The prescribed wrench ``tau = [X, Y, N]`` is injected straight into the engine's equations
    of motion via ``external_wrench_body`` (bypassing the thruster model), so this validates the
    engine's rigid-body/hydrodynamic dynamics against Fossen — not the actuator layer — and can
    exercise sway just like the reference model.
    """
    wrench_fn, duration = otr.MANEUVERS[maneuver]
    profile = load_otter_profile()
    engine = PhysicsEngine.from_vehicle_profile(profile, max_thruster_force=1.0, planar_dof=True)
    engine.reset(VehicleState())

    steps = int(round(duration / dt))
    rec = {v: np.zeros(steps + 1, dtype=float) for v in otr.STATE_VARS}
    time = np.zeros(steps + 1, dtype=float)

    def _record(i: int) -> None:
        s = engine.state
        rec["x"][i] = s.x
        rec["y"][i] = s.y
        rec["psi"][i] = s.psi
        rec["u"][i] = s.u
        rec["v"][i] = s.v
        rec["r"][i] = s.r
        time[i] = engine.time

    _record(0)
    for i in range(steps):
        x_surge, y_sway, n_yaw = wrench_fn(i * dt)
        wrench6 = np.array([x_surge, y_sway, 0.0, 0.0, 0.0, n_yaw], dtype=float)
        engine.step(
            thruster_power=0.0,
            thruster_theta=0.0,
            thruster_phi=0.0,
            ballast_cmd=0.0,
            dt=dt,
            external_wrench_body=wrench6,
        )
        _record(i + 1)

    rec["t"] = time
    return rec


def load_reference_csv(maneuver: str = otr.DEFAULT_MANEUVER) -> dict[str, np.ndarray]:
    """Load the golden Otter reference trajectory produced by ``tools/otter_reference.py``."""
    path = otr.reference_csv_path(maneuver)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference CSV not found at {path}. Generate it with: python tools/otter_reference.py"
        )
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def compute_normalized_rms(
    sim: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    variables: Sequence[str] = otr.STATE_VARS,
) -> dict[str, float]:
    """Per-variable nRMS between sim and reference, sim interpolated onto the ref time grid."""
    t_ref = ref["t"]
    t_sim = sim["t"]
    out: dict[str, float] = {}
    for var in variables:
        ref_v = ref[var]
        sim_v = np.interp(t_ref, t_sim, sim[var])
        rmse = float(np.sqrt(np.mean((sim_v - ref_v) ** 2)))
        span = float(np.max(ref_v) - np.min(ref_v))
        peak = float(np.max(np.abs(ref_v)))
        denom = span if span > 1e-9 else (peak if peak > 1e-9 else 1.0)
        out[var] = rmse / denom
    return out


def validate(maneuver: str = otr.DEFAULT_MANEUVER) -> dict[str, float]:
    """Run one maneuver end-to-end and return the per-variable nRMS."""
    sim = run_engine_as_otter(maneuver)
    ref = load_reference_csv(maneuver)
    return compute_normalized_rms(sim, ref)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the engine-as-Otter validation and report nRMS.")
    parser.add_argument(
        "--maneuver", choices=sorted(otr.MANEUVERS) + ["all"], default="all", help="Maneuver to validate."
    )
    parser.add_argument(
        "--outdir", type=str, default=str(ROOT / "runs" / "otter_validation"), help="Report output directory."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = sorted(otr.MANEUVERS) if args.maneuver == "all" else [args.maneuver]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {}
    for name in names:
        nrms = validate(name)
        report[name] = {
            "reference_csv": str(otr.reference_csv_path(name)),
            "normalized_rms": nrms,
            "normalized_rms_pct": {k: 100.0 * v for k, v in nrms.items()},
            "max_normalized_rms": max(nrms.values()),
            "max_normalized_rms_pct": 100.0 * max(nrms.values()),
        }
    (outdir / "otter_validation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
