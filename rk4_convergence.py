"""RK4 convergence benchmark for the USV physics engine.

Runs the same straight-line simulation with multiple timesteps, compares the
resulting XY trajectories, and optionally saves a plot and JSON summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine, VehicleState


@dataclass(frozen=True)
class TrajectoryResult:
    dt: float
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    speed: np.ndarray


def build_physics() -> PhysicsEngine:
    geo = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geo, max_thruster_force=10.0, planar_dof=True)
    physics.reset(VehicleState(z=3.0))
    return physics


def run_straight_line_simulation(
    dt: float,
    target_distance: float = 50.0,
    max_time: float = 180.0,
    thruster_power: float = 0.35,
) -> TrajectoryResult:
    physics = build_physics()

    times: list[float] = [0.0]
    xs: list[float] = [float(physics.state.x)]
    ys: list[float] = [float(physics.state.y)]
    zs: list[float] = [float(physics.state.z)]
    speeds: list[float] = [float(np.linalg.norm([physics.state.u, physics.state.v, physics.state.w]))]

    while physics.time < max_time and physics.state.x < target_distance:
        physics.step(
            thruster_power=thruster_power,
            thruster_theta=0.0,
            thruster_phi=0.0,
            ballast_cmd=0.0,
            dt=dt,
        )
        times.append(float(physics.time))
        xs.append(float(physics.state.x))
        ys.append(float(physics.state.y))
        zs.append(float(physics.state.z))
        speeds.append(float(np.linalg.norm([physics.state.u, physics.state.v, physics.state.w])))

    return TrajectoryResult(
        dt=float(dt),
        time=np.asarray(times, dtype=float),
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        z=np.asarray(zs, dtype=float),
        speed=np.asarray(speeds, dtype=float),
    )


def _common_x_grid(trajectories: Sequence[TrajectoryResult], samples: int = 1000) -> np.ndarray:
    max_common_x = min(float(traj.x[-1]) for traj in trajectories)
    return np.linspace(0.0, max_common_x, num=samples, dtype=float)


def _interp_y(traj: TrajectoryResult, x_grid: np.ndarray) -> np.ndarray:
    return np.interp(x_grid, traj.x, traj.y)


def compare_trajectories(trajectories: Sequence[TrajectoryResult]) -> dict[str, object]:
    ordered = sorted(trajectories, key=lambda item: item.dt)
    x_grid = _common_x_grid(ordered)
    ref = ordered[-1]
    ref_y = _interp_y(ref, x_grid)

    comparisons: dict[str, dict[str, float]] = {}
    for traj in ordered:
        y_interp = _interp_y(traj, x_grid)
        delta = y_interp - ref_y
        comparisons[f"dt={traj.dt:g}"] = {
            "rmse_y_vs_ref": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs_y_vs_ref": float(np.max(np.abs(delta))),
            "final_x": float(traj.x[-1]),
            "final_y": float(traj.y[-1]),
            "final_speed": float(traj.speed[-1]),
            "duration": float(traj.time[-1]),
        }

    return {
        "x_grid": x_grid,
        "reference_dt": float(ref.dt),
        "comparisons": comparisons,
    }


def plot_trajectories(trajectories: Sequence[TrajectoryResult], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        0.1: "#b85c38",
        0.01: "#1f77b4",
        0.001: "#2ca02c",
    }

    for traj in sorted(trajectories, key=lambda item: item.dt, reverse=True):
        label = f"dt = {traj.dt:g} s"
        ax.plot(traj.x, traj.y, linewidth=2.0, label=label, color=colors.get(traj.dt))

    ax.set_title("RK4 convergence: straight-line trajectory (50 m)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_benchmark(
    dt_values: Iterable[float] = (0.1, 0.01, 0.001),
    target_distance: float = 50.0,
    max_time: float = 180.0,
    thruster_power: float = 0.35,
) -> tuple[list[TrajectoryResult], dict[str, object]]:
    trajectories = [
        run_straight_line_simulation(
            dt=float(dt),
            target_distance=float(target_distance),
            max_time=float(max_time),
            thruster_power=float(thruster_power),
        )
        for dt in dt_values
    ]
    summary = compare_trajectories(trajectories)
    return trajectories, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RK4 timestep convergence benchmark.")
    parser.add_argument("--target-distance", type=float, default=50.0, help="Straight-line target distance in meters.")
    parser.add_argument("--max-time", type=float, default=180.0, help="Maximum simulated time in seconds.")
    parser.add_argument("--thruster-power", type=float, default=0.35, help="Constant forward thruster power.")
    parser.add_argument("--outdir", type=str, default="runs/rk4_convergence", help="Directory for the plot and JSON summary.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trajectories, summary = run_benchmark(
        dt_values=(0.1, 0.01, 0.001),
        target_distance=float(args.target_distance),
        max_time=float(args.max_time),
        thruster_power=float(args.thruster_power),
    )

    plot_path = outdir / "rk4_convergence_xy.png"
    plot_trajectories(trajectories, plot_path)

    serializable_summary = {
        "target_distance": float(args.target_distance),
        "max_time": float(args.max_time),
        "thruster_power": float(args.thruster_power),
        "plot_path": str(plot_path),
        "reference_dt": summary["reference_dt"],
        "comparisons": summary["comparisons"],
    }
    summary_path = outdir / "rk4_convergence_summary.json"
    summary_path.write_text(json.dumps(serializable_summary, indent=2), encoding="utf-8")

    print(json.dumps(serializable_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())