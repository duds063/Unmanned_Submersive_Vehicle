"""Wave-robustness benchmark for the 6-DOF physics model.

The benchmark drives the vehicle on a straight reference while injecting
periodic lateral wave forcing, then plots yaw and roll over time.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine, VehicleState
from vehicle_profiles import load_vtec_s4_profile
from sensor_engine import Environment, ExtendedKalmanFilter, SensorEngine


@dataclass(frozen=True)
class WaveBenchmarkResult:
    time: np.ndarray
    yaw_deg: np.ndarray
    roll_deg: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    commands: np.ndarray


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _build_simulation(pool_depth: float, pool_radius: float, target_depth: float) -> tuple[PhysicsEngine, SensorEngine, ExtendedKalmanFilter]:
    profile = load_vtec_s4_profile(None)
    geo = GeometryEngine(L=profile.length_m, D=profile.beam_m)
    physics = PhysicsEngine(
        geo,
        max_thruster_force=10.0,
        rigid_body_mass=profile.mass_kg,
        rigid_body_inertia=profile.inertia_kgm2,
        thruster_port_position=profile.thruster_port_position_m,
        thruster_starboard_position=profile.thruster_starboard_position_m,
        planar_dof=True,
    )
    initial_state = VehicleState(z=target_depth)
    physics.reset(initial_state)

    env = Environment(pool_depth=pool_depth, pool_radius=pool_radius)
    sensors = SensorEngine(env, noise_scale=0.0, enable_rayleigh=False, seed=42)
    ekf = ExtendedKalmanFilter(physics, pool_radius=pool_radius, pool_depth=pool_depth)

    ekf.reset(np.concatenate([initial_state.eta, initial_state.nu]))
    return physics, sensors, ekf


def _simple_heading_hold_command(ekf_state, target_depth: float) -> dict[str, float]:
    yaw = float(ekf_state.orientation[2])
    roll = float(ekf_state.orientation[0])
    pitch = float(ekf_state.orientation[1])
    yaw_rate = float(ekf_state.velocity_angular[2])
    roll_rate = float(ekf_state.velocity_angular[0])
    depth = float(ekf_state.position[2])
    heave = float(ekf_state.velocity_linear[2])
    surge = float(ekf_state.velocity_linear[0])

    yaw_error = _wrap_angle(0.0 - yaw)
    depth_error = float(target_depth - depth)

    base_power = float(np.clip(0.22 + 0.02 * np.tanh((12.0 - surge) / 6.0), 0.12, 0.28))
    yaw_trim = float(np.clip(0.42 * yaw_error - 0.18 * yaw_rate - 0.05 * roll, -0.16, 0.16))
    heave_trim = float(np.clip(0.12 * depth_error - 0.05 * heave - 0.03 * pitch, -0.14, 0.14))
    ballast_trim = float(np.clip(0.10 * depth_error - 0.04 * heave, -0.25, 0.25))

    return {
        "thruster_power": base_power,
        "thruster_theta": heave_trim,
        "thruster_phi": yaw_trim,
        "ballast_cmd": ballast_trim,
    }


def run_wave_robustness(
    duration: float = 60.0,
    dt: float = 0.01,
    wave_angle_deg: float = 90.0,
    wave_freq: float = 0.6,
    wave_turbulence: float = 0.18,
    wave_force_gain: float = 1.0,
    target_depth: float = 3.0,
) -> WaveBenchmarkResult:
    physics, sensors, ekf = _build_simulation(pool_depth=10.0, pool_radius=60.0, target_depth=target_depth)

    angle_rad = np.radians(float(wave_angle_deg))
    wave_dir_world = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
    physics.env_wave_freq = float(wave_freq)
    physics.env_force_gain = float(wave_force_gain)
    physics.env_turbulence_gain = 1.0

    times: list[float] = []
    yaws: list[float] = []
    rolls: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    commands: list[tuple[float, float, float, float]] = []

    steps = int(np.ceil(float(duration) / float(dt)))
    for _ in range(steps):
        bundle = sensors.read(physics.state, physics.time)

        ekf.predict(dt)
        ekf.update_imu(bundle.imu)
        ekf.update_barometer(bundle.barometer)
        ekf.update_sonar(bundle.sonar)
        ekf.update_sonar_position(bundle.sonar)

        cmd = _simple_heading_hold_command(ekf.state_estimate, target_depth=target_depth)

        physics.step(dt=dt, env_turbulence=float(wave_turbulence), env_current_world=wave_dir_world, **cmd)

        times.append(float(physics.time))
        yaws.append(float(np.degrees(physics.state.psi)))
        rolls.append(float(np.degrees(physics.state.phi)))
        xs.append(float(physics.state.x))
        ys.append(float(physics.state.y))
        zs.append(float(physics.state.z))
        commands.append((float(cmd["thruster_power"]), float(cmd["thruster_theta"]), float(cmd["thruster_phi"]), float(cmd["ballast_cmd"])))

    return WaveBenchmarkResult(
        time=np.asarray(times, dtype=float),
        yaw_deg=np.asarray(yaws, dtype=float),
        roll_deg=np.asarray(rolls, dtype=float),
        x=np.asarray(xs, dtype=float),
        y=np.asarray(ys, dtype=float),
        z=np.asarray(zs, dtype=float),
        commands=np.asarray(commands, dtype=float),
    )


def summarize(result: WaveBenchmarkResult, wave_angle_deg: float, wave_freq: float, wave_turbulence: float) -> dict[str, float | str]:
    return {
        "wave_angle_deg": float(wave_angle_deg),
        "wave_frequency_hz": float(wave_freq),
        "wave_turbulence": float(wave_turbulence),
        "max_abs_yaw_deg": float(np.max(np.abs(result.yaw_deg))),
        "max_abs_roll_deg": float(np.max(np.abs(result.roll_deg))),
        "yaw_rms_deg": float(np.sqrt(np.mean(result.yaw_deg ** 2))),
        "roll_rms_deg": float(np.sqrt(np.mean(result.roll_deg ** 2))),
        "final_x_m": float(result.x[-1]),
        "final_y_m": float(result.y[-1]),
        "final_z_m": float(result.z[-1]),
    }


def plot_wave_response(result: WaveBenchmarkResult, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(result.time, result.yaw_deg, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Yaw [deg]")
    axes[0].set_title("6-DOF wave robustness under lateral periodic forcing")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(result.time, result.roll_deg, color="#d62728", linewidth=1.8)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Roll [deg]")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the wave-robustness benchmark.")
    parser.add_argument("--duration", type=float, default=60.0, help="Benchmark duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep.")
    parser.add_argument("--wave-angle-deg", type=float, default=90.0, help="Wave direction relative to the bow in degrees.")
    parser.add_argument("--wave-frequency-hz", type=float, default=0.6, help="Wave oscillation frequency.")
    parser.add_argument("--wave-turbulence", type=float, default=0.4, help="Wave disturbance magnitude.")
    parser.add_argument("--wave-force-gain", type=float, default=3.0, help="Multiplier for the environmental force model.")
    parser.add_argument("--outdir", type=str, default="runs/wave_robustness", help="Output directory for plot and summary.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    result = run_wave_robustness(
        duration=float(args.duration),
        dt=float(args.dt),
        wave_angle_deg=float(args.wave_angle_deg),
        wave_freq=float(args.wave_frequency_hz),
        wave_turbulence=float(args.wave_turbulence),
        wave_force_gain=float(args.wave_force_gain),
    )

    plot_path = outdir / "wave_robustness_yaw_roll.png"
    plot_wave_response(result, plot_path)

    summary = summarize(result, float(args.wave_angle_deg), float(args.wave_frequency_hz), float(args.wave_turbulence))
    summary.update(
        {
            "duration_s": float(args.duration),
            "dt_s": float(args.dt),
            "wave_force_gain": float(args.wave_force_gain),
            "plot_path": str(plot_path),
        }
    )
    summary_path = outdir / "wave_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())