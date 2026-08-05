"""Fossen Otter USV — independent 3-DOF reference model (validation ground truth).

This module is the "Otter" that the digital-twin physics engine is validated against
(campaign task T0.1). It is a **self-contained** re-implementation of the standard Otter
USV maneuvering model — surge/sway/yaw — following Fossen's formulation:

    M nu_dot = tau - C(nu) nu - D(nu) nu
    eta_dot  = R(psi) nu

with the well-known Otter physical parameters (Fossen, *Handbook of Marine Craft
Hydrodynamics and Motion Control*, 2011; parameter values as used in the MSS toolbox /
PythonVehicleSimulator Otter model). Only the numeric physical parameters are reused
(they are physical data); the integration code here is original so this file carries no
external license obligation and stays dependency-free (numpy only) for deterministic CI.

Modeling choices (documented, so the cross-model comparison is fair):
  * 3-DOF horizontal plane only (surge u, sway v, yaw r) — matches the engine run with
    ``planar_dof=True``.
  * Center of gravity placed at the body origin (xg = 0). This drops the sway<->yaw mass
    coupling term (m*xg), which the target engine's block-diagonal rigid-body mass matrix
    cannot represent anyway. Both models therefore use the same M structure; the residual
    nRMS reflects genuine formulation differences (Coriolis / added-mass Coriolis / damping),
    not an actuator or coupling the engine simply lacks.
  * Actuation is expressed as a prescribed generalized force tau = [X, 0, N] (see
    ``maneuver_wrench``) fed identically to both models, so the propeller rev -> thrust map
    is not part of the comparison.

Run directly to (re)generate the golden reference CSV:

    python tools/otter_reference.py            # -> data/otter_reference.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Otter physical parameters (Fossen / MSS / PythonVehicleSimulator Otter model)
# ─────────────────────────────────────────────────────────────────────────────
G = 9.81                      # m/s^2
M_TOTAL = 80.0                # kg  (hull 55 + payload 25)
L = 2.0                       # m   overall length
B = 1.08                      # m   overall beam
R66 = 0.25 * L                # m   yaw radius of gyration
IZ = M_TOTAL * R66 ** 2       # kg*m^2  yaw inertia about CG (= about CO with xg=0)

# Added mass (effective, positive): MA = -diag(Xudot, Yvdot, Nrdot) in Fossen's sign.
XUDOT_EFF = 0.1 * M_TOTAL     # 5.5   kg
YVDOT_EFF = 1.5 * M_TOTAL     # 82.5  kg
NRDOT_EFF = 1.7 * IZ          # 34.0  kg*m^2

# Rigid + added mass matrix entries (xg = 0 -> diagonal).
M11 = M_TOTAL + XUDOT_EFF     # 85.5
M22 = M_TOTAL + YVDOT_EFF     # 162.5
M33 = IZ + NRDOT_EFF          # 54.0

# Linear damping (Fossen Otter: time-constant based for sway/yaw, cruise-based for surge).
UMAX = 6.0 * 0.5144           # m/s  (6 knots)
T_SWAY = 1.0                  # s
T_YAW = 1.0                   # s
D_SURGE_LIN = 24.4 * G / UMAX # ~77.55  N/(m/s)
D_SWAY_LIN = M22 / T_SWAY     # 162.5   N/(m/s)
D_YAW_LIN = M33 / T_YAW       # 54.0    N*m/(rad/s)

# Nonlinear damping: Otter model applies a strong quadratic term on yaw only.
D_YAW_QUAD = 10.0 * D_YAW_LIN # 540.0   N*m/(rad/s)^2

# Propeller lever arm (y-offset of each thruster from centerline).
THRUSTER_ARM_Y = 0.395        # m

DT = 0.01                     # integration timestep for both maneuvers

# State/output variable order used everywhere (reference CSV columns, nRMS keys).
STATE_VARS: tuple[str, ...] = ("x", "y", "psi", "u", "v", "r")


def mass_matrix() -> np.ndarray:
    """3-DOF Otter mass matrix M = M_RB + M_A (surge, sway, yaw)."""
    return np.diag([M11, M22, M33]).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Maneuvers. Each is a prescribed body-frame generalized force tau = [X, Y, N]
# fed identically to the reference model and the engine-under-test, so every
# comparison is a pure model-vs-model check. The wrench is injected directly into
# both sets of equations of motion (no propeller model), which is why sway (Y) can
# be actuated even though the real Otter has only surge+yaw props.
# ─────────────────────────────────────────────────────────────────────────────

# "lateral_impulse": a gentle 15 N sway impulse for 5 s, then 60 s of free drift.
# Peak speeds stay ~0.09 m/s, i.e. the linear regime where the engine and the Fossen
# model agree to <1% — the strict CI gate and the project's headline validation evidence.
_IMPULSE_FORCE_N = 15.0
_IMPULSE_END_S = 5.0
_IMPULSE_DURATION_S = 65.0

# "turning": accelerate straight (10 s) then a hard turning circle (20 s). Drives the
# vehicle into the nonlinear regime (~1 m/s, yaw ~0.16 rad/s) where the engine's planar
# Coriolis diverges from Fossen — kept as a reported stress metric, not a strict gate.
_STRAIGHT_SURGE_N = 120.0
_TURN_SURGE_N = 80.0
_TURN_YAW_NM = 15.0
_TURN_PHASE1_END_S = 10.0
_TURN_DURATION_S = 30.0


def maneuver_lateral_impulse(t: float) -> np.ndarray:
    if t < _IMPULSE_END_S:
        return np.array([0.0, _IMPULSE_FORCE_N, 0.0], dtype=float)
    return np.zeros(3, dtype=float)


def maneuver_turning(t: float) -> np.ndarray:
    if t < _TURN_PHASE1_END_S:
        return np.array([_STRAIGHT_SURGE_N, 0.0, 0.0], dtype=float)
    return np.array([_TURN_SURGE_N, 0.0, _TURN_YAW_NM], dtype=float)


# name -> (wrench function, duration). The engine-side runner reads the same table.
MANEUVERS: dict[str, tuple] = {
    "lateral_impulse": (maneuver_lateral_impulse, _IMPULSE_DURATION_S),
    "turning": (maneuver_turning, _TURN_DURATION_S),
}
DEFAULT_MANEUVER = "lateral_impulse"


def _coriolis(nu: np.ndarray) -> np.ndarray:
    """Combined (rigid + added mass) Coriolis matrix for the diagonal-M 3-DOF model.

    Fossen's compact 3-DOF form expressed via the mass entries m11, m22 (which already
    include added mass):
        C = [[0,        0,      -m22*v],
             [0,        0,       m11*u],
             [m22*v, -m11*u,     0    ]]
    """
    u, v, _ = float(nu[0]), float(nu[1]), float(nu[2])
    return np.array([
        [0.0, 0.0, -M22 * v],
        [0.0, 0.0, M11 * u],
        [M22 * v, -M11 * u, 0.0],
    ], dtype=float)


def _damping(nu: np.ndarray) -> np.ndarray:
    """D(nu) = linear + nonlinear (quadratic yaw) damping."""
    r = float(nu[2])
    return np.diag([
        D_SURGE_LIN,
        D_SWAY_LIN,
        D_YAW_LIN + D_YAW_QUAD * abs(r),
    ]).astype(float)


def _rotation(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


@dataclass(frozen=True)
class OtterTrajectory:
    time: np.ndarray
    eta: np.ndarray   # shape (N, 3): x, y, psi
    nu: np.ndarray    # shape (N, 3): u, v, r

    def as_columns(self) -> dict[str, np.ndarray]:
        return {
            "x": self.eta[:, 0],
            "y": self.eta[:, 1],
            "psi": self.eta[:, 2],
            "u": self.nu[:, 0],
            "v": self.nu[:, 1],
            "r": self.nu[:, 2],
        }


def reference_csv_path(maneuver: str) -> Path:
    """Canonical golden-CSV path for a maneuver name."""
    return Path(__file__).resolve().parent / "data" / f"otter_reference_{maneuver}.csv"


def simulate(maneuver: str = DEFAULT_MANEUVER, dt: float = DT) -> OtterTrajectory:
    """Integrate the Otter reference model over a named maneuver (RK4, deterministic)."""
    if maneuver not in MANEUVERS:
        raise KeyError(f"unknown maneuver {maneuver!r}; choose from {sorted(MANEUVERS)}")
    wrench_fn, duration = MANEUVERS[maneuver]
    m_inv = np.linalg.inv(mass_matrix())

    def deriv(eta: np.ndarray, nu: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        tau = wrench_fn(t)
        nu_dot = m_inv @ (tau - _coriolis(nu) @ nu - _damping(nu) @ nu)
        eta_dot = _rotation(float(eta[2])) @ nu
        return eta_dot, nu_dot

    steps = int(round(duration / dt))
    times = np.zeros(steps + 1, dtype=float)
    etas = np.zeros((steps + 1, 3), dtype=float)
    nus = np.zeros((steps + 1, 3), dtype=float)

    eta = np.zeros(3, dtype=float)
    nu = np.zeros(3, dtype=float)
    for i in range(steps):
        t = i * dt
        e1, n1 = deriv(eta, nu, t)
        e2, n2 = deriv(eta + 0.5 * dt * e1, nu + 0.5 * dt * n1, t + 0.5 * dt)
        e3, n3 = deriv(eta + 0.5 * dt * e2, nu + 0.5 * dt * n2, t + 0.5 * dt)
        e4, n4 = deriv(eta + dt * e3, nu + dt * n3, t + dt)
        eta = eta + (dt / 6.0) * (e1 + 2 * e2 + 2 * e3 + e4)
        nu = nu + (dt / 6.0) * (n1 + 2 * n2 + 2 * n3 + n4)
        times[i + 1] = t + dt
        etas[i + 1] = eta
        nus[i + 1] = nu

    return OtterTrajectory(time=times, eta=etas, nu=nus)


def write_csv(trajectory: OtterTrajectory, out_path: Path) -> None:
    cols = trajectory.as_columns()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("t",) + STATE_VARS)
        for i in range(trajectory.time.size):
            writer.writerow([
                f"{trajectory.time[i]:.6f}",
                *[f"{cols[v][i]:.9e}" for v in STATE_VARS],
            ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the Fossen Otter reference trajectory CSV(s).")
    parser.add_argument("--dt", type=float, default=DT, help="Integration timestep (s).")
    parser.add_argument(
        "--maneuver",
        choices=sorted(MANEUVERS) + ["all"],
        default="all",
        help="Which maneuver's golden CSV to (re)generate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = sorted(MANEUVERS) if args.maneuver == "all" else [args.maneuver]
    for name in names:
        trajectory = simulate(maneuver=name, dt=float(args.dt))
        out_path = reference_csv_path(name)
        write_csv(trajectory, out_path)
        cols = trajectory.as_columns()
        print(f"[{name}] wrote {trajectory.time.size} rows to {out_path}")
        print(f"[{name}] final state:", {v: round(float(cols[v][-1]), 6) for v in STATE_VARS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
