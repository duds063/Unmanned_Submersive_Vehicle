# Verification & Validation

> **Read this distinction first — it is the credibility line the ARAMUSS campaign (§2) depends on.**
>
> - **Verification** = "is the math implemented correctly?" We check the engine against an
>   *independent reference model* (equations vs equations). **This is what is done today.**
> - **Validation** = "does the math match reality?" We check the engine against *recorded
>   real-world telemetry* (sea trial / tow tank). **This is still pending real data** and is
>   campaign objective #1.
>
> Describe the current status as *"verified against the Fossen Otter reference model"* —
> **never** as *"validated against the Otter"* or *"calibrated to real data."*

CI (`.github/workflows/ci.yml`) runs the guards below so the verification cannot break silently.

---

## Protocol 1 — Fossen Otter cross-verification (task T0.1) ✅ done

**What it checks.** The physics engine (`physics_engine.py`), parameterized as a
[Maritime Robotics **Otter** USV](https://github.com/cybergalactic/PythonVehicleSimulator),
is compared against an **independent** re-implementation of the standard Fossen Otter 3-DOF
maneuvering model. Both integrate the *same* prescribed generalized force (injected directly
into the equations of motion) over the *same* maneuver. Metric — per-variable normalized RMS:

```
nRMS_i = RMS(sim_i - ref_i) / range(ref_i)
```

### Components

| Piece | File |
|---|---|
| Independent Fossen Otter reference model + golden CSV generator | [`otter/otter_reference.py`](../otter/otter_reference.py) |
| Golden reference trajectories (committed) | `otter/data/otter_reference_lateral_impulse.csv`, `otter/data/otter_reference_turning.csv` |
| Otter vehicle profile for the engine | `load_otter_profile()` in [`vehicle_profiles.py`](../vehicle_profiles.py) |
| Engine-as-Otter runner + nRMS metric | [`otter/otter_validation.py`](../otter/otter_validation.py) |
| CI regression guard (strict 1% gate) | [`otter/test_otter_validation.py`](../otter/test_otter_validation.py) |

### Maneuvers (both fed identically to engine and reference)

The wrench `τ = [X, Y, N]` is injected straight into both sets of equations of motion via the
engine's `external_wrench_body` input (bypassing the propeller model), so the comparison
isolates the **equations of motion** (mass, Coriolis, damping, integration) — which is exactly
what "verification" should test — and lets sway be actuated just like the reference.

- **`lateral_impulse`** — 15 N sway impulse for 5 s, then 60 s free drift. Peak speeds ~0.09 m/s
  (linear regime). This is the project's headline evidence figure.
- **`turning`** — straight-line accel (10 s, `X=120 N`) then a turning circle (20 s, `X=80 N`,
  `N=15 N·m`). ~1 m/s, yaw ~0.16 rad/s (nonlinear regime).

`dt = 0.01 s`, RK4, fully deterministic (regenerating a CSV is bit-identical).

### Modeling choices (so the comparison is fair)

- 3-DOF horizontal plane (surge `u`, sway `v`, yaw `r`); the engine runs with `planar_dof=True`.
- Center of gravity at the body origin (`xg = 0`) in both models, removing a sway↔yaw mass
  coupling term the engine's block-diagonal rigid-body mass matrix cannot represent.
- Parameters are the standard Otter values (mass 80 kg, added mass, damping); identical M and D
  are given to both models, so any residual reflects *formulation/integration* differences only.

### Results (2026-07-27) — max nRMS **0.03%** on both maneuvers

| Variable | `lateral_impulse` | `turning` |
|---|---:|---:|
| `x` | 0.000% | 0.010% |
| `y` | 0.032% | 0.008% |
| `psi` | 0.000% | 0.007% |
| `u` | 0.000% | 0.007% |
| `v` | 0.015% | 0.020% |
| `r` | 0.000% | 0.029% |
| **max** | **0.032%** | **0.029%** |

CI gate: **1% per variable** on both maneuvers — passes with ~30× headroom, absorbs
cross-platform float noise, and still trips on any real regression.

### Engine change this protocol drove

Reaching <1% on the nonlinear `turning` maneuver required fixing one term in the engine:
`_coriolis_matrix` ([physics_engine.py](../physics_engine.py)) previously used a single rigid
mass in the translational Coriolis coupling, **omitting the added-mass Coriolis** `C_A(ν)`. It
now uses per-axis effective masses `m + X_u̇`, `m + Y_v̇`, `m + Z_ẇ` (Fossen 2011, Theorem 3.2),
which restores the added-mass Coriolis and its Munk-moment coupling. Isolation test: removing
that term again sends the `turning` residual from 0.03% back to **22%** — i.e. the term is the
entire nonlinear-regime gap, and the 1% gate catches its removal.

> ⚠️ This changed the **core dynamics** for every consumer of the engine (LQR/MPC controllers,
> RL training, other benchmarks). It is physically more correct, but any tuning done against the
> old (simplified) Coriolis may shift and should be re-checked.

### Reproduce

```bash
python otter/otter_reference.py        # regenerate both golden CSVs (deterministic)
python otter/otter_validation.py       # print nRMS for both maneuvers -> runs/otter_validation/
pytest otter/ -v
```

---

## Turning verification into real validation (pending real data)

The harness is source-agnostic: `otter/otter_validation.py:load_reference_csv()` reads *any* CSV with
columns `t, x, y, psi, u, v, r`. To upgrade from verification to physical **validation**:

1. Record a real run (sea trial or tow tank) of the Otter — or of the Vtec-S4 — performing one of
   the maneuvers above, logging `t, x, y, psi, u, v, r`.
2. Drop it in as `otter/data/otter_reference_<maneuver>.csv` (or point `load_reference_csv` at it).
3. Re-run the same nRMS gate. The number now means *"how well the simulator matches reality."*

Note: the `Vtec-S4 ... MainData.csv` in the repo root is a **parameters** sheet (mass, inertia,
added-mass), **not** a trajectory — it cannot serve as a validation reference on its own.

### Honest status chain

| Link | Status |
|---|---|
| Engine ≈ Fossen Otter *model* | ✅ verified, <0.03% (both regimes) |
| Fossen Otter model ≈ *real* Otter USV | ⚠️ partial (Fossen params are CAD/estimate-based, not this-boat sea-trial IDs) |
| Engine ≈ *real* water | ❌ not yet established — needs telemetry (ARAMUSS objective #1) |

---

## Protocol 2 — Standard maneuvering suite (task T1.1) — planned

- **Zig-zag (10°/10°, 20°/20°)** and **turning circle** (advance, transfer, tactical diameter).
- These are the IMO/ITTC standard maneuvers; documented physical benchmark data exists for
  displacement ships (SIMMAN KVLCC2/KCS), but those are rudder-steered ships — not directly
  comparable to a thruster-driven USV, and the raw tank time-series is not publicly downloadable.
  Prefer real Otter/Vtec telemetry for a like-for-like physical check.
