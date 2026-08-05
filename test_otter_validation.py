"""Regression guard for the Otter cross-verification (campaign task T0.1).

This is a VERIFICATION test, not a validation against real-world data: it checks that the
physics engine, parameterized as a Fossen Otter and driven by a wrench injected directly into
its equations of motion, reproduces an INDEPENDENT re-implementation of the standard Fossen
Otter 3-DOF model (``tools/otter_reference.py``) to within 1% normalized RMS on every variable.

Two maneuvers, both gated at 1%:
  * ``lateral_impulse`` — gentle 15 N sway impulse then free drift (linear regime).
  * ``turning``         — accelerate-then-turning-circle (nonlinear regime).

Measured on 2026-07-27 (after adding the added-mass Coriolis term to the engine):
    lateral_impulse: max nRMS 0.032%
    turning        : max nRMS 0.029%
So the engine agrees with the reference to <0.1% and the 1% gate leaves headroom for
cross-platform float noise while still tripping on any real dynamics regression (verified:
reverting the added-mass Coriolis pushes the turning maneuver back to ~22%).

Swap the golden CSV in ``data/`` for a real telemetry file of the same maneuver and this same
harness becomes a physical *validation* test — see docs/VALIDATION.md.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent      # otter/
ROOT = HERE.parent                          # repo root
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import otter_reference as otr
import otter_validation as ov

# Strict gate: every variable must agree with the Fossen reference to within 1% nRMS.
STRICT_GATE = 0.01


@pytest.mark.parametrize("maneuver", sorted(otr.MANEUVERS))
def test_reference_csv_matches_generator(maneuver):
    """Each committed golden CSV must match what tools/otter_reference.py produces."""
    ref = ov.load_reference_csv(maneuver)
    cols = otr.simulate(maneuver).as_columns()
    assert ref["t"].size == otr.simulate(maneuver).time.size
    for var in otr.STATE_VARS:
        assert np.allclose(ref[var], cols[var], rtol=1e-6, atol=1e-6), (
            f"{maneuver}:{var} drifted from generator"
        )


def test_maneuvers_are_nondegenerate():
    """Guard against trivially-passing runs: each maneuver must actually move the vehicle."""
    impulse = ov.load_reference_csv("lateral_impulse")
    assert float(np.ptp(impulse["y"])) > 0.1
    turning = ov.load_reference_csv("turning")
    assert float(np.ptp(turning["x"])) > 1.0
    assert float(np.ptp(turning["psi"])) > 0.5


@pytest.mark.parametrize("maneuver", sorted(otr.MANEUVERS))
def test_engine_matches_fossen_otter_within_1_percent(maneuver):
    """Verification gate: engine reproduces the Fossen Otter reference to <1% nRMS per variable."""
    nrms = ov.validate(maneuver)
    for var, value in nrms.items():
        assert value <= STRICT_GATE, (
            f"[{maneuver}] nRMS[{var}] = {value:.4f} exceeds the {STRICT_GATE:.1%} gate "
            f"(engine dynamics diverged from the Fossen Otter reference)"
        )
