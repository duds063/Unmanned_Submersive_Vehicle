"""Vehicle profile loaders for benchmark calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class VehicleProfile:
    name: str
    mass_kg: float
    inertia_kgm2: np.ndarray
    length_m: float
    beam_m: float
    beam_total_m: float
    draft_m: float
    thruster_port_position_m: np.ndarray
    thruster_starboard_position_m: np.ndarray
    cog_position_m: Optional[np.ndarray] = None
    cob_position_m: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    X_uu: float = 0.0
    Y_vv: float = 0.0
    Z_ww: float = 0.0
    K_pp: float = 0.0
    M_qq: float = 0.0
    N_rr: float = 0.0
    X_u: float = 0.0
    Y_v: float = 0.0
    Z_w: float = 0.0
    K_p: float = 0.0
    M_q: float = 0.0
    N_r: float = 0.0
    X_udot: float = 0.0
    Y_vdot: float = 0.0
    Z_wdot: float = 0.0
    K_pdot: float = 0.0
    M_qdot: float = 0.0
    N_rdot: float = 0.0


def _array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _vtec_s4_base_profile(source_path: str | None = None) -> VehicleProfile:
    return VehicleProfile(
        name="Vtec-S4",
        mass_kg=39.511702,
        inertia_kgm2=_array([
            [7.081866048, 0.004666351957, -0.1420442146],
            [0.004666351957, 6.529613972, -0.03545440812],
            [-0.1420442146, -0.03545440812, 3.60496537],
        ]),
        length_m=1.405072,
        beam_m=0.259267,
        beam_total_m=0.518534,
        draft_m=0.094,
        thruster_port_position_m=_array([-0.45, -0.2725, 0.275]),
        thruster_starboard_position_m=_array([-0.45, 0.2725, 0.275]),
        cog_position_m=_array([-0.09933087087, 0.0, -0.1347069574]),
        cob_position_m=_array([-0.05430119811, 0.0, 0.0384]),
        source_path=source_path,
        X_udot=1.1898,
        Y_vdot=14.9220,
        Z_wdot=49.6246,
        K_pdot=3.9017,
        M_qdot=3.9333,
        N_rdot=1.1902,
    )


def load_vtec_s4_profile(csv_path: str | Path | None = None) -> VehicleProfile:
    """Return the literal Vtec-S4 profile used by the benchmark."""
    source_path = None if csv_path is None else str(Path(csv_path))
    return _vtec_s4_base_profile(source_path=source_path)


def load_taluy_profile(source_path: str | Path | None = None) -> VehicleProfile:
    """Return the Taluy profile described by the provided vehicle data."""
    normalized_source = None if source_path is None else str(Path(source_path))
    metadata = {
        "model_xml_path": "asset_zoo/vehicles/taluy/taluy_mjcf.xml",
        "body_name": "taluy_body",
        "hydro": {
            "linear_damping_matrix": [
                [45.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 42.37, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 89.2, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 5.646],
            ],
            "quadratic_damping_matrix": [
                [141.9, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 221.7, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 136.16, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 2.608],
            ],
            "added_mass_6x6": [[0.0] * 6 for _ in range(6)],
            "fluid_density_kg_m3": 1025.0,
            "gravity_m_s2": 9.81,
            "buoyancy_n": 295.0,
            "displaced_volume_m3": 0.029337908057979662,
            "center_of_buoyancy_b_m": [0.0, 0.0, 0.05],
            "current_world_m_s": [0.0, 0.0, 0.0],
            "current_body_m_s": None,
            "include_restoring": True,
            "include_damping": True,
            "include_added_mass": False,
            "include_added_coriolis": False,
        },
        "thruster": {
            "model": "t200",
            "site_names": [
                "thruster_0_site",
                "thruster_1_site",
                "thruster_2_site",
                "thruster_3_site",
                "thruster_4_site",
                "thruster_5_site",
                "thruster_6_site",
                "thruster_7_site",
            ],
        },
        "body_wrench_limit": [165.43806975086645, 163.72246043976622, 199.37886486443375, 48.168, 38.49599999999993, 97.02392361020456],
        "demo": {
            "surge_command": [0.84, 0.84, -0.84, -0.84, -6.0, -6.0, 6.0, 6.0],
            "yaw_command": [0.0, 0.0, 0.0, 0.0, 2.0, -2.0, -2.0, 2.0],
            "heave_command": [-4.0, -4.0, -4.0, -4.0, 0.0, 0.0, 0.0, 0.0],
            "coast_command": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "viewer": {
            "distance": 3.0,
            "elevation": -12.0,
            "azimuth": 55.0,
        },
        "control_limits": {
            "command_limit": 60.0,
            "tau_s": 0.05,
            "force_deadzone_n": 0.5,
            "min_thrust_n": -60.0,
            "max_thrust_n": 60.0,
        },
        "pwm": {
            "supply_voltage": 16.0,
            "pwm_min_us": 1100.0,
            "pwm_max_us": 1900.0,
            "pwm_neutral_us": 1500.0,
            "newton_per_kgf": 9.81,
            "force_to_pwm_coeffs_forward": [
                -7.65890867108129,
                -2.272619267631361,
                1.5070809906898863,
                140.79738360070297,
                -50.325593642656244,
                1954.5975713257544,
            ],
            "force_to_pwm_coeffs_reverse": [
                12.292935764557399,
                -3.0745135927968605,
                -1.0664649834600253,
                181.68212348405177,
                36.69456713973092,
                1151.1307183400536,
            ],
        },
    }
    return VehicleProfile(
        name="Taluy",
        mass_kg=29.8419,
        inertia_kgm2=_array([
            [0.931063, 0.031700, -0.129000],
            [0.031700, 2.108063, 0.107000],
            [-0.129000, 0.107000, 2.180000],
        ]),
        length_m=0.692626,
        beam_m=0.225406,
        beam_total_m=0.450812,
        draft_m=0.094,
        thruster_port_position_m=_array([-0.346313, 0.225406, -0.031653]),
        thruster_starboard_position_m=_array([-0.346313, -0.225406, -0.031653]),
        cog_position_m=_array([0.0, 0.0, -0.02]),
        cob_position_m=_array([0.0, 0.0, 0.05]),
        source_path=normalized_source,
        X_udot=1.1898,
        Y_vdot=14.9220,
        Z_wdot=49.6246,
        K_pdot=3.9017,
        M_qdot=3.9333,
        N_rdot=1.1902,
        metadata=metadata,
    )


def load_vehicle_profile(profile_source: str | Path | None = None) -> VehicleProfile:
    """Load a vehicle profile from a path or profile identifier."""
    if profile_source is None:
        return load_vtec_s4_profile(None)

    path = Path(str(profile_source))
    hint = f"{path.as_posix().lower()} {path.name.lower()} {path.stem.lower()}"
    if "taluy" in hint:
        return load_taluy_profile(path)
    return load_vtec_s4_profile(path)
