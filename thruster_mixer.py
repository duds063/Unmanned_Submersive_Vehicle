"""Simple thruster mixer with configurable mapping.

This initial implementation uses a direct mapping (identity) and
provides a small helper to load JSON configs later.
"""
from typing import List, Sequence
import json


class ThrusterMixer:
    """Map abstract control signals to individual thruster outputs.

    For the initial version we assume the controller provides one power
    value per physical thruster. The mixer can be extended to support
    allocation matrices and pseudo-inverse solving for over-actuated layouts.
    """

    def __init__(self, num_thrusters: int = 2, config_path: str = None):
        self.num_thrusters = num_thrusters
        self.config = None
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                self.num_thrusters = self.config.get('num_thrusters', self.num_thrusters)

    def allocate(self, powers: Sequence[float]) -> List[float]:
        """Return a list of per-thruster outputs (clipped to [-1,1])."""
        out = [0.0] * self.num_thrusters
        for i in range(self.num_thrusters):
            if i < len(powers):
                v = float(powers[i])
            else:
                v = 0.0
            # clip
            if v > 1.0:
                v = 1.0
            if v < -1.0:
                v = -1.0
            out[i] = v
        return out
"""Thruster mixer: maps abstract ControlCommand into individual thruster signals.

This initial implementation supports a simple dual-thruster layout and a
configurable matrix for future expansion.
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ThrusterSpec:
    name: str
    position: np.ndarray  # 3D position of thruster relative to CoM
    direction: np.ndarray # unit vector of thrust direction
    max_force: float = 50.0


@dataclass
class ThrusterFaultProfile:
    """Fault/saturation profile for a single thruster."""

    failed: bool = False
    stuck_output: Optional[float] = None
    saturation_scale: float = 1.0
    noise_std: float = 0.0


class ThrusterMixer:
    """Simple mixer supporting 2 thrusters (catamaran-like) by default.

    The mixer returns an array of forces (N) for each thruster and optional
    PWM/command values in [-1,1].
    """

    def __init__(self, specs: Optional[List[ThrusterSpec]] = None):
        if specs is None:
            # default: two thrusters along +X at y=+/-0.3m
            specs = [
                ThrusterSpec('thruster_left',  position=np.array([0.0,  0.3, 0.0]), direction=np.array([1.0, 0.0, 0.0])),
                ThrusterSpec('thruster_right', position=np.array([0.0, -0.3, 0.0]), direction=np.array([1.0, 0.0, 0.0])),
            ]
        self.specs = specs
        self.fault_profiles = {spec.name: ThrusterFaultProfile() for spec in self.specs}

    def set_fault_profile(self, thruster_name: str, fault_profile: ThrusterFaultProfile) -> None:
        if thruster_name not in self.fault_profiles:
            raise KeyError(f"Unknown thruster '{thruster_name}'")
        self.fault_profiles[thruster_name] = fault_profile

    def clear_fault_profiles(self) -> None:
        for spec in self.specs:
            self.fault_profiles[spec.name] = ThrusterFaultProfile()

    def allocate(self, cmd_power: float, cmd_power2: Optional[float] = None) -> List[float]:
        """Map abstract forward power to each thruster.

        For the default layout we interpret `cmd_power` as forward thrust
        and `cmd_power2` as secondary thruster power (if provided).
        Values are in [-1,1]. Output is force in Newtons per thruster.
        """
        powers = []
        if cmd_power2 is None:
            # split power equally
            p_left = float(np.clip(cmd_power, -1.0, 1.0))
            p_right = p_left
        else:
            p_left = float(np.clip(cmd_power, -1.0, 1.0))
            p_right = float(np.clip(cmd_power2, -1.0, 1.0))

        for spec, p in zip(self.specs, [p_left, p_right]):
            fault = self.fault_profiles.get(spec.name, ThrusterFaultProfile())

            if fault.failed:
                commanded = 0.0
            elif fault.stuck_output is not None:
                commanded = float(fault.stuck_output)
            else:
                commanded = float(p)

            commanded *= float(max(0.0, fault.saturation_scale))
            commanded = float(np.clip(commanded, -1.0, 1.0))

            forces = commanded * spec.max_force * spec.direction
            scalar_force = float(np.linalg.norm(forces))

            if fault.noise_std > 0.0:
                scalar_force += float(np.random.normal(0.0, fault.noise_std))

            scalar_force = float(np.clip(scalar_force, 0.0, spec.max_force))
            powers.append(scalar_force)

        return powers
