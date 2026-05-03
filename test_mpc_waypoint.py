#!/usr/bin/env python3
"""Quick diagnostic for MPC waypoint completion logic."""

import numpy as np
from sensor_engine import EKFState
from control_engine import ControlEngine
from mpc_controller import integrate_mpc
from physics_engine import PhysicsEngine, GeometryEngine, VehicleState

# Setup
geo = GeometryEngine(L=0.8, D=0.1)
physics = PhysicsEngine(geo, max_thruster_force=10.0)
control = ControlEngine(physics, hover_depth=2.5)
integrate_mpc(control, hover_depth=2.5)
control.set_controller('mpc')

# Set waypoint
waypoint = np.array([5.0, 0.0, 5.0], dtype=float)
control.set_waypoints([waypoint], waypoint_threshold=0.5)

# Initialize vehicle at origin
initial_state = VehicleState(z=5.0)
physics.reset(initial_state)

# Test state
ekf_state = EKFState(
    eta=np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=float),
    nu=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
    P=np.eye(12),
    timestamp=0.0
)

print(f"Initial position: {ekf_state.position}")
print(f"Target waypoint: {waypoint}")
print(f"Current waypoint index: {control.waypoint_index}")
print(f"Waypoint threshold: {control._waypoint_threshold}")
print()

# Step 1: Check if is_near_waypoint method exists and works
print(f"Has is_near_waypoint? {hasattr(control, 'is_near_waypoint')}")
if hasattr(control, 'is_near_waypoint'):
    near = control.is_near_waypoint(ekf_state.position)
    print(f"Is near waypoint (at 0,0,5 -> 5,0,5)? {near}")
print()

# Step 2: Run MPC once
print("Running MPC compute...")
cmd = control.compute(ekf_state, 0.0)
print(f"Command: power={cmd.thruster_power:.4f}, theta={cmd.thruster_theta:.4f}, phi={cmd.thruster_phi:.4f}")
print(f"Command 2: power={cmd.thruster2_power:.4f}, theta={cmd.thruster2_theta:.4f}")
print()

# Step 3: Check MPC controller state
mpc = control._mpc
print(f"MPC has control_engine? {hasattr(mpc, 'control_engine') and mpc.control_engine is not None}")
print(f"MPC control_engine is same object? {mpc.control_engine is control if hasattr(mpc, 'control_engine') else False}")
print()

# Step 4: Simulate a few steps forward
print("Simulating 100 steps...")
for step in range(100):
    # Advance ekf state by moving forward
    if step > 0 and step % 10 == 0:
        # Move 0.1m per 10 steps in x direction (crude simulation)
        ekf_state = EKFState(
            eta=np.array([step * 0.01, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=float),
            nu=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            P=np.eye(12),
            timestamp=step * 0.1
        )
        cmd = control.compute(ekf_state, step * 0.1)
        horiz_dist = np.linalg.norm(ekf_state.position[:2] - waypoint[:2])
        near = control.is_near_waypoint(ekf_state.position)
        print(f"Step {step}: pos=[{ekf_state.position[0]:.2f}, {ekf_state.position[1]:.2f}, {ekf_state.position[2]:.2f}], " +
              f"horiz_dist={horiz_dist:.2f}m, near={near}, " +
              f"cmd_power={cmd.thruster_power:.4f}, wp_idx={control.waypoint_index}")

print(f"\nFinal: waypoint_index={control.waypoint_index}, mission_complete={control.mission_complete}")
