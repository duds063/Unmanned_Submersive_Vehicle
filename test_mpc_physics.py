#!/usr/bin/env python3
"""Debug MPC waypoint completion with actual physics simulation."""

import numpy as np
from sensor_engine import EKFState, SensorEngine, ExtendedKalmanFilter, Environment
from control_engine import ControlEngine
from mpc_controller import integrate_mpc
from physics_engine import PhysicsEngine, GeometryEngine, VehicleState

# Setup
geo = GeometryEngine(L=0.8, D=0.1)
physics = PhysicsEngine(geo, max_thruster_force=10.0)
env = Environment(pool_depth=10.0, pool_radius=50.0)
sensors = SensorEngine(env, noise_scale=0.0, enable_rayleigh=False, seed=42)
ekf = ExtendedKalmanFilter(physics)

import sys, io
old_stdout = sys.stdout
sys.stdout = io.StringIO()
control = ControlEngine(physics, hover_depth=5.0)
integrate_mpc(control, hover_depth=5.0)
sys.stdout = old_stdout

control.set_controller('mpc')

# Set waypoint
waypoint = np.array([5.0, 0.0, 5.0], dtype=float)
control.set_waypoints([waypoint], waypoint_threshold=0.5)

# Initialize vehicle
initial_state = VehicleState(z=5.0)
physics.reset(initial_state)
ekf.reset(np.concatenate([initial_state.eta, initial_state.nu]))

print(f"Target waypoint: {waypoint}")
print(f"Waypoint threshold: {control._waypoint_threshold}")
print(f"Intermediate threshold: 1.5 * {control._waypoint_threshold} = {1.5 * control._waypoint_threshold}")
print()

# Simulate 500 steps
dt = 0.01
for step in range(500):
    # Read sensors
    bundle = sensors.read(physics.state, physics.time)
    
    # Update EKF
    ekf.predict(dt)
    ekf.update_imu(bundle.imu)
    ekf.update_barometer(bundle.barometer)
    est = ekf.state_estimate
    
    # Compute control
    cmd = control.compute(est, physics.time)
    
    # Step physics
    physics.step(**cmd.__dict__, dt=dt)
    
    # Print progress
    if step % 50 == 0 or step < 10:
        pos = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
        horiz_dist = np.linalg.norm(pos[:2] - waypoint[:2])
        near = control.is_near_waypoint(pos)
        print(f"Step {step:3d}: pos=[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:5.2f}], " +
              f"horiz={horiz_dist:.2f}m, near={near}, " +
              f"cmd_power={cmd.thruster_power:.3f}, wp_idx={control.waypoint_index}, " +
              f"reached={control.check_waypoint_reached(pos) if step == 0 else '?'}")

final_pos = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
final_dist = np.linalg.norm(final_pos[:2] - waypoint[:2])
print()
print(f"Final position: {final_pos}")
print(f"Final distance to waypoint: {final_dist:.2f}m")
print(f"Final waypoint index: {control.waypoint_index}")
print(f"Mission complete: {control.mission_complete}")
