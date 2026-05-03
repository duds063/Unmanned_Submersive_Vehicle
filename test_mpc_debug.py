#!/usr/bin/env python3
"""Deep dive into MPC guidance law to understand the braking issue."""

import numpy as np
from sensor_engine import EKFState, SensorEngine, ExtendedKalmanFilter, Environment
from control_engine import ControlEngine
from mpc_controller import integrate_mpc
from physics_engine import PhysicsEngine, GeometryEngine, VehicleState
import sys, io

# Setup
geo = GeometryEngine(L=0.8, D=0.1)
physics = PhysicsEngine(geo, max_thruster_force=10.0)
env = Environment(pool_depth=10.0, pool_radius=50.0)
sensors = SensorEngine(env, noise_scale=0.0, enable_rayleigh=False, seed=42)
ekf = ExtendedKalmanFilter(physics)

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

print(f"Target: {waypoint}, Threshold: 0.5m")
print()

# Test a few steps and examine the guidance values
dt = 0.01
for step in range(200):
    # Read sensors
    bundle = sensors.read(physics.state, physics.time)
    
    # Update EKF
    ekf.predict(dt)
    ekf.update_imu(bundle.imu)
    ekf.update_barometer(bundle.barometer)
    est = ekf.state_estimate
    
    # Get current position and velocity
    pos = est.position
    vel_linear = est.velocity_linear
    
    # Manually compute what MPC should see
    target = waypoint
    delta_world = target - pos
    horiz_dist = float(np.linalg.norm(delta_world[:2]))
    surge = float(vel_linear[0])
    
    # MPC guidance speed logic
    max_cruise = 1.15
    speed_from_distance = max_cruise * float(np.tanh(horiz_dist / 2.8))
    heading_scale = 0.08 + 0.92  # assume aligned
    desired_surge = speed_from_distance * heading_scale
    if horiz_dist < 1.5:
        desired_surge *= float(np.clip(horiz_dist / 1.5, 0.0, 1.0))
    
    # Control law
    base_power = 0.42 * (desired_surge - surge) - 0.10 * np.tanh(surge / 0.45)
    base_power = float(np.clip(base_power, -0.28, 0.24))
    
    # Compute control
    cmd = control.compute(est, physics.time)
    
    # Step physics
    physics.step(**cmd.__dict__, dt=dt)
    
    if step % 20 != 0:
        cmd = control.compute(est, physics.time)
        physics.step(**cmd.__dict__, dt=dt)
        continue
    
    print(f"Step {step}:")
    print(f"  EKF pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print(f"  EKF vel_linear: [{vel_linear[0]:.4f}, {vel_linear[1]:.4f}, {vel_linear[2]:.4f}]")
    print(f"  Physics pos: [{physics.state.x:.4f}, {physics.state.y:.4f}, {physics.state.z:.4f}]")
    print(f"  Horiz dist: {horiz_dist:.4f}m")
    print(f"  Surge (velocity[0]): {surge:.4f} m/s")
    print(f"  Desired surge: {desired_surge:.4f} m/s")
    print(f"  Error term: {desired_surge - surge:.4f}")
    print(f"  Base power (computed): {base_power:.4f}")
    print(f"  Actual command power: {cmd.thruster_power:.4f}")
    print(f"  Reference in MPC: {control._mpc._x_ref[:3]}")
    print()
