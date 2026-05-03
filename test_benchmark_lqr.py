#!/usr/bin/env python3
"""Diagnose why benchmark breaks LQR when isolated test works."""

import numpy as np
import sys, io
from sensor_engine import (
    EKFState, SensorEngine, ExtendedKalmanFilter, Environment
)
from control_engine import ControlEngine
from mpc_controller import integrate_mpc
from rl_controller import integrate_rl
from physics_engine import PhysicsEngine, GeometryEngine, VehicleState
from benchmark_engine import BenchmarkScenario

# Suppress stdout during initialization
old_stdout = sys.stdout
sys.stdout = io.StringIO()

# Create benchmark scenario (mission mode) - WITH DEFAULT NOISE/DISTURBANCE
scenario = BenchmarkScenario(
    waypoints=[[5.0, 0.0, 5.0]],
    static_obstacles=[],
    dynamic_obstacles=[],
    benchmark_mode="mission",
    position_tolerance_m=0.3,
    pool_depth=10.0,
    pool_radius=50.0,
    noise_scale=0.5,  # DEFAULT: high noise
    rayleigh_enabled=True,  # DEFAULT: enabled
    rayleigh_sigma=0.03,  # DEFAULT
    env_disturbance_scale=0.5,  # DEFAULT: disturbance enabled
    dt=0.01,
    max_steps=500,
)

# Setup identical to benchmark
geo = GeometryEngine(L=0.8, D=0.1)
physics = PhysicsEngine(geo, max_thruster_force=10.0)
env = Environment(pool_depth=scenario.pool_depth, pool_radius=scenario.pool_radius)
sensors = SensorEngine(
    env,
    noise_scale=scenario.noise_scale,
    rayleigh_sigma=0.1,
    enable_rayleigh=scenario.rayleigh_enabled,
    seed=42,
)
sensors.set_environmental_disturbance(
    enabled=scenario.rayleigh_enabled,
    scale=scenario.env_disturbance_scale,
    rayleigh_sigma=0.1,
)
ekf = ExtendedKalmanFilter(physics, pool_radius=scenario.pool_radius, pool_depth=scenario.pool_depth)

control = ControlEngine(physics, hover_depth=scenario.pool_depth / 2.0)
integrate_mpc(control, hover_depth=scenario.pool_depth / 2.0)
integrate_rl(control, './checkpoints')

sys.stdout = old_stdout

# Set waypoint (benchmark uses 0.3m tolerance)
waypoint = np.array([5.0, 0.0, 5.0], dtype=float)
control.set_waypoints([waypoint], waypoint_threshold=scenario.position_tolerance_m)
control.set_controller('lqr')

# Initialize
initial_state = VehicleState(z=scenario.pool_depth / 2.0)
physics.reset(initial_state)
ekf.reset(np.concatenate([initial_state.eta, initial_state.nu]))

print(f"Benchmark LQR Diagnostic")
print(f"=======================")
print(f"Scenario: mission, waypoint=[5.0, 0.0, 5.0], tolerance=0.3m")
print(f"Pool: depth={scenario.pool_depth}m, radius={scenario.pool_radius}m")
print(f"Noise: scale={scenario.noise_scale}, rayleigh={scenario.rayleigh_enabled}")
print(f"Environmental disturbance: {scenario.env_disturbance_scale}")
print()

dt = scenario.dt
max_steps = 500  # Full benchmark duration
collided = False

for step in range(max_steps):
    # Read sensors and update EKF
    bundle = sensors.read(physics.state, physics.time)
    
    ekf.predict(scenario.dt)
    ekf.update_imu(bundle.imu)
    ekf.update_barometer(bundle.barometer)
    ekf.update_sonar(bundle.sonar)
    ekf.update_sonar_position(bundle.sonar)  # NEW: Constrain X/Y position using sonar
    est = ekf.state_estimate
    
    # Compute control
    cmd = control.compute(est, physics.time)
    
    # Step physics
    physics.step(**cmd.__dict__, dt=dt)
    
    # Check for collision
    pos = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
    horiz_dist = np.linalg.norm(pos[:2] - waypoint[:2])
    
    # Calculate EKF error
    ekf_error = np.linalg.norm(est.position[:2] - pos[:2])
    
    # Print diagnostics
    if step % 50 == 0 or step < 5 or step >= 450 or ekf_error > 2.0:
        print(f"Step {step:3d}: EKF_error={ekf_error:.2f}m, phys_dist={horiz_dist:.2f}m, attitude=[{np.degrees(physics.state.phi):7.2f}°, {np.degrees(physics.state.tht):7.2f}°]")
        
        # Check for collision
        if horiz_dist > 15.0 or abs(physics.state.tht) > np.radians(45):
            print(f"  ⚠️ WARNING: Large error or attitude! Distance={horiz_dist:.2f}m, pitch={np.degrees(physics.state.tht):.2f}°")
            collided = True

final_pos = np.array([physics.state.x, physics.state.y, physics.state.z], dtype=float)
final_dist = np.linalg.norm(final_pos[:2] - waypoint[:2])
print(f"\nFinal Results:")
print(f"  Final position: {final_pos}")
print(f"  Final distance: {final_dist:.2f}m")
print(f"  Final attitude: [{np.degrees(physics.state.phi):.2f}°, {np.degrees(physics.state.tht):.2f}°]")
print(f"  Collision detected: {collided}")
print(f"  Waypoint index: {control.waypoint_index}/{len(control._waypoints)}")
