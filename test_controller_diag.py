"""Diagnostic test to check controller output and state."""
import numpy as np
from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine, VehicleState
from sensor_engine import SensorEngine, ExtendedKalmanFilter, Environment
from control_engine import ControlEngine
from mpc_controller import integrate_mpc
from rl_controller import integrate_rl

# Setup
geo = GeometryEngine(L=0.8, D=0.1)
physics = PhysicsEngine(geo, max_thruster_force=10.0)
env = Environment(pool_depth=10.0, pool_radius=30.0)
sensors = SensorEngine(env, noise_scale=0.5, enable_rayleigh=False)
ekf = ExtendedKalmanFilter(physics)
control = ControlEngine(physics, hover_depth=5.0)

# Initialize controllers
integrate_mpc(control, hover_depth=5.0)
hrl = integrate_rl(control, "./checkpoints")

# Set up mission
target = np.array([5.0, 0.0, 5.0], dtype=float)
control.set_waypoints([target], waypoint_threshold=0.2)
control.set_controller('lqr')

# Reset and run
physics.reset(VehicleState(z=5.0))
ekf.reset(np.concatenate([physics.state.eta, physics.state.nu]))

print("Initial state:")
print(f"  Position: {physics.state.eta[:3]}")
print(f"  Attitude (degrees): {np.degrees(physics.state.eta[3:6])}")
print()

# Run for a few steps
for step in range(20):
    bundle = sensors.read(physics.state, physics.time)
    
    ekf.predict(0.01)
    ekf.update_imu(bundle.imu)
    ekf.update_barometer(bundle.barometer)
    ekf.update_sonar(bundle.sonar)
    
    cmd = control.compute(ekf.state_estimate, physics.time)
    
    if step % 5 == 0:
        print(f"Step {step}:")
        print(f"  Position: {physics.state.eta[:3]}")
        print(f"  Attitude (deg): {np.degrees(physics.state.eta[3:6])}")
        print(f"  Velocity: {physics.state.nu[:3]}")
        print(f"  Command: power={cmd.thruster_power:.3f}, theta={np.degrees(cmd.thruster_theta):.1f}°, phi={np.degrees(cmd.thruster_phi):.1f}°")
        print()
    
    physics.step(
        thruster_power=cmd.thruster_power,
        thruster_theta=cmd.thruster_theta,
        thruster_phi=cmd.thruster_phi,
        thruster2_power=cmd.thruster2_power,
        thruster2_theta=cmd.thruster2_theta,
        thruster2_phi=cmd.thruster2_phi,
        ballast_cmd=cmd.ballast_cmd,
        dt=0.01,
    )

print("Final state:")
print(f"  Position: {physics.state.eta[:3]}")
print(f"  Attitude (deg): {np.degrees(physics.state.eta[3:6])}")
