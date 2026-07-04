"""
Manual Teleoperation Test Rig.

This script demonstrates how to use `UDPManualCommandSource` to control the USV
simulation in real-time. It sets up a listener and prints the received commands,
which can then be passed to the Control or Mission engines.
"""
import time
import sys
import os
import numpy as np

# Ensure project root is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manual_control_sources import build_manual_source
from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine, ComponentMasses
from sensor_engine import Environment
from vehicle_profiles import get_rig_profiles, load_vehicle_profile

def run_manual_test(profile_name="default"):
    # 0. Load Vehicle Profile
    rig_profiles = get_rig_profiles()

    if profile_name in rig_profiles:
        prof = rig_profiles[profile_name]
        # Initialize Simulation Stack for the simplified rig profiles
        geo = GeometryEngine(**prof["geometry"])
        physics = PhysicsEngine(geo, max_thruster_force=prof["max_force"])
        name, l_val, d_val, force = profile_name.upper(), geo.L, geo.D, prof["max_force"]
    else:
        # Load full vehicle profile from the registry (e.g. "taluy")
        profile = load_vehicle_profile(profile_name)
        # The classmethod handles the complex multi-thruster setup from metadata
        physics = PhysicsEngine.from_vehicle_profile(profile)
        name, l_val, d_val, force = profile.name, profile.length_m, profile.beam_m, 10.0

    env = Environment(pool_depth=5.0, pool_radius=20.0)
    
    # Reset state to a neutral floating position
    physics.reset()

    # 1. Initialize the manual command source via the factory
    # We use UDP because it's non-blocking and easy to test externally.
    config = {
        "host": "127.0.0.1",
        "port": 14570,
        "autostart": True
    }
    source = build_manual_source("udp", config)

    print(f"=== USV Manual Test Rig ({name} Profile) ===")
    print(f"Listening for JSON commands on {config['host']}:{config['port']}")
    print(f"Vehicle: L={l_val}m, D={d_val}m, Max Force={force}N")
    print("-" * 60)
    print("To test, run this in another terminal:")
    print("echo '{\"thruster_power\": 0.6, \"thruster_theta_deg\": 25}' | nc -u -w1 127.0.0.1 14570")
    print("-" * 60)
    print("Press Ctrl+C to stop.\n")

    last_cmd = None
    t_sim = 0.0
    dt = 0.1 # 10Hz loop

    try:
        while True:
            # 2. Read the latest command from the source
            # The UDP source runs a background thread that catches incoming packets.
            cmd = source.read(ekf_state=None, time_s=t_sim)

            # 3. Step the Physics Engine with the manual command
            physics.step(
                thruster_power=cmd.thruster_power,
                thruster_theta=cmd.thruster_theta,
                thruster_phi=cmd.thruster_phi,
                ballast_cmd=cmd.ballast_cmd,
                thruster2_power=cmd.thruster2_power,
                dt=dt
            )

            if cmd != last_cmd:
                print(f"[t={t_sim:0.1f}s] CMD -> P:{cmd.thruster_power:.2f} T:{np.degrees(cmd.thruster_theta):.1f}° | "
                      f"POS -> X:{physics.state.x:.2f} Y:{physics.state.y:.2f} Z:{physics.state.z:.2f}")
                last_cmd = cmd
            
            
            time.sleep(dt)
            t_sim += dt

    except KeyboardInterrupt:
        print("\nShutting down manual test...")
    finally:
        source.stop()

if __name__ == "__main__":
    # You can now pass the profile via command line or environment variable
    selected_profile = sys.argv[1] if len(sys.argv) > 1 else "default"
    run_manual_test(selected_profile)