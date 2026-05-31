import json
import socket
import time
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from geometry_engine import GeometryEngine
from mission_engine import MissionEngine
from manual_control_sources import (
    KeyboardManualCommandSource,
    JoystickManualCommandSource,
    ReplayManualCommandSource,
    UDPManualCommandSource,
)


def main() -> None:
    k = KeyboardManualCommandSource(power_step=0.2, theta_step_deg=10.0, phi_step_deg=15.0, ballast_step=0.3)
    k.update_keys({"w", "i", "u"})
    cmd = k.read(None, 0.0)
    assert cmd.thruster_power > 0.0 and cmd.thruster_theta > 0.0 and cmd.ballast_cmd > 0.0

    j = JoystickManualCommandSource()
    j.update_axes({"throttle": 0.5, "throttle2": -0.25, "theta": 0.2, "phi": -0.1, "ballast": 0.4})
    cmd = j.read(None, 0.0)
    assert abs(cmd.thruster_power - 0.5) < 1e-6
    assert abs(cmd.thruster2_power - (-0.25)) < 1e-6

    replay = ReplayManualCommandSource.from_records(
        [
            {"time_s": 0.0, "command": {"thruster_power": 0.1, "ballast_cmd": 0.0}},
            {"time_s": 0.2, "command": {"thruster_power": 0.6, "ballast_cmd": -0.2}},
        ],
        use_time=True,
    )
    cmd0 = replay.read(None, 0.0)
    cmd1 = replay.read(None, 0.3)
    assert abs(cmd0.thruster_power - 0.1) < 1e-6
    assert abs(cmd1.thruster_power - 0.6) < 1e-6

    udp = UDPManualCommandSource(host="127.0.0.1", port=0, autostart=True)
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        packet = {"command": {"thruster_power": 0.55, "ballast_cmd": -0.15}}
        sender.sendto(json.dumps(packet).encode("utf-8"), ("127.0.0.1", udp.local_port))
        sender.close()

        cmd_udp = None
        for _ in range(30):
            time.sleep(0.02)
            cmd_udp = udp.read(None, 0.0)
            if abs(cmd_udp.thruster_power - 0.55) < 1e-6:
                break

        assert cmd_udp is not None
        assert abs(cmd_udp.thruster_power - 0.55) < 1e-6
        assert abs(cmd_udp.ballast_cmd - (-0.15)) < 1e-6
    finally:
        udp.stop()

    geo = GeometryEngine(L=0.8, D=0.1)
    mission = MissionEngine(
        geo,
        checkpoint_dir="./checkpoints_test",
        control_mode="manual",
        manual_source_mode="replay",
        manual_source_config={
            "records": [
                {"time_s": 0.0, "command": {"thruster_power": 0.25, "ballast_cmd": 0.0}},
                {"time_s": 1.0, "command": {"thruster_power": 0.5, "ballast_cmd": 0.1}},
            ],
            "use_time": True,
        },
    )
    result = mission._run_episode(dt=0.01, training=False, max_steps=1)
    assert result.total_steps == 1
    assert mission.control.active_controller == "manual"

    print("manual input sources validation passed")


if __name__ == "__main__":
    main()
