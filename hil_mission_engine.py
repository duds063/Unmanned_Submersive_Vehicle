"""HIL-aware Mission Engine variant.

This module adds `HILMissionEngine` which subclasses `MissionEngine` and
can operate in a handshake (deterministic step) mode using an injected
HIL bridge (e.g., `MockLoopbackBridge`). The implementation here is a
small prototype: it provides `send_command_and_wait()` for the Mission
loop to call; full integration into `_run_episode()` will be done in a
subsequent change.
"""
from typing import Optional
import time

from mission_engine import MissionEngine, EpisodeResult
from hil_interface import MockLoopbackBridge, HILProtocol
from control_engine import ControlCommand


class HILMissionEngine(MissionEngine):
    def __init__(self, *args, hil_bridge: Optional[MockLoopbackBridge] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hil_bridge = hil_bridge or MockLoopbackBridge()

    def send_command_and_wait(self, cmd: ControlCommand, mode: str = 'json', timeout: float = 0.1):
        """Serialize `cmd`, send to the bridge, and wait for sensor reply.

        Returns the parsed sensor dict (may be empty if bridge has no data).
        """
        if mode == 'json':
            payload = HILProtocol.serialize_json_command(cmd)
        else:
            payload = HILProtocol.serialize_binary_command(cmd)

        # send and block for the emulated latency
        self.hil_bridge.send_command(payload, mode=mode)
        sensors = self.hil_bridge.recv_sensors(timeout=timeout)
        return sensors

    # NOTE: `_run_episode()` integration will be implemented in a follow-up
    # patch so we can keep changes small and reviewable.
