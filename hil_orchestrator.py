"""
Lightweight HIL Orchestrator

Manages multiple HIL bridge instances and provides a consolidated
snapshot for telemetry dashboards. This is a minimal implementation
used by MissionEngine.telemetry aggregation.
"""
import time
from typing import Dict, Any, List


class HILOrchestrator:
    def __init__(self):
        self.devices: Dict[str, Any] = {}

    def register(self, name: str, bridge: object) -> None:
        self.devices[name] = {
            'bridge': bridge,
            'last_seen': time.time(),
            'status': 'unknown',
        }

    def unregister(self, name: str) -> None:
        if name in self.devices:
            del self.devices[name]

    def touch(self, name: str) -> None:
        if name in self.devices:
            self.devices[name]['last_seen'] = time.time()
            self.devices[name]['status'] = 'online'

    def snapshot(self) -> Dict[str, Any]:
        """Return a telemetry-friendly snapshot of registered devices."""
        out: Dict[str, Any] = {}
        for name, info in self.devices.items():
            bridge = info.get('bridge')
            out[name] = {
                'status': info.get('status', 'unknown'),
                'last_seen': info.get('last_seen'),
                'bridge_type': type(bridge).__name__ if bridge is not None else None,
            }
        return out
