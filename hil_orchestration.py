"""Lightweight HIL orchestration helpers.

This module provides a small registry for multiple HIL devices so the
mission loop can expose a Level-4 style snapshot: multiple bridges,
telemetry state, and an optional digital twin callback.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Any


@dataclass
class HILDeviceHandle:
    name: str
    bridge: object
    role: str = 'sensor'
    enabled: bool = True
    last_sensor: Optional[dict] = None
    last_ack_ok: Optional[bool] = None


class HILOrchestrator:
    """Registry for multiple HIL endpoints plus an optional twin snapshot."""

    def __init__(self):
        self.devices: Dict[str, HILDeviceHandle] = {}
        self._digital_twin_provider: Optional[Callable[[], Dict[str, Any]]] = None

    def register_device(self, name: str, bridge: object, role: str = 'sensor') -> HILDeviceHandle:
        handle = HILDeviceHandle(name=name, bridge=bridge, role=role)
        self.devices[name] = handle
        return handle

    def unregister_device(self, name: str) -> None:
        self.devices.pop(name, None)

    def set_digital_twin_provider(self, provider: Optional[Callable[[], Dict[str, Any]]]) -> None:
        self._digital_twin_provider = provider

    def start_all(self) -> None:
        for handle in self.devices.values():
            if hasattr(handle.bridge, 'start'):
                try:
                    handle.bridge.start()
                except Exception:
                    handle.enabled = False

    def stop_all(self) -> None:
        for handle in self.devices.values():
            if hasattr(handle.bridge, 'stop'):
                try:
                    handle.bridge.stop()
                except Exception:
                    pass

    def send_command(self, name: str, payload: bytes, mode: str = 'json', ack_timeout: float = 0.2) -> bool:
        handle = self.devices[name]
        bridge = handle.bridge
        if hasattr(bridge, 'send_command_with_ack'):
            ok = bool(bridge.send_command_with_ack(payload, mode=mode, ack_timeout=ack_timeout))
            handle.last_ack_ok = ok
            return ok
        if hasattr(bridge, 'send_command'):
            bridge.send_command(payload, mode=mode)
            handle.last_ack_ok = None
            return True
        return False

    def recv_sensors(self, name: str, timeout: Optional[float] = None) -> Optional[dict]:
        handle = self.devices[name]
        bridge = handle.bridge
        if hasattr(bridge, 'recv_sensors'):
            frame = bridge.recv_sensors(timeout=timeout)
            handle.last_sensor = frame
            return frame
        return None

    def snapshot(self) -> Dict[str, Any]:
        devices_snapshot = {
            name: {
                'role': handle.role,
                'enabled': handle.enabled,
                'bridge': type(handle.bridge).__name__,
                'last_ack_ok': handle.last_ack_ok,
                'has_sensor': handle.last_sensor is not None,
            }
            for name, handle in self.devices.items()
        }
        twin_snapshot = self._digital_twin_provider() if self._digital_twin_provider is not None else {}
        return {
            'devices': devices_snapshot,
            'digital_twin': twin_snapshot,
        }
