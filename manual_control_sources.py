"""Manual command sources for independent teleoperation and replay.

These sources are transport/input adapters that emit ControlCommand objects.
They can be used with ControlEngine manual mode in both USV and UAV simulations.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from control_engine import (
    ControlCommand,
    ManualCommandSource,
    QueuedManualCommandSource,
    SequenceManualCommandSource,
)


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_command(payload: Mapping[str, object], fallback: Optional[ControlCommand] = None) -> ControlCommand:
    base = fallback or ControlCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    theta = payload.get("thruster_theta")
    if theta is None and "thruster_theta_deg" in payload:
        theta = np.radians(_to_float(payload.get("thruster_theta_deg"), 0.0))
    theta = _to_float(theta, base.thruster_theta)

    phi = payload.get("thruster_phi")
    if phi is None and "thruster_phi_deg" in payload:
        phi = np.radians(_to_float(payload.get("thruster_phi_deg"), 0.0))
    phi = _to_float(phi, base.thruster_phi)

    theta2 = payload.get("thruster2_theta")
    if theta2 is None and "thruster2_theta_deg" in payload:
        theta2 = np.radians(_to_float(payload.get("thruster2_theta_deg"), 0.0))
    theta2 = _to_float(theta2, base.thruster2_theta if base.thruster2_theta is not None else theta)

    phi2 = payload.get("thruster2_phi")
    if phi2 is None and "thruster2_phi_deg" in payload:
        phi2 = np.radians(_to_float(payload.get("thruster2_phi_deg"), 0.0))
    phi2 = _to_float(phi2, base.thruster2_phi if base.thruster2_phi is not None else phi)

    return ControlCommand(
        thruster_power=_to_float(payload.get("thruster_power"), base.thruster_power),
        thruster_theta=theta,
        thruster_phi=phi,
        ballast_cmd=_to_float(payload.get("ballast_cmd"), base.ballast_cmd),
        thruster2_power=_to_float(payload.get("thruster2_power"), base.thruster2_power if base.thruster2_power is not None else base.thruster_power),
        thruster2_theta=theta2,
        thruster2_phi=phi2,
    ).clip()


class KeyboardManualCommandSource(QueuedManualCommandSource):
    """Key-state adapter for local keyboard control.

    The host application should call update_keys() each frame with pressed keys.
    """

    def __init__(
        self,
        power_step: float = 0.1,
        theta_step_deg: float = 5.0,
        phi_step_deg: float = 8.0,
        ballast_step: float = 0.1,
        initial_command: Optional[ControlCommand] = None,
    ):
        super().__init__(initial_command=initial_command)
        self.power_step = float(power_step)
        self.theta_step = float(np.radians(theta_step_deg))
        self.phi_step = float(np.radians(phi_step_deg))
        self.ballast_step = float(ballast_step)

    def update_keys(self, pressed_keys: Iterable[str]) -> ControlCommand:
        keys = {str(k).lower() for k in pressed_keys}
        current = self.read(None, 0.0)

        power = current.thruster_power
        power2 = current.thruster2_power if current.thruster2_power is not None else current.thruster_power
        theta = current.thruster_theta
        phi = current.thruster_phi
        ballast = current.ballast_cmd

        if "w" in keys:
            power += self.power_step
        if "s" in keys:
            power -= self.power_step
        if "e" in keys:
            power2 += self.power_step
        if "d" in keys:
            power2 -= self.power_step
        if "i" in keys:
            theta += self.theta_step
        if "k" in keys:
            theta -= self.theta_step
        if "j" in keys:
            phi -= self.phi_step
        if "l" in keys:
            phi += self.phi_step
        if "u" in keys:
            ballast += self.ballast_step
        if "o" in keys:
            ballast -= self.ballast_step
        if "space" in keys:
            power = 0.0
            power2 = 0.0
            theta = 0.0
            phi = 0.0
            ballast = 0.0

        cmd = ControlCommand(
            thruster_power=power,
            thruster_theta=theta,
            thruster_phi=phi,
            ballast_cmd=ballast,
            thruster2_power=power2,
            thruster2_theta=theta,
            thruster2_phi=phi,
        ).clip()
        self.push(cmd)
        return cmd


class JoystickManualCommandSource(QueuedManualCommandSource):
    """Joystick adapter for axes/buttons data emitted by any frontend/input API."""

    def __init__(self, initial_command: Optional[ControlCommand] = None):
        super().__init__(initial_command=initial_command)

    def update_axes(self, axes: Mapping[str, float] | Sequence[float]) -> ControlCommand:
        if isinstance(axes, Mapping):
            throttle = _to_float(axes.get("throttle", axes.get("surge", 0.0)))
            throttle2 = _to_float(axes.get("throttle2", axes.get("surge2", throttle)))
            theta = _to_float(axes.get("theta", axes.get("pitch", 0.0)))
            phi = _to_float(axes.get("phi", axes.get("yaw", 0.0)))
            ballast = _to_float(axes.get("ballast", axes.get("heave", 0.0)))
            deg_mode = bool(axes.get("angles_deg", False))
        else:
            values = list(axes)
            throttle = _to_float(values[0], 0.0) if len(values) > 0 else 0.0
            theta = _to_float(values[1], 0.0) if len(values) > 1 else 0.0
            phi = _to_float(values[2], 0.0) if len(values) > 2 else 0.0
            ballast = _to_float(values[3], 0.0) if len(values) > 3 else 0.0
            throttle2 = _to_float(values[4], throttle) if len(values) > 4 else throttle
            deg_mode = False

        if deg_mode:
            theta = float(np.radians(theta))
            phi = float(np.radians(phi))

        cmd = ControlCommand(
            thruster_power=throttle,
            thruster_theta=theta,
            thruster_phi=phi,
            ballast_cmd=ballast,
            thruster2_power=throttle2,
            thruster2_theta=theta,
            thruster2_phi=phi,
        ).clip()
        self.push(cmd)
        return cmd


class UDPManualCommandSource(QueuedManualCommandSource):
    """Receives JSON commands over UDP and keeps the latest command."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 14570,
        initial_command: Optional[ControlCommand] = None,
        buffer_size: int = 4096,
        timeout_s: float = 0.1,
        autostart: bool = True,
    ):
        super().__init__(initial_command=initial_command)
        self.host = str(host)
        self.port = int(port)
        self.buffer_size = int(buffer_size)
        self.timeout_s = float(timeout_s)
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if autostart:
            self.start()

    @property
    def local_port(self) -> int:
        return int(self.port)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self.host, self.port))
        self.port = int(self._sock.getsockname()[1])
        self._sock.settimeout(self.timeout_s)
        self._stop.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None

    def _recv_loop(self) -> None:
        while not self._stop.is_set() and self._sock is not None:
            try:
                raw, _addr = self._sock.recvfrom(self.buffer_size)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                payload = json.loads(raw.decode("utf-8"))
                if isinstance(payload, Mapping):
                    body = payload.get("command", payload)
                    current = self.read(None, 0.0)
                    cmd = _build_command(body, fallback=current)
                    self.push(cmd)
            except Exception:
                continue


@dataclass
class ReplayFrame:
    time_s: float
    command: ControlCommand


class ReplayManualCommandSource(ManualCommandSource):
    """Offline playback source for deterministic manual command replay."""

    def __init__(
        self,
        frames: Sequence[ReplayFrame],
        use_time: bool = True,
        hold_last: bool = True,
        speed: float = 1.0,
    ):
        self.frames = sorted(list(frames), key=lambda f: f.time_s)
        self.use_time = bool(use_time)
        self.hold_last = bool(hold_last)
        self.speed = max(1e-6, float(speed))
        self._index = 0
        self._step_reads = 0

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, object]],
        use_time: bool = True,
        hold_last: bool = True,
        speed: float = 1.0,
    ) -> "ReplayManualCommandSource":
        frames: List[ReplayFrame] = []
        current = ControlCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for idx, record in enumerate(records):
            payload = record.get("command", record)
            if not isinstance(payload, Mapping):
                continue
            t = _to_float(record.get("time_s", idx), float(idx))
            current = _build_command(payload, fallback=current)
            frames.append(ReplayFrame(time_s=t, command=current))
        return cls(frames=frames, use_time=use_time, hold_last=hold_last, speed=speed)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        use_time: bool = True,
        hold_last: bool = True,
        speed: float = 1.0,
    ) -> "ReplayManualCommandSource":
        rows: List[Dict[str, object]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return cls.from_records(rows, use_time=use_time, hold_last=hold_last, speed=speed)

    def read(self, ekf_state, time_s: float) -> Optional[ControlCommand]:
        if not self.frames:
            return ControlCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        if self.use_time:
            scaled_t = float(time_s) * self.speed
            while self._index + 1 < len(self.frames) and self.frames[self._index + 1].time_s <= scaled_t:
                self._index += 1
        else:
            self._index = min(self._step_reads, len(self.frames) - 1)
            self._step_reads += 1

        if self._index >= len(self.frames):
            if self.hold_last:
                return self.frames[-1].command
            return ControlCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return self.frames[self._index].command


def build_manual_source(mode: str, config: Optional[Mapping[str, object]] = None) -> ManualCommandSource:
    """Factory helper for MissionEngine config wiring."""
    cfg = dict(config or {})
    name = str(mode or "").strip().lower()

    if name in ("keyboard", "keys"):
        return KeyboardManualCommandSource(
            power_step=_to_float(cfg.get("power_step", 0.1), 0.1),
            theta_step_deg=_to_float(cfg.get("theta_step_deg", 5.0), 5.0),
            phi_step_deg=_to_float(cfg.get("phi_step_deg", 8.0), 8.0),
            ballast_step=_to_float(cfg.get("ballast_step", 0.1), 0.1),
        )

    if name in ("joystick", "gamepad"):
        return JoystickManualCommandSource()

    if name in ("udp", "network", "teleop"):
        return UDPManualCommandSource(
            host=str(cfg.get("host", "127.0.0.1")),
            port=int(cfg.get("port", 14570)),
            timeout_s=_to_float(cfg.get("timeout_s", 0.1), 0.1),
            autostart=bool(cfg.get("autostart", True)),
        )

    if name in ("replay", "file"):
        if "records" in cfg and isinstance(cfg["records"], Sequence):
            return ReplayManualCommandSource.from_records(
                cfg["records"],
                use_time=bool(cfg.get("use_time", True)),
                hold_last=bool(cfg.get("hold_last", True)),
                speed=_to_float(cfg.get("speed", 1.0), 1.0),
            )
        path = cfg.get("path")
        if not path:
            raise ValueError("Replay manual source requires 'path' or 'records'.")
        return ReplayManualCommandSource.from_jsonl(
            path,
            use_time=bool(cfg.get("use_time", True)),
            hold_last=bool(cfg.get("hold_last", True)),
            speed=_to_float(cfg.get("speed", 1.0), 1.0),
        )

    raise ValueError(f"Unsupported manual source mode: {mode}")
