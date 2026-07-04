"""Manual command sources for independent teleoperation and replay.

These sources are transport/input adapters that emit ControlCommand objects.
They can be used with ControlEngine manual mode in both USV and UAV simulations.
"""

from __future__ import annotations

import ast
import json
import math
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


class _SafeMathExpression:
    """Validate and evaluate a restricted math expression over (t, dt, step)."""

    _ALLOWED_NODE_TYPES = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Constant,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
    }

    _ALLOWED_FUNCTIONS = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "sqrt": np.sqrt,
        "abs": abs,
        "min": min,
        "max": max,
        "clip": np.clip,
    }

    _ALLOWED_NAMES = {"t", "dt", "step", "pi", "e"}

    def __init__(self, expression: str):
        self.expression = str(expression)
        self._code = self._compile(self.expression)

    @classmethod
    def _compile(cls, expression: str):
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc}") from exc

        for node in ast.walk(parsed):
            if type(node) not in cls._ALLOWED_NODE_TYPES:
                raise ValueError(f"Unsupported expression node: {type(node).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only direct function calls are allowed")
                if node.func.id not in cls._ALLOWED_FUNCTIONS:
                    raise ValueError(f"Function not allowed: {node.func.id}")
            if isinstance(node, ast.Name):
                if node.id not in cls._ALLOWED_NAMES and node.id not in cls._ALLOWED_FUNCTIONS:
                    raise ValueError(f"Name not allowed: {node.id}")

        return compile(parsed, "<manual_expression>", "eval")

    def eval(self, t: float, dt: float, step: int) -> float:
        env = {
            "t": float(t),
            "dt": float(dt),
            "step": int(step),
            "pi": float(math.pi),
            "e": float(math.e),
        }
        env.update(self._ALLOWED_FUNCTIONS)
        value = eval(self._code, {"__builtins__": {}}, env)
        return float(value)


class ExpressionManualCommandSource(ManualCommandSource):
    """Computes thruster commands from safe math expressions per simulation tick."""

    _CHANNEL_DEFAULTS = {
        "thruster_power": "0.0",
        "thruster_theta": "0.0",
        "thruster_phi": "0.0",
        "ballast_cmd": "0.0",
        "thruster2_power": None,
        "thruster2_theta": None,
        "thruster2_phi": None,
    }

    def __init__(
        self,
        expressions: Mapping[str, object],
        max_abs_output: float = 1.0,
        theta_max_deg: float = 60.0,
        finite_fallback: float = 0.0,
    ):
        self._step = 0
        self._last_time: Optional[float] = None
        self.max_abs_output = float(max_abs_output)
        self.theta_max = float(np.radians(theta_max_deg))
        self.finite_fallback = float(finite_fallback)

        expr_map = dict(self._CHANNEL_DEFAULTS)
        for key, value in dict(expressions or {}).items():
            expr_map[str(key)] = value

        self._compiled: Dict[str, Optional[_SafeMathExpression]] = {}
        for key in self._CHANNEL_DEFAULTS:
            expr = expr_map.get(key)
            if expr is None:
                self._compiled[key] = None
            else:
                self._compiled[key] = _SafeMathExpression(str(expr))

    def _eval_channel(self, channel: str, t: float, dt: float, step: int, default: float) -> float:
        compiled = self._compiled.get(channel)
        if compiled is None:
            return float(default)
        try:
            value = float(compiled.eval(t=t, dt=dt, step=step))
        except Exception:
            value = self.finite_fallback
        if not np.isfinite(value):
            value = self.finite_fallback
        return float(value)

    def read(self, ekf_state, time_s: float) -> Optional[ControlCommand]:
        t = float(time_s)
        if self._last_time is None:
            dt = 0.0
        else:
            dt = max(0.0, t - self._last_time)

        power = np.clip(self._eval_channel("thruster_power", t, dt, self._step, 0.0), -self.max_abs_output, self.max_abs_output)
        theta = np.clip(self._eval_channel("thruster_theta", t, dt, self._step, 0.0), -self.theta_max, self.theta_max)
        phi = self._eval_channel("thruster_phi", t, dt, self._step, 0.0)
        ballast = np.clip(self._eval_channel("ballast_cmd", t, dt, self._step, 0.0), -1.0, 1.0)

        power2_default = power
        theta2_default = theta
        phi2_default = phi

        power2 = np.clip(
            self._eval_channel("thruster2_power", t, dt, self._step, power2_default),
            -self.max_abs_output,
            self.max_abs_output,
        )
        theta2 = np.clip(
            self._eval_channel("thruster2_theta", t, dt, self._step, theta2_default),
            -self.theta_max,
            self.theta_max,
        )
        phi2 = self._eval_channel("thruster2_phi", t, dt, self._step, phi2_default)

        self._step += 1
        self._last_time = t

        return ControlCommand(
            thruster_power=float(power),
            thruster_theta=float(theta),
            thruster_phi=float(phi),
            ballast_cmd=float(ballast),
            thruster2_power=float(power2),
            thruster2_theta=float(theta2),
            thruster2_phi=float(phi2),
        ).clip()


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

    if name in ("expression", "expr", "function"):
        expressions = cfg.get("expressions", cfg)
        if not isinstance(expressions, Mapping):
            raise ValueError("Expression manual source requires a mapping under 'expressions'.")
        return ExpressionManualCommandSource(
            expressions=expressions,
            max_abs_output=_to_float(cfg.get("max_abs_output", 1.0), 1.0),
            theta_max_deg=_to_float(cfg.get("theta_max_deg", 60.0), 60.0),
            finite_fallback=_to_float(cfg.get("finite_fallback", 0.0), 0.0),
        )

    raise ValueError(f"Unsupported manual source mode: {mode}")
