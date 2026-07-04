#!/usr/bin/env python3
"""vHIL physics microservice.

Runs the physical simulator loop at a fixed rate (default 100 Hz), receives
control commands over UDP, and publishes sensor frames over UDP.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine
from sensor_engine import Environment, SensorEngine
from hil_interface import HILProtocol
from vhil_runtime_metrics import OnlineStats, SequenceTracker

CMD_FMT_7F = "<7f"
CMD_FMT_4F = "<4f"


@dataclass
class ControlCommand:
    thruster_power: float
    thruster_theta: float
    thruster_phi: float
    ballast_cmd: float
    thruster2_power: Optional[float] = None
    thruster2_theta: Optional[float] = None
    thruster2_phi: Optional[float] = None

    def clip(self) -> "ControlCommand":
        t2_power = self.thruster_power if self.thruster2_power is None else self.thruster2_power
        t2_theta = self.thruster_theta if self.thruster2_theta is None else self.thruster2_theta
        t2_phi = self.thruster_phi if self.thruster2_phi is None else self.thruster2_phi
        max_theta = math.radians(60.0)
        return ControlCommand(
            thruster_power=float(max(-1.0, min(1.0, self.thruster_power))),
            thruster_theta=float(max(0.0, min(max_theta, self.thruster_theta))),
            thruster_phi=float(self.thruster_phi % (2.0 * math.pi)),
            ballast_cmd=float(max(-1.0, min(1.0, self.ballast_cmd))),
            thruster2_power=float(max(-1.0, min(1.0, t2_power))),
            thruster2_theta=float(max(0.0, min(max_theta, t2_theta))),
            thruster2_phi=float(t2_phi % (2.0 * math.pi)),
        )


class RuntimeState:
    def __init__(self, target_hz: float) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.last_tick_s = 0.0
        self.last_command_at = 0.0
        self.last_sensor_at = 0.0
        self.command_packets = 0
        self.sensor_packets = 0
        self.last_error: Optional[str] = None
        self.target_hz = float(target_hz)
        self.target_dt_ms = 1000.0 / max(1.0, self.target_hz)
        self.loop_period_ms = OnlineStats()
        self.loop_jitter_ms = OnlineStats()
        self.command_interarrival_ms = OnlineStats()
        self.command_sequence = SequenceTracker()
        self.last_loop_perf: Optional[float] = None
        self.last_command_perf: Optional[float] = None
        self.last_command_sequence: Optional[int] = None
        self.last_sensor_sequence: Optional[int] = None

    def record_loop(self, perf_now: float) -> None:
        with self.lock:
            if self.last_loop_perf is not None:
                period_ms = (perf_now - self.last_loop_perf) * 1000.0
                self.loop_period_ms.update(period_ms)
                self.loop_jitter_ms.update(abs(period_ms - self.target_dt_ms))
            self.last_loop_perf = perf_now
            self.last_tick_s = time.time()

    def record_command(self, sequence: Optional[int], recv_wall: float, recv_perf: float) -> None:
        with self.lock:
            self.command_packets += 1
            self.last_command_at = recv_wall
            self.command_sequence.update(sequence)
            self.last_command_sequence = sequence
            if self.last_command_perf is not None:
                self.command_interarrival_ms.update((recv_perf - self.last_command_perf) * 1000.0)
            self.last_command_perf = recv_perf

    def record_sensor(self, sequence: int, sent_wall: float) -> None:
        with self.lock:
            self.sensor_packets += 1
            self.last_sensor_at = sent_wall
            self.last_sensor_sequence = int(sequence)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "uptime_s": max(0.0, time.time() - self.started_at),
                "last_tick_s": self.last_tick_s,
                "last_command_at": self.last_command_at,
                "last_sensor_at": self.last_sensor_at,
                "command_packets": self.command_packets,
                "sensor_packets": self.sensor_packets,
                "last_error": self.last_error,
                "target_hz": self.target_hz,
                "metrics": {
                    "loop_period_ms": self.loop_period_ms.snapshot(),
                    "loop_jitter_ms": self.loop_jitter_ms.snapshot(),
                    "command_interarrival_ms": self.command_interarrival_ms.snapshot(),
                    "command_sequence": self.command_sequence.snapshot(),
                    "last_command_sequence": self.last_command_sequence,
                    "last_sensor_sequence": self.last_sensor_sequence,
                },
            }


def _command_from_dict(obj: dict) -> ControlCommand:
    return ControlCommand(
        thruster_power=float(obj.get("thruster_power", 0.0)),
        thruster_theta=float(obj.get("thruster_theta", 0.0)),
        thruster_phi=float(obj.get("thruster_phi", 0.0)),
        ballast_cmd=float(obj.get("ballast_cmd", 0.0)),
        thruster2_power=float(obj.get("thruster2_power", obj.get("thruster_power", 0.0))),
        thruster2_theta=float(obj.get("thruster2_theta", obj.get("thruster_theta", 0.0))),
        thruster2_phi=float(obj.get("thruster2_phi", obj.get("thruster_phi", 0.0))),
    ).clip()


def _decode_command(packet: bytes) -> tuple[Optional[ControlCommand], dict]:
    if not packet:
        return None, {}

    try:
        if HILProtocol.is_vhil_frame(packet, HILProtocol.VHIL_FRAME_COMMAND):
            frame = HILProtocol.deserialize_vhil_command(packet)
            return _command_from_dict(frame["cmd"]), frame["meta"]

        if len(packet) >= struct.calcsize(CMD_FMT_7F):
            vals = struct.unpack(CMD_FMT_7F, packet[: struct.calcsize(CMD_FMT_7F)])
            return ControlCommand(
                thruster_power=float(vals[0]),
                thruster_theta=float(vals[1]),
                thruster_phi=float(vals[2]),
                ballast_cmd=float(vals[3]),
                thruster2_power=float(vals[4]),
                thruster2_theta=float(vals[5]),
                thruster2_phi=float(vals[6]),
            ).clip(), {}

        if len(packet) >= struct.calcsize(CMD_FMT_4F):
            vals = struct.unpack(CMD_FMT_4F, packet[: struct.calcsize(CMD_FMT_4F)])
            return ControlCommand(
                thruster_power=float(vals[0]),
                thruster_theta=float(vals[1]),
                thruster_phi=float(vals[2]),
                ballast_cmd=float(vals[3]),
            ).clip(), {}
    except Exception:
        pass

    try:
        obj = json.loads(packet.decode("utf-8"))
        if isinstance(obj, dict) and "cmd" in obj and isinstance(obj["cmd"], dict):
            obj = obj["cmd"]
        if not isinstance(obj, dict):
            return None, {}

        theta = float(obj.get("thruster_theta", 0.0))
        phi = float(obj.get("thruster_phi", 0.0))
        theta2 = float(obj.get("thruster2_theta", theta))
        phi2 = float(obj.get("thruster2_phi", phi))

        # JSON path in this repo usually uses degrees; convert to radians.
        to_rad = 3.141592653589793 / 180.0
        return ControlCommand(
            thruster_power=float(obj.get("thruster_power", 0.0)),
            thruster_theta=theta * to_rad,
            thruster_phi=phi * to_rad,
            ballast_cmd=float(obj.get("ballast_cmd", 0.0)),
            thruster2_power=float(obj.get("thruster2_power", obj.get("thruster_power", 0.0))),
            thruster2_theta=theta2 * to_rad,
            thruster2_phi=phi2 * to_rad,
        ).clip(), {}
    except Exception:
        return None, {}


def _health_server(state: RuntimeState, host: str, port: int) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            payload = {
                "service": "physim",
                "ok": True,
                "state": state.snapshot(),
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            return

    return HTTPServer((host, port), Handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the physics vHIL microservice")
    parser.add_argument("--hz", type=float, default=float(os.getenv("VHIL_RATE_HZ", "100")))
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs forever")
    parser.add_argument("--cmd-bind-host", type=str, default=os.getenv("PHYSIM_COMMAND_BIND_HOST", "0.0.0.0"))
    parser.add_argument("--cmd-bind-port", type=int, default=int(os.getenv("PHYSIM_COMMAND_PORT", "9000")))
    parser.add_argument("--sensor-target-host", type=str, default=os.getenv("PHYSIM_SENSOR_TARGET_HOST", "controller"))
    parser.add_argument("--sensor-target-port", type=int, default=int(os.getenv("PHYSIM_SENSOR_TARGET_PORT", "9001")))
    parser.add_argument("--pool-depth", type=float, default=float(os.getenv("PHYSIM_POOL_DEPTH", "10.0")))
    parser.add_argument("--pool-radius", type=float, default=float(os.getenv("PHYSIM_POOL_RADIUS", "50.0")))
    parser.add_argument("--max-thruster-force", type=float, default=float(os.getenv("PHYSIM_MAX_THRUSTER_FORCE", "10.0")))
    parser.add_argument("--health-host", type=str, default=os.getenv("PHYSIM_HEALTH_HOST", "0.0.0.0"))
    parser.add_argument("--health-port", type=int, default=int(os.getenv("PHYSIM_HEALTH_PORT", "8081")))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hz = max(1.0, float(args.hz))
    dt = 1.0 / hz

    state = RuntimeState(hz)
    health = _health_server(state, args.health_host, args.health_port)
    health_thread = threading.Thread(target=health.serve_forever, daemon=True)
    health_thread.start()

    geo = GeometryEngine(L=0.8, D=0.1)
    physics = PhysicsEngine(geo, max_thruster_force=float(args.max_thruster_force))
    sensors = SensorEngine(Environment(pool_depth=float(args.pool_depth), pool_radius=float(args.pool_radius)), noise_scale=1.0)

    last_command = ControlCommand(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    last_command_sequence = 0
    last_command_sent_at = 0.0
    sensor_sequence = 0

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.bind((args.cmd_bind_host, int(args.cmd_bind_port)))
    cmd_sock.setblocking(False)

    sensor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sensor_target = (str(args.sensor_target_host), int(args.sensor_target_port))

    started = time.perf_counter()
    next_tick = started

    try:
        while True:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += dt
            state.record_loop(time.perf_counter())

            if args.duration > 0.0 and (time.perf_counter() - started) >= float(args.duration):
                break

            while True:
                try:
                    packet, _addr = cmd_sock.recvfrom(4096)
                except BlockingIOError:
                    break
                except Exception as exc:
                    with state.lock:
                        state.last_error = str(exc)
                    break

                decoded, meta = _decode_command(packet)
                if decoded is not None:
                    last_command = decoded
                    last_command_sequence = int(meta.get("sequence", last_command_sequence))
                    last_command_sent_at = float(meta.get("sent_at", last_command_sent_at))
                    state.record_command(
                        int(meta["sequence"]) if "sequence" in meta else None,
                        time.time(),
                        time.perf_counter(),
                    )

            physics.step(**last_command.__dict__, dt=dt)
            bundle = sensors.read(physics.state, physics.time)
            sensor_payload = {
                "timestamp": bundle.timestamp,
                "imu": {
                    "accel": [float(x) for x in bundle.imu.accel],
                    "gyro": [float(x) for x in bundle.imu.gyro],
                },
                "barometer": {
                    "depth": float(bundle.barometer.depth),
                },
            }
            sensor_sequence += 1
            sent_wall = time.time()
            data = HILProtocol.serialize_vhil_sensor(
                sensor_payload,
                sensor_sequence,
                sent_wall,
                ack_command_sequence=last_command_sequence,
                ack_command_sent_at=last_command_sent_at,
            )
            try:
                sensor_sock.sendto(data, sensor_target)
                state.record_sensor(sensor_sequence, sent_wall)
            except Exception as exc:
                with state.lock:
                    state.last_error = str(exc)
    finally:
        health.shutdown()
        health.server_close()
        cmd_sock.close()
        sensor_sock.close()

    print(json.dumps({"service": "physim", "ok": True, "state": state.snapshot()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
