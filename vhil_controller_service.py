#!/usr/bin/env python3
"""vHIL controller microservice.

Runs a control loop at a fixed rate (default 100 Hz), consumes sensor frames
from UDP, computes commands, and publishes command frames over UDP.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import numpy as np

from hil_interface import HILProtocol
from sensor_engine import IMUReading, BarometerReading, ExtendedKalmanFilter
from geometry_engine import GeometryEngine
from physics_engine import PhysicsEngine
from vhil_runtime_metrics import OnlineStats, SequenceTracker

CMD_FMT_7F = "<7f"


@dataclass
class CommandFrame:
    thruster_power: float
    thruster_theta: float
    thruster_phi: float
    ballast_cmd: float
    thruster2_power: float
    thruster2_theta: float
    thruster2_phi: float

    def clip(self) -> "CommandFrame":
        return CommandFrame(
            thruster_power=float(np.clip(self.thruster_power, -1.0, 1.0)),
            thruster_theta=float(np.clip(self.thruster_theta, 0.0, np.radians(60.0))),
            thruster_phi=float(self.thruster_phi % (2.0 * np.pi)),
            ballast_cmd=float(np.clip(self.ballast_cmd, -1.0, 1.0)),
            thruster2_power=float(np.clip(self.thruster2_power, -1.0, 1.0)),
            thruster2_theta=float(np.clip(self.thruster2_theta, 0.0, np.radians(60.0))),
            thruster2_phi=float(self.thruster2_phi % (2.0 * np.pi)),
        )


class RuntimeState:
    def __init__(self, target_hz: float) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.last_tick_s = 0.0
        self.last_sensor_at = 0.0
        self.last_command_at = 0.0
        self.sensor_packets = 0
        self.command_packets = 0
        self.last_error: Optional[str] = None
        self.active_controller = "lqr"
        self.target_hz = float(target_hz)
        self.target_dt_ms = 1000.0 / max(1.0, self.target_hz)
        self.loop_period_ms = OnlineStats()
        self.loop_jitter_ms = OnlineStats()
        self.sensor_interarrival_ms = OnlineStats()
        self.closed_loop_rtt_ms = OnlineStats()
        self.sensor_sequence = SequenceTracker()
        self.last_loop_perf: Optional[float] = None
        self.last_sensor_perf: Optional[float] = None
        self.last_sensor_sequence: Optional[int] = None
        self.last_ack_command_sequence: Optional[int] = None
        self.last_command_sequence: Optional[int] = None

    def record_loop(self, perf_now: float) -> None:
        with self.lock:
            if self.last_loop_perf is not None:
                period_ms = (perf_now - self.last_loop_perf) * 1000.0
                self.loop_period_ms.update(period_ms)
                self.loop_jitter_ms.update(abs(period_ms - self.target_dt_ms))
            self.last_loop_perf = perf_now
            self.last_tick_s = time.time()

    def record_sensor(
        self,
        sequence: Optional[int],
        recv_wall: float,
        recv_perf: float,
        ack_command_sequence: Optional[int],
        ack_command_sent_at: float,
    ) -> None:
        with self.lock:
            self.sensor_packets += 1
            self.last_sensor_at = recv_wall
            self.sensor_sequence.update(sequence)
            self.last_sensor_sequence = sequence
            if self.last_sensor_perf is not None:
                self.sensor_interarrival_ms.update((recv_perf - self.last_sensor_perf) * 1000.0)
            self.last_sensor_perf = recv_perf
            if ack_command_sequence:
                self.last_ack_command_sequence = int(ack_command_sequence)
            if ack_command_sent_at > 0.0:
                self.closed_loop_rtt_ms.update(max(0.0, (recv_wall - ack_command_sent_at) * 1000.0))

    def record_command(self, sequence: int, sent_wall: float) -> None:
        with self.lock:
            self.command_packets += 1
            self.last_command_at = sent_wall
            self.last_command_sequence = int(sequence)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "uptime_s": max(0.0, time.time() - self.started_at),
                "last_tick_s": self.last_tick_s,
                "last_sensor_at": self.last_sensor_at,
                "last_command_at": self.last_command_at,
                "sensor_packets": self.sensor_packets,
                "command_packets": self.command_packets,
                "last_error": self.last_error,
                "active_controller": self.active_controller,
                "target_hz": self.target_hz,
                "metrics": {
                    "loop_period_ms": self.loop_period_ms.snapshot(),
                    "loop_jitter_ms": self.loop_jitter_ms.snapshot(),
                    "sensor_interarrival_ms": self.sensor_interarrival_ms.snapshot(),
                    "closed_loop_rtt_ms": self.closed_loop_rtt_ms.snapshot(),
                    "sensor_sequence": self.sensor_sequence.snapshot(),
                    "last_sensor_sequence": self.last_sensor_sequence,
                    "last_command_sequence": self.last_command_sequence,
                    "last_ack_command_sequence": self.last_ack_command_sequence,
                },
            }


def _health_server(state: RuntimeState, host: str, port: int) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            payload = {
                "service": "controller",
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


def _decode_sensor(packet: bytes) -> Optional[dict]:
    if not packet:
        return None

    try:
        # Fast path for compact binary frame.
        if len(packet) >= struct.calcsize(HILProtocol.SENSOR_FMT):
            return HILProtocol.deserialize_binary_sensor(packet)
    except Exception:
        pass

    try:
        return json.loads(packet.decode("utf-8"))
    except Exception:
        return None


def _encode_legacy_command(command: CommandFrame) -> bytes:
    cmd = command.clip()
    return struct.pack(CMD_FMT_7F, *(
        float(cmd.thruster_power),
        float(cmd.thruster_theta),
        float(cmd.thruster_phi),
        float(cmd.ballast_cmd),
        float(cmd.thruster2_power),
        float(cmd.thruster2_theta),
        float(cmd.thruster2_phi),
    ))


def _compute_lightweight_command(ekf_state, target_xyz: np.ndarray) -> CommandFrame:
    # Keep this service-level controller intentionally simple and deterministic.
    pos = np.asarray(ekf_state.position, dtype=float)
    vel = np.asarray(ekf_state.velocity_linear, dtype=float)
    err = target_xyz - pos

    forward = float(np.clip(0.18 * err[0] - 0.08 * vel[0], -0.45, 0.45))
    yaw_cmd = float(np.clip(0.10 * err[1] - 0.06 * vel[1], -0.18, 0.18))
    ballast = float(np.clip(0.22 * err[2] - 0.08 * vel[2], -0.4, 0.4))

    p1 = float(np.clip(forward - yaw_cmd, -1.0, 1.0))
    p2 = float(np.clip(forward + yaw_cmd, -1.0, 1.0))
    return CommandFrame(
        thruster_power=p1,
        thruster_theta=0.0,
        thruster_phi=0.0,
        ballast_cmd=ballast,
        thruster2_power=p2,
        thruster2_theta=0.0,
        thruster2_phi=0.0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the controller vHIL microservice")
    parser.add_argument("--hz", type=float, default=float(os.getenv("VHIL_RATE_HZ", "100")))
    parser.add_argument("--duration", type=float, default=0.0, help="0 runs forever")
    parser.add_argument("--sensor-bind-host", type=str, default=os.getenv("CONTROLLER_SENSOR_BIND_HOST", "0.0.0.0"))
    parser.add_argument("--sensor-bind-port", type=int, default=int(os.getenv("CONTROLLER_SENSOR_PORT", "9001")))
    parser.add_argument("--cmd-target-host", type=str, default=os.getenv("CONTROLLER_COMMAND_TARGET_HOST", "physim"))
    parser.add_argument("--cmd-target-port", type=int, default=int(os.getenv("CONTROLLER_COMMAND_TARGET_PORT", "9000")))
    parser.add_argument("--controller", type=str, default=os.getenv("CONTROLLER_TYPE", "service_pd"))
    parser.add_argument("--ref-x", type=float, default=float(os.getenv("CONTROLLER_REF_X", "0.0")))
    parser.add_argument("--ref-y", type=float, default=float(os.getenv("CONTROLLER_REF_Y", "0.0")))
    parser.add_argument("--ref-z", type=float, default=float(os.getenv("CONTROLLER_REF_Z", "2.0")))
    parser.add_argument("--health-host", type=str, default=os.getenv("CONTROLLER_HEALTH_HOST", "0.0.0.0"))
    parser.add_argument("--health-port", type=int, default=int(os.getenv("CONTROLLER_HEALTH_PORT", "8082")))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hz = max(1.0, float(args.hz))
    dt = 1.0 / hz

    state = RuntimeState(hz)
    health = _health_server(state, args.health_host, args.health_port)
    health_thread = threading.Thread(target=health.serve_forever, daemon=True)
    health_thread.start()

    # Shadow model + EKF keep perception realistic in this isolated controller service.
    shadow_physics = PhysicsEngine(GeometryEngine(L=0.8, D=0.1), max_thruster_force=10.0)
    ekf = ExtendedKalmanFilter(shadow_physics)
    requested_controller = str(args.controller).strip().lower() or "service_pd"
    target = np.array([float(args.ref_x), float(args.ref_y), float(args.ref_z)], dtype=float)
    with state.lock:
        state.active_controller = requested_controller

    sensor_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sensor_sock.bind((args.sensor_bind_host, int(args.sensor_bind_port)))
    sensor_sock.setblocking(False)

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_target = (str(args.cmd_target_host), int(args.cmd_target_port))

    last_command = CommandFrame(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    command_sequence = 0
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

            latest_sensor = None
            while True:
                try:
                    packet, _addr = sensor_sock.recvfrom(4096)
                except BlockingIOError:
                    break
                except Exception as exc:
                    with state.lock:
                        state.last_error = str(exc)
                    break

                decoded = _decode_sensor(packet)
                if decoded is not None:
                    latest_sensor = decoded
                    recv_wall = time.time()
                    state.record_sensor(
                        decoded.get("sequence") if isinstance(decoded, dict) else None,
                        recv_wall,
                        time.perf_counter(),
                        decoded.get("ack_command_sequence") if isinstance(decoded, dict) else None,
                        float(decoded.get("ack_command_sent_at", 0.0)) if isinstance(decoded, dict) else 0.0,
                    )

            if latest_sensor is not None:
                imu_obj = latest_sensor.get("imu", {}) if isinstance(latest_sensor, dict) else {}
                baro_obj = latest_sensor.get("barometer", {}) if isinstance(latest_sensor, dict) else {}
                ts = float(latest_sensor.get("timestamp", time.time())) if isinstance(latest_sensor, dict) else time.time()

                accel = np.asarray(imu_obj.get("accel", [0.0, 0.0, 9.81]), dtype=float)
                gyro = np.asarray(imu_obj.get("gyro", [0.0, 0.0, 0.0]), dtype=float)
                depth = float(baro_obj.get("depth", 0.0))

                ekf.predict(dt)
                ekf.update_imu(IMUReading(accel=accel, gyro=gyro, timestamp=ts))
                ekf.update_barometer(BarometerReading(depth=depth, pressure=depth * 100.0, timestamp=ts))

                try:
                    last_command = _compute_lightweight_command(ekf.state_estimate, target)
                except Exception as exc:
                    with state.lock:
                        state.last_error = str(exc)

            # Keep the internal model in sync with actuation history.
            shadow_physics.step(
                thruster_power=last_command.thruster_power,
                thruster_theta=last_command.thruster_theta,
                thruster_phi=last_command.thruster_phi,
                ballast_cmd=last_command.ballast_cmd,
                thruster2_power=last_command.thruster2_power,
                thruster2_theta=last_command.thruster2_theta,
                thruster2_phi=last_command.thruster2_phi,
                dt=dt,
            )

            try:
                command_sequence += 1
                sent_wall = time.time()
                cmd_sock.sendto(HILProtocol.serialize_vhil_command(last_command, command_sequence, sent_wall), cmd_target)
                state.record_command(command_sequence, sent_wall)
            except Exception as exc:
                with state.lock:
                    state.last_error = str(exc)
    finally:
        health.shutdown()
        health.server_close()
        sensor_sock.close()
        cmd_sock.close()

    print(json.dumps({"service": "controller", "ok": True, "state": state.snapshot()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
