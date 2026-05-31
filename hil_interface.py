"""HIL protocol and bridge implementations.

Provides JSON and compact binary (struct) encodings for commands and
sensor frames, a `MockLoopbackBridge` for unit tests, and a UDP-based
`UDPHILBridge` suitable for RPi/embedded clients.

Binary sensor frame format (little-endian, floats):
    [timestamp, ax, ay, az, p, q, r, depth]  -> 8 floats (32 bytes)

Binary command frame format (little-endian, floats):
    [thruster_power, thruster_theta_deg, thruster_phi_deg, ballast] -> 4 floats
"""
from dataclasses import asdict
import json
import struct
import threading
import time
from typing import Optional, Tuple, Dict

import socket
import numpy as np

from control_engine import ControlCommand


class HILProtocol:
    """Serialize / deserialize control and sensor frames in JSON and binary."""

    CMD_FMT = '<ffff'   # 4 floats
    SENSOR_FMT = '<8f'  # timestamp, ax,ay,az, p,q,r, depth

    @staticmethod
    def serialize_json_command(cmd: ControlCommand) -> bytes:
        return json.dumps(cmd.to_dict()).encode('utf-8')

    @staticmethod
    def deserialize_json_command(payload: bytes) -> dict:
        return json.loads(payload.decode('utf-8'))

    @staticmethod
    def serialize_binary_command(cmd: ControlCommand) -> bytes:
        t = cmd.to_dict()
        return struct.pack(HILProtocol.CMD_FMT,
                           float(t['thruster_power']),
                           float(t['thruster_theta']),
                           float(t['thruster_phi']),
                           float(t['ballast_cmd']))

    @staticmethod
    def deserialize_binary_command(payload: bytes) -> dict:
        size = struct.calcsize(HILProtocol.CMD_FMT)
        vals = struct.unpack(HILProtocol.CMD_FMT, payload[:size])
        return {
            'thruster_power': vals[0],
            'thruster_theta': vals[1],
            'thruster_phi':   vals[2],
            'ballast_cmd':    vals[3],
        }

    @staticmethod
    def serialize_binary_sensor(sensor: Dict) -> bytes:
        # sensor: dict with keys timestamp, imu: {accel, gyro}, barometer:{depth}
        ts = float(sensor.get('timestamp', time.time()))
        imu = sensor.get('imu', {})
        accel = imu.get('accel', [0.0, 0.0, 0.0])[:3]
        gyro  = imu.get('gyro', [0.0, 0.0, 0.0])[:3]
        depth = float(sensor.get('barometer', {}).get('depth', 0.0))
        vals = (ts, float(accel[0]), float(accel[1]), float(accel[2]),
                float(gyro[0]), float(gyro[1]), float(gyro[2]), float(depth))
        return struct.pack(HILProtocol.SENSOR_FMT, *vals)

    @staticmethod
    def deserialize_binary_sensor(payload: bytes) -> dict:
        size = struct.calcsize(HILProtocol.SENSOR_FMT)
        if len(payload) < size:
            raise ValueError('payload too short for binary sensor frame')
        vals = struct.unpack(HILProtocol.SENSOR_FMT, payload[:size])
        return {
            'timestamp': float(vals[0]),
            'imu': {
                'accel': [float(vals[1]), float(vals[2]), float(vals[3])],
                'gyro':  [float(vals[4]), float(vals[5]), float(vals[6])],
                'timestamp': float(vals[0]),
            },
            'barometer': {
                'depth': float(vals[7]),
                'pressure': float(vals[7]) * 100.0,  # placeholder
                'timestamp': float(vals[0]),
            }
        }


class MockLoopbackBridge:
    """Simple in-process loopback bridge for deterministic HIL testing."""

    def __init__(self, latency_s: float = 0.001):
        self.latency_s = latency_s
        self._lock = threading.Lock()
        self._last_command = None
        self._last_sensor = None

    def send_command(self, payload: bytes, mode: str = 'json') -> None:
        with self._lock:
            if mode == 'json':
                try:
                    self._last_command = HILProtocol.deserialize_json_command(payload)
                except Exception:
                    self._last_command = None
            else:
                try:
                    self._last_command = HILProtocol.deserialize_binary_command(payload)
                except Exception:
                    self._last_command = None

    def recv_sensors(self, timeout: Optional[float] = None) -> dict:
        time.sleep(self.latency_s)
        with self._lock:
            if self._last_command is None:
                return {}
            # echo back command as a minimal sensor dict
            return {'timestamp': time.time(), 'echo_command': self._last_command}


class UDPHILBridge:
    """UDP server-side HIL bridge.

    Listens for incoming packets from hardware. When `send_command` is
    called, it will send to the last known hardware address (learned from
    incoming packets). `recv_sensors` returns the last sensor packet
    received from hardware, blocking until available or timeout.
    """

    def __init__(self, bind_addr: Tuple[str, int] = ('0.0.0.0', 5005)):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(bind_addr)
        self.sock.settimeout(0.1)
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._stop = threading.Event()
        self._last_addr: Optional[Tuple[str, int]] = None
        self._last_sensor: Optional[dict] = None
        self._seq = 0
        self._last_ack_seq: Optional[int] = None
        self._ack_event = threading.Event()
        self._lock = threading.Lock()
        self._listener_thread.start()

    def _listen_loop(self):
        while not self._stop.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception:
                break

            # learn addr
            with self._lock:
                self._last_addr = addr
                # detect JSON
                if data and data[0] in (123, 91):  # '{' or '['
                    try:
                        parsed = json.loads(data.decode('utf-8'))
                        # ack frame handling
                        if isinstance(parsed, dict) and parsed.get('ack'):
                            try:
                                self._last_ack_seq = int(parsed.get('seq'))
                                self._ack_event.set()
                            except Exception:
                                pass
                        else:
                            self._last_sensor = parsed
                        continue
                    except Exception:
                        pass
                # try binary sensor
                try:
                    parsed = HILProtocol.deserialize_binary_sensor(data)
                    self._last_sensor = parsed
                except Exception:
                    # unknown packet — ignore
                    continue

    def send_command(self, payload: bytes, mode: str = 'json') -> None:
        # default send without ACK
        with self._lock:
            if self._last_addr is None:
                return
            try:
                self.sock.sendto(payload, self._last_addr)
            except Exception:
                pass

    def send_command_with_ack(self, payload: bytes, mode: str = 'json', ack_timeout: float = 0.2) -> bool:
        """Send a command and wait for an ACK from the hardware client.

        Only JSON-mode ack is supported: the bridge will wrap the provided
        JSON command in an envelope `{'seq': N, 'cmd': <original>}` before sending.
        Returns True if ACK with matching seq received within timeout.
        """
        with self._lock:
            if self._last_addr is None:
                return False
            # only support JSON ack wrapping
            if mode != 'json':
                try:
                    self.sock.sendto(payload, self._last_addr)
                except Exception:
                    pass
                return False

            # prepare envelope
            try:
                cmd_obj = json.loads(payload.decode('utf-8'))
            except Exception:
                return False

            self._seq += 1
            seq = self._seq
            envelope = {'seq': seq, 'cmd': cmd_obj}
            try:
                self._ack_event.clear()
                self.sock.sendto(json.dumps(envelope).encode('utf-8'), self._last_addr)
            except Exception:
                return False

        # wait for ack
        waited = self._ack_event.wait(timeout=ack_timeout)
        if not waited:
            return False
        with self._lock:
            ok = (self._last_ack_seq == seq)
            return ok

    def recv_sensors(self, timeout: Optional[float] = None) -> Optional[dict]:
        wait_until = time.time() + (timeout or 0.0)
        while True:
            with self._lock:
                if self._last_sensor is not None:
                    s = self._last_sensor
                    # clear after read
                    self._last_sensor = None
                    return s
            if timeout is None or timeout <= 0.0:
                return None
            if time.time() >= wait_until:
                return None
            time.sleep(0.001)

    def close(self):
        self._stop.set()
        try:
            self.sock.close()
        except Exception:
            pass


class SerialHILBridge:
    """Serial-based HIL bridge.

    Supports an injectable `serial_inst` (object with `read`, `readline`,
    `write`, `in_waiting`) for tests. If `serial_inst` is None, attempts to
    open using `serial.Serial(port, baudrate, timeout)` (pyserial).

    Protocol over serial:
      - JSON frames delimited by newline (`\n`) for readability
      - Binary sensor frames accepted as fixed-size structs (no delimiter)
      - ACK for send_command_with_ack: hardware should send JSON `{'ack':True,'seq':N}`\n
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, serial_inst: object = None):
        self._provided = serial_inst is not None
        self.serial = serial_inst
        self.port = port
        self.baudrate = baudrate
        self._stop = threading.Event()
        self._listener = threading.Thread(target=self._reader_loop, daemon=True)
        self._lock = threading.Lock()
        self._last_sensor = None
        self._seq = 0
        self._last_ack_seq = None
        self._ack_event = threading.Event()

        if not self._provided:
            try:
                import serial as _serial
                if port is None:
                    raise ValueError('port must be provided when not injecting serial_inst')
                self.serial = _serial.Serial(port, baudrate=baudrate, timeout=0.1)
            except Exception:
                raise

        self._listener.start()

    def _reader_loop(self):
        # read lines (JSON) or fixed-size binary frames
        sensor_size = struct.calcsize(HILProtocol.SENSOR_FMT)
        while not self._stop.is_set():
            try:
                # prefer readline for JSON frames
                if hasattr(self.serial, 'readline'):
                    line = self.serial.readline()
                    if not line:
                        time.sleep(0.01)
                        continue
                    # try json
                    try:
                        parsed = json.loads(line.decode('utf-8'))
                        if isinstance(parsed, dict) and parsed.get('ack'):
                            try:
                                self._last_ack_seq = int(parsed.get('seq'))
                                self._ack_event.set()
                            except Exception:
                                pass
                        else:
                            self._last_sensor = parsed
                        continue
                    except Exception:
                        # not json — maybe binary packed without newline
                        data = line
                else:
                    # no readline -> read fixed size and attempt binary
                    data = self.serial.read(sensor_size)
                    if not data:
                        time.sleep(0.01)
                        continue

                # try binary parse
                try:
                    parsed = HILProtocol.deserialize_binary_sensor(data)
                    self._last_sensor = parsed
                except Exception:
                    # unknown format — ignore
                    continue
            except Exception:
                time.sleep(0.01)
                continue

    def send_command(self, payload: bytes, mode: str = 'json') -> None:
        # write raw payload; add newline for JSON
        with self._lock:
            try:
                if mode == 'json':
                    self.serial.write(payload + b'\n')
                else:
                    self.serial.write(payload)
            except Exception:
                pass

    def send_command_with_ack(self, payload: bytes, mode: str = 'json', ack_timeout: float = 0.2) -> bool:
        with self._lock:
            if mode != 'json':
                try:
                    self.serial.write(payload)
                except Exception:
                    pass
                return False

            try:
                cmd_obj = json.loads(payload.decode('utf-8'))
            except Exception:
                return False

            self._seq += 1
            seq = self._seq
            envelope = {'seq': seq, 'cmd': cmd_obj}
            try:
                self._ack_event.clear()
                self.serial.write(json.dumps(envelope).encode('utf-8') + b'\n')
            except Exception:
                return False

        waited = self._ack_event.wait(timeout=ack_timeout)
        if not waited:
            return False
        with self._lock:
            return (self._last_ack_seq == seq)

    def recv_sensors(self, timeout: Optional[float] = None) -> Optional[dict]:
        wait_until = time.time() + (timeout or 0.0)
        while True:
            with self._lock:
                if self._last_sensor is not None:
                    s = self._last_sensor
                    self._last_sensor = None
                    return s
            if timeout is None or timeout <= 0.0:
                return None
            if time.time() >= wait_until:
                return None
            time.sleep(0.001)

    def close(self):
        self._stop.set()
        try:
            if not self._provided and hasattr(self.serial, 'close'):
                self.serial.close()
        except Exception:
            pass
