"""MAVLink <-> HIL gateway

Provides a small adapter to translate between MAVLink messages and the
HIL protocol implemented in `hil_interface.py`.

This module avoids importing `pymavlink` at import time; the gateway will
attempt to import `pymavlink` only when `start()` is called.
"""
from typing import Optional
import threading
import time
import json

from hil_interface import HILProtocol
from control_engine import ControlCommand


class MAVLinkHILGateway:
    """Translate MAVLink messages to HIL commands and sensors.

    Usage:
        gw = MAVLinkHILGateway(conn_url, hil_bridge)
        gw.start()
        gw.stop()

    For light testing, use `run_fake_message(msg_dict)` to simulate incoming
    MAVLink messages (no pymavlink needed).
    """

    def __init__(self, conn_url: str, hil_bridge, target_system: int = 1):
        self.conn_url = conn_url
        self.hil_bridge = hil_bridge
        self.target_system = target_system
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mav = None

    def start(self):
        # lazy import to avoid hard dependency at module import
        try:
            from pymavlink import mavutil
        except Exception as e:
            raise RuntimeError('pymavlink required to start MAVLinkHILGateway') from e

        self._mav = mavutil.mavlink_connection(self.conn_url)
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while not self._stop.is_set():
            m = self._mav.recv_match(timeout=0.1)
            if m is None:
                continue
            # basic handling: if actuator controls received, translate to HIL command
            try:
                m_type = m.get_type()
            except Exception:
                continue
            if m_type == 'ACTUATOR_CONTROL_TARGET' or m_type == 'RC_CHANNELS':
                cmd = self._mav_msg_to_control_command(m)
                if cmd is not None:
                    payload = HILProtocol.serialize_json_command(cmd)
                    # use send_command_with_ack if available
                    if hasattr(self.hil_bridge, 'send_command_with_ack'):
                        self.hil_bridge.send_command_with_ack(payload, mode='json')
                    else:
                        self.hil_bridge.send_command(payload, mode='json')
            # sensor messages like RAW_IMU, SCALED_IMU, etc., could be forwarded
            if m_type in ('RAW_IMU', 'SCALED_IMU', 'HIL_SENSOR'):
                sensor = self._mav_msg_to_sensor_frame(m)
                if sensor is not None:
                    # send as JSON sensor frame to the bridge's recv path by simulating an incoming packet
                    # many bridges listen only for incoming packets from hardware; here we just set last_sensor
                    if hasattr(self.hil_bridge, '_last_sensor'):
                        with getattr(self.hil_bridge, '_lock', threading.Lock()):
                            self.hil_bridge._last_sensor = sensor

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass

    def _mav_msg_to_control_command(self, m):
        """Convert basic MAVLink actuator/rc message to `ControlCommand`.

        This is a best-effort mapping: tests can call `run_fake_message` with
        a dict containing normalized fields.
        """
        try:
            # try attribute access for real pymavlink messages
            if hasattr(m, 'controls'):
                # ACTUATOR_CONTROL_TARGET: controls[0].. controls[3]
                thr = float(m.controls[0])
                theta = float(m.controls[1]) if len(m.controls) > 1 else 0.0
                phi = float(m.controls[2]) if len(m.controls) > 2 else 0.0
                ballast = float(m.controls[3]) if len(m.controls) > 3 else 0.0
            elif hasattr(m, 'chan1_raw'):
                # RC_CHANNELS parsing fallback — map channels 1..4
                thr = (m.chan1_raw - 1500) / 500.0
                theta = (m.chan2_raw - 1500) / 500.0
                phi = (m.chan3_raw - 1500) / 500.0
                ballast = (m.chan4_raw - 1500) / 500.0
            else:
                return None
        except Exception:
            return None

        return ControlCommand(thruster_power=thr, thruster_theta=theta, thruster_phi=phi, ballast_cmd=ballast)

    def _mav_msg_to_sensor_frame(self, m):
        # best-effort translation for RAW_IMU / SCALED_IMU
        try:
            d = {'timestamp': time.time()}
            if hasattr(m, 'xacc') and hasattr(m, 'xgyro'):
                d['imu'] = {'accel': [m.xacc, m.yacc, m.zacc], 'gyro': [m.xgyro, m.ygyro, m.zgyro]}
            elif hasattr(m, 'xgyro') and hasattr(m, 'ygyro'):
                d['imu'] = {'accel': [0.0, 0.0, 0.0], 'gyro': [m.xgyro, m.ygyro, m.zgyro]}
            return d
        except Exception:
            return None

    # helper for tests: feed a fake message dict
    def run_fake_message(self, msg: dict):
        """Accept a lightweight fake message dict and process it like a MAVLink msg."""
        # expected keys: type: 'ACTUATOR_CONTROL_TARGET' or 'RAW_IMU' etc.
        m_type = msg.get('type')
        if m_type in ('ACTUATOR_CONTROL_TARGET', 'RC_CHANNELS'):
            # msg should contain 'controls' list
            class M:
                pass
            m = M()
            m.controls = msg.get('controls', [])
            cmd = self._mav_msg_to_control_command(m)
            if cmd is not None:
                payload = HILProtocol.serialize_json_command(cmd)
                if hasattr(self.hil_bridge, 'send_command_with_ack'):
                    return self.hil_bridge.send_command_with_ack(payload, mode='json')
                else:
                    return self.hil_bridge.send_command(payload, mode='json')
        elif m_type in ('RAW_IMU', 'SCALED_IMU'):
            sensor = {'timestamp': time.time(), 'imu': msg.get('imu')}
            if hasattr(self.hil_bridge, '_last_sensor'):
                with getattr(self.hil_bridge, '_lock', threading.Lock()):
                    self.hil_bridge._last_sensor = sensor
            return True
        return False
