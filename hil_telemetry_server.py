"""Lightweight telemetry broadcaster for HIL experiments.

Provides a UDP broadcaster that can be polled by a dashboard. Keeps a
latest-frame cache and broadcasts periodic updates when `start()` is
called.
"""
import socket
import threading
import time
import json
from typing import Dict, Any, Tuple


class TelemetryBroadcaster:
    def __init__(self, remote_addr: Tuple[str, int] = ('127.0.0.1', 5006), rate_hz: float = 50.0):
        self.remote_addr = remote_addr
        self.rate_hz = rate_hz
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._frame = None
        self._stop = threading.Event()
        self._thread = None

    def update_frame(self, frame: Dict[str, Any]):
        self._frame = frame

    def _run(self):
        interval = 1.0 / max(1.0, self.rate_hz)
        while not self._stop.is_set():
            if self._frame is not None:
                try:
                    payload = json.dumps(self._frame).encode('utf-8')
                    self.sock.sendto(payload, self.remote_addr)
                except Exception:
                    pass
            time.sleep(interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        try:
            self.sock.close()
        except Exception:
            pass
