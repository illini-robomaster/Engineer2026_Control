"""
TCP socket client — sends 6D end-effector target poses to the ROS
socket_teleop_node.

Protocol: newline-delimited JSON
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

The client automatically reconnects if the connection is lost.
"""

from __future__ import annotations

import json
import socket
import threading
import time


class PoseSocketClient:
    """
    Non-blocking TCP client that queues the latest pose and sends it
    in a background thread.

    Parameters
    ----------
    host    : ROS host running socket_teleop_node (default: 127.0.0.1)
    port    : TCP port (default: 9999)
    timeout : Connection / send timeout in seconds (default: 2.0)
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9999,
                 timeout: float = 2.0):
        self._host    = host
        self._port    = port
        self._timeout = timeout

        self._lock         = threading.Lock()
        self._pending: str | None = None   # JSON line ready to send
        self._sock: socket.socket | None = None
        self._running = False

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self):
        """Start the background sender thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background sender thread and close the socket."""
        self._running = False
        self._close()

    def send_pose(
        self,
        x: float, y: float, z: float,
        qx: float, qy: float, qz: float, qw: float,
    ):
        """
        Queue a target pose for sending.  Thread-safe.
        Only the most recent pose is kept — older ones are discarded if the
        socket is congested.
        """
        line = json.dumps({
            'x': x, 'y': y, 'z': z,
            'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
        }) + '\n'
        with self._lock:
            self._pending = line

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self):
        while self._running:
            try:
                self._connect()
                self._send_loop()
            except OSError:
                pass
            if self._running:
                print(f'[socket_client] Reconnecting to {self._host}:{self._port} in 1 s…')
                time.sleep(1.0)

    def _connect(self):
        self._close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect((self._host, self._port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        print(f'[socket_client] Connected to {self._host}:{self._port}')

    def _send_loop(self):
        while self._running:
            with self._lock:
                line = self._pending
                self._pending = None
            if line is not None:
                self._sock.sendall(line.encode('utf-8'))
            else:
                time.sleep(0.005)   # ~200 Hz idle poll

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
