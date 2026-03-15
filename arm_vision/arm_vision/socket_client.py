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

# Max time a single send_pose() call may block the main (display) loop.
# connect() at startup uses the full self._timeout; reconnects inside
# send_pose() use _RECONNECT_TIMEOUT so the window stays responsive.
_SEND_TIMEOUT      = 0.05   # 50 ms — sendall() deadline
_RECONNECT_TIMEOUT = 0.30   # 300 ms — inline reconnect deadline


class PoseSocketClient:
    """Single-thread TCP client — send_pose() transmits inline, no background thread.

    Timeouts are chosen so the display loop is never stalled more than
    ~350 ms even when the ROS node is unreachable.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9999,
                 timeout: float = 2.0):
        self._host    = host
        self._port    = port
        self._timeout = timeout          # used only for the initial connect()
        self._sock: socket.socket | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self):
        """Connect to the ROS node (called once at startup).
        Silently ignores errors — send_pose() will reconnect on the next frame."""
        try:
            self._connect(self._timeout)
        except OSError:
            pass

    def stop(self):
        """Close the socket."""
        self._close()

    def send_pose(self, x, y, z, qx, qy, qz, qw):
        """Send one pose immediately.  Auto-reconnects once on broken connection.

        Blocks the caller for at most _SEND_TIMEOUT + _RECONNECT_TIMEOUT (~350 ms)
        so the display loop is never frozen.
        """
        line = json.dumps({
            'x': x, 'y': y, 'z': z,
            'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
        }) + '\n'
        if not self._send(line):
            try:
                self._connect(_RECONNECT_TIMEOUT)
                self._send(line)
            except OSError:
                pass                    # drop this frame; retry on next

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _send(self, line: str) -> bool:
        if self._sock is None:
            return False
        try:
            self._sock.sendall(line.encode('utf-8'))
            return True
        except OSError:
            self._close()
            return False

    def _connect(self, timeout: float):
        """Open a fresh TCP connection with the given timeout.
        After connecting, sendall() uses _SEND_TIMEOUT so it never blocks long."""
        self._close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((self._host, self._port))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(_SEND_TIMEOUT)   # short timeout for all subsequent I/O
        self._sock = sock
        print(f'[socket_client] Connected to {self._host}:{self._port}')

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
