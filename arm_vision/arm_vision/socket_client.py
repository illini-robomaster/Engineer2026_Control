"""
TCP socket client — sends 6D end-effector target poses to the ROS
ik_teleop_node and receives FK feedback in return.

Protocol: newline-delimited JSON (both directions)

  Mac → ROS (100 Hz):
    {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

  ROS → Mac (~10 Hz):
    {"fk_x": 0.526, "fk_y": 0.0, "fk_z": 0.396, "ik_ok": true,
     "j1": 0.0, "j2": 0.3, "j3": 1.5, "j4": 0.0, "j5": 0.0, "j6": 0.0}

The client automatically reconnects if the connection is lost.
"""

from __future__ import annotations

import json
import socket
import threading
import time

# Max time a single send_pose() call may block the main (display) loop.
# connect() at startup uses the full self._timeout; reconnects inside
# send_pose() use _RECONNECT_TIMEOUT so the window stays responsive.
_SEND_TIMEOUT      = 0.05   # 50 ms — sendall() deadline
_RECONNECT_TIMEOUT = 0.30   # 300 ms — inline reconnect deadline


class PoseSocketClient:
    """Full-duplex TCP client.

    send_pose() transmits inline on the main thread.
    A background thread receives FK feedback from the ROS node and exposes
    it via the feedback property.

    Timeouts are chosen so the display loop is never stalled more than
    ~350 ms even when the ROS node is unreachable.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9999,
                 timeout: float = 2.0):
        self._host    = host
        self._port    = port
        self._timeout = timeout          # used only for the initial connect()
        self._sock: socket.socket | None = None

        self._feedback: dict = {}
        self._feedback_lock = threading.Lock()

        self._recv_stop  = threading.Event()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self):
        """Connect to the ROS node (called once at startup) and start the
        background receive thread.  Silently ignores connection errors —
        send_pose() will reconnect on the next frame."""
        self._recv_stop.clear()
        self._recv_thread.start()
        try:
            self._connect(self._timeout)
        except OSError:
            pass

    def stop(self):
        """Close the socket and stop the receive thread."""
        self._recv_stop.set()
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

    def send_home(self):
        """Send a homing trigger command to the ROS node.

        The node will spawn homing_node locally on Linux and report progress
        via the FK feedback stream (homing=True while running, then absent).
        """
        line = json.dumps({'cmd': 'home'}) + '\n'
        if not self._send(line):
            try:
                self._connect(_RECONNECT_TIMEOUT)
                self._send(line)
            except OSError:
                pass

    def send_joints(self, positions_deg: list, duration_s: float):
        """Send a joint-space target (degrees) with a motion duration."""
        line = json.dumps({
            'cmd': 'joints',
            'positions': positions_deg,
            'duration': duration_s,
        }) + '\n'
        if not self._send(line):
            try:
                self._connect(_RECONNECT_TIMEOUT)
                self._send(line)
            except OSError:
                pass

    def send_plan_joints(self, positions_deg: list):
        """Send a joint-space target (degrees) to be executed via MoveIt planning.

        The ROS node will plan a collision-free trajectory with OMPL and execute
        it, reporting progress via planning_active=True/False in the FK feedback.
        """
        line = json.dumps({
            'cmd': 'plan_joints',
            'positions': positions_deg,
        }) + '\n'
        if not self._send(line):
            try:
                self._connect(_RECONNECT_TIMEOUT)
                self._send(line)
            except OSError:
                pass

    def send_claw(self, open: bool):
        """Stub — claw control via separate channel (not yet wired)."""
        pass  # TODO: wire to UART / separate topic

    @property
    def feedback(self) -> dict:
        """Latest FK feedback from the ROS node.

        Keys present when IK node is connected and has solved at least once:
          fk_x, fk_y, fk_z  — FK end-effector position (metres)
          ik_ok              — True when IK converged; False when snapped or failed
          j1 … j6           — solved joint angles (radians)
        Returns an empty dict until the first feedback message arrives.
        """
        with self._feedback_lock:
            return dict(self._feedback)

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
        After connecting, I/O uses _SEND_TIMEOUT so it never blocks long."""
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

    def _recv_loop(self):
        """Background thread: reads newline-delimited JSON from the server.

        The socket's timeout (_SEND_TIMEOUT = 50 ms) means recv() loops at
        up to 20 Hz, which is fast enough to catch the 10 Hz feedback stream.
        Thread-safe: uses a local snapshot of self._sock each iteration.
        """
        buf = ''
        while not self._recv_stop.is_set():
            sock = self._sock   # snapshot — may become None if send fails
            if sock is None:
                buf = ''
                time.sleep(0.05)
                continue
            try:
                chunk = sock.recv(4096).decode('utf-8', errors='replace')
                if not chunk:
                    # Server closed connection
                    time.sleep(0.05)
                    continue
                buf += chunk
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            with self._feedback_lock:
                                self._feedback = msg
                        except json.JSONDecodeError:
                            pass
            except socket.timeout:
                continue    # no data yet — normal at 50 ms timeout
            except OSError:
                buf = ''    # connection reset; wait for send_pose to reconnect
                time.sleep(0.05)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
