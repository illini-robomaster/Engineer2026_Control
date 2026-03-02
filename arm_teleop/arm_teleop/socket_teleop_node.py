#!/usr/bin/env python3
"""
Socket-based teleoperation node.

Listens on a TCP socket for absolute end-effector target poses (6D Cartesian)
sent by the standalone arm_vision client.  Runs a proportional controller to
drive MoveIt Servo toward the target via TwistStamped.

Socket protocol: newline-delimited JSON messages
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Parameters:
  host                : bind host                (default: "0.0.0.0")
  port                : bind port                (default: 9999)
  base_frame          : robot base frame         (default: "base_link")
  ee_frame            : end-effector TF frame    (default: "End_Effector")
  kp_linear           : proportional gain m/s/m  (default: 1.5)
  kp_angular          : proportional gain rad/s/rad (default: 2.0)
  max_linear_speed    : output clamp m/s         (default: 0.25)
  max_angular_speed   : output clamp rad/s       (default: 0.5)
  deadband_pos_m      : position deadband m      (default: 0.008)
  deadband_rot_rad    : rotation deadband rad    (default: 0.05)
  detection_timeout_s : halt if no msg for this long (default: 0.4)
  control_orientation : also control EE orientation (default: false)
"""

import json
import math
import socket
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
from scipy.spatial.transform import Rotation


# ── Pure-numpy quaternion helpers (xyzw convention) ──────────────────────────

def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_error_rotvec(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Rotation vector (axis*angle) from current to target orientation."""
    q_err = qmul(q_target, qconj(q_current))
    if q_err[3] < 0:        # keep shortest-path
        q_err = -q_err
    vec = q_err[:3]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-7:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vec_norm, float(q_err[3]))
    return (angle / vec_norm) * vec


# ─────────────────────────────────────────────────────────────────────────────

class SocketTeleopNode(Node):

    def __init__(self):
        super().__init__('socket_teleop_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('host', '0.0.0.0')
        self.declare_parameter('port', 9999)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('ee_frame', 'End_Effector')
        self.declare_parameter('kp_linear', 1.5)
        self.declare_parameter('kp_angular', 2.0)
        self.declare_parameter('max_linear_speed', 0.25)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('deadband_pos_m', 0.008)
        self.declare_parameter('deadband_rot_rad', 0.05)
        self.declare_parameter('detection_timeout_s', 0.4)
        self.declare_parameter('control_orientation', True)

        self._host        = self.get_parameter('host').value
        self._port        = self.get_parameter('port').value
        self._base_frame  = self.get_parameter('base_frame').value
        self._ee_frame    = self.get_parameter('ee_frame').value
        self._kp_lin      = self.get_parameter('kp_linear').value
        self._kp_ang      = self.get_parameter('kp_angular').value
        self._max_lin     = self.get_parameter('max_linear_speed').value
        self._max_ang     = self.get_parameter('max_angular_speed').value
        self._db_pos      = self.get_parameter('deadband_pos_m').value
        self._db_rot      = self.get_parameter('deadband_rot_rad').value
        self._timeout     = self.get_parameter('detection_timeout_s').value
        self._ctrl_orient = self.get_parameter('control_orientation').value

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Publisher ─────────────────────────────────────────────────────────
        self._twist_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # ── Shared state (socket thread → control timer) ──────────────────────
        self._lock              = threading.Lock()
        self._target_pos: Optional[np.ndarray] = None   # [x, y, z] in base_frame
        self._target_quat: Optional[np.ndarray] = None  # [x, y, z, w]
        self._last_recv_time: float = 0.0

        # ── Diagnostics counters ───────────────────────────────────────────────
        self._diag_lock        = threading.Lock()
        self._recv_count       = 0   # socket messages received since last report
        self._pub_count        = 0   # twist messages published since last report
        self._tf_fail_count    = 0   # TF lookup failures since last report
        self._last_diag_time   = time.monotonic()
        self._first_msg_logged = False

        # ── Socket server (background thread) ─────────────────────────────────
        self._srv_thread = threading.Thread(
            target=self._socket_server, daemon=True)
        self._srv_thread.start()

        # ── Control loop at 30 Hz ─────────────────────────────────────────────
        self.create_timer(1.0 / 30.0, self._control_loop)

        # ── Diagnostics log every 2 s ─────────────────────────────────────────
        self.create_timer(2.0, self._diag_loop)

        self.get_logger().info(
            f'socket_teleop_node ready — listening on {self._host}:{self._port}  '
            f'control_orientation={self._ctrl_orient}  '
            f'kp_lin={self._kp_lin}  kp_ang={self._kp_ang}  '
            f'deadband_rot={math.degrees(self._db_rot):.1f}°')

    # ── Socket server ─────────────────────────────────────────────────────────

    def _socket_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, self._port))
            srv.listen(1)
            srv.settimeout(1.0)
            self.get_logger().info(
                f'TCP socket server listening on {self._host}:{self._port}')
            while rclpy.ok():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                self.get_logger().info(f'arm_vision client connected: {addr}')
                self._handle_client(conn, addr)

    def _handle_client(self, conn: socket.socket, addr):
        buf = ''
        with conn:
            conn.settimeout(1.0)
            while rclpy.ok():
                try:
                    chunk = conn.recv(4096).decode('utf-8', errors='replace')
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    self._parse_message(line.strip())
        self.get_logger().info(f'arm_vision client disconnected: {addr}')
        # Clear target so servo halts
        with self._lock:
            self._target_pos = None
            self._target_quat = None

    def _parse_message(self, line: str):
        if not line:
            return
        try:
            msg = json.loads(line)
            pos  = np.array([float(msg['x']),  float(msg['y']),  float(msg['z'])],
                            dtype=float)
            quat = np.array([float(msg['qx']), float(msg['qy']),
                             float(msg['qz']), float(msg['qw'])], dtype=float)
            n = np.linalg.norm(quat)
            if n < 1e-6:
                self.get_logger().warn('Received near-zero quaternion — message dropped.')
                return
            quat /= n
            with self._lock:
                self._target_pos  = pos
                self._target_quat = quat
                self._last_recv_time = time.monotonic()
                first = not self._first_msg_logged
                self._first_msg_logged = True
            with self._diag_lock:
                self._recv_count += 1
            if first:
                rpy = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
                self.get_logger().info(
                    f'[socket] First message received  '
                    f'pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  '
                    f'quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})  '
                    f'rpy=({rpy[0]:.1f}°, {rpy[1]:.1f}°, {rpy[2]:.1f}°)')
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'Bad socket message: {exc}  raw={line[:80]}')

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        with self._lock:
            target_pos  = self._target_pos
            target_quat = self._target_quat
            age = time.monotonic() - self._last_recv_time

        if target_pos is None or age > self._timeout:
            self._publish_zero()
            return

        # Lookup current EE pose
        try:
            tf = self._tf_buf.lookup_transform(
                self._base_frame, self._ee_frame, Time())
        except Exception as exc:
            with self._diag_lock:
                self._tf_fail_count += 1
            self._publish_zero()
            return

        t = tf.transform.translation
        r = tf.transform.rotation
        cur_pos  = np.array([t.x, t.y, t.z])
        cur_quat = np.array([r.x, r.y, r.z, r.w])

        # ── Position P-controller ─────────────────────────────────────────────
        pos_err  = target_pos - cur_pos
        pos_norm = float(np.linalg.norm(pos_err))
        linear   = np.zeros(3)
        if pos_norm > self._db_pos:
            linear = self._kp_lin * pos_err
            spd = float(np.linalg.norm(linear))
            if spd > self._max_lin:
                linear *= self._max_lin / spd

        # ── Orientation P-controller (optional) ───────────────────────────────
        angular = np.zeros(3)
        rot_err_deg = 0.0
        if self._ctrl_orient:
            rot_err      = q_error_rotvec(target_quat, cur_quat)
            rot_norm     = float(np.linalg.norm(rot_err))
            rot_err_deg  = math.degrees(rot_norm)
            if rot_norm > self._db_rot:
                angular = self._kp_ang * rot_err
                spd = float(np.linalg.norm(angular))
                if spd > self._max_ang:
                    angular *= self._max_ang / spd

        # ── Publish ───────────────────────────────────────────────────────────
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.twist.linear.x  = float(linear[0])
        msg.twist.linear.y  = float(linear[1])
        msg.twist.linear.z  = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self._twist_pub.publish(msg)

        with self._diag_lock:
            self._pub_count += 1

    def _publish_zero(self):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        self._twist_pub.publish(msg)
        with self._diag_lock:
            self._pub_count += 1

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _diag_loop(self):
        now = time.monotonic()
        with self._lock:
            target_pos  = self._target_pos
            target_quat = self._target_quat
            age = now - self._last_recv_time if self._last_recv_time else -1.0

        with self._diag_lock:
            recv   = self._recv_count;   self._recv_count   = 0
            pubs   = self._pub_count;    self._pub_count    = 0
            tf_fails = self._tf_fail_count; self._tf_fail_count = 0
            dt = now - self._last_diag_time
            self._last_diag_time = now

        recv_hz = recv / max(dt, 1e-3)
        pub_hz  = pubs / max(dt, 1e-3)

        if target_pos is None:
            self.get_logger().info(
                f'[diag] No socket data yet  '
                f'pub={pub_hz:.1f}Hz  tf_fails={tf_fails}')
            return

        rpy_tgt = Rotation.from_quat(target_quat).as_euler('xyz', degrees=True)

        # Current EE pose for orientation error display
        rot_err_str = 'n/a (ctrl_ori=false)'
        if self._ctrl_orient:
            try:
                tf = self._tf_buf.lookup_transform(
                    self._base_frame, self._ee_frame, Time())
                r = tf.transform.rotation
                cur_q = np.array([r.x, r.y, r.z, r.w])
                rot_err = q_error_rotvec(target_quat, cur_q)
                rot_err_str = f'{math.degrees(float(np.linalg.norm(rot_err))):.1f}°'
            except Exception as exc:
                rot_err_str = f'TF fail: {exc}'

        self.get_logger().info(
            f'[diag] socket={recv_hz:.0f}Hz  pub={pub_hz:.0f}Hz  '
            f'tf_fails={tf_fails}  msg_age={age:.2f}s  '
            f'ctrl_ori={self._ctrl_orient}  '
            f'tgt_pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  '
            f'tgt_rpy=({rpy_tgt[0]:.1f}°,{rpy_tgt[1]:.1f}°,{rpy_tgt[2]:.1f}°)  '
            f'ori_err={rot_err_str}')


def main(args=None):
    rclpy.init(args=args)
    node = SocketTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
