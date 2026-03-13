#!/usr/bin/env python3
"""
Socket-based teleoperation node.

Listens on a TCP socket for absolute end-effector target poses (6D Cartesian)
sent by the standalone arm_vision client.  Runs a proportional controller and
converts the desired Cartesian velocity to joint velocities using the
Singularity-Robust (SR) Damped Least-Squares (DLS) Jacobian inverse.

DLS guarantees bounded joint velocities even at exact singularities — the arm
will never e-stop due to a Jacobian blow-up.  Away from singularity the DLS
inverse equals the standard pseudoinverse (no accuracy penalty).

Requires python3-pykdl and python3-kdl-parser (ros-<distro>-kdl-parser-py).
Also requires robot_description to be passed as a node parameter — see
teleop.launch.py.

Socket protocol: newline-delimited JSON messages
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Parameters:
  host                : bind host                (default: "0.0.0.0")
  port                : bind port                (default: 9999)
  base_frame          : robot base frame         (default: "base_link")
  ee_frame            : end-effector TF frame    (default: "End_Effector")
  kp_linear           : proportional gain m/s/m  (default: 5.0)
  kp_angular          : proportional gain rad/s/rad (default: 4.0)
  max_linear_speed    : Cartesian output clamp m/s   (default: 1.5)
  max_angular_speed   : Cartesian output clamp rad/s (default: 2.0)
  deadband_pos_m      : position deadband m      (default: 0.008)
  deadband_rot_rad    : rotation deadband rad    (default: 0.05)
  detection_timeout_s : halt if no msg for this long (default: 0.4)
  control_orientation : also control EE orientation  (default: True)
  dls_lambda_max      : max DLS damping factor   (default: 0.1)
  dls_sigma_threshold : σ_min below which damping activates (default: 0.05)
  joint_names         : ordered joint names      (default: Joint1…Joint6)
  joint_velocity_limits : per-joint velocity caps rad/s (default: [1.2,1.2,1.2,1.5,1.5,1.8])
  robot_description   : URDF XML string for KDL chain (passed by launch file)
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Optional

import numpy as np
import PyKDL
from kdl_parser_py import urdf as kdl_urdf
from urdf_parser_py import urdf as urdf_parser

import rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
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
        self.declare_parameter('kp_linear', 5.0)
        self.declare_parameter('kp_angular', 4.0)
        self.declare_parameter('max_linear_speed', 1.5)
        self.declare_parameter('max_angular_speed', 2.0)
        self.declare_parameter('deadband_pos_m', 0.008)
        self.declare_parameter('deadband_rot_rad', 0.05)
        self.declare_parameter('detection_timeout_s', 0.4)
        self.declare_parameter('control_orientation', True)
        self.declare_parameter('dls_lambda_max', 0.1)
        self.declare_parameter('dls_sigma_threshold', 0.05)
        self.declare_parameter('joint_names',
            ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6'])
        self.declare_parameter('joint_velocity_limits',
            [1.2, 1.2, 1.2, 1.5, 1.5, 1.8])
        self.declare_parameter('robot_description', '')

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
        self._dls_lam_max = self.get_parameter('dls_lambda_max').value
        self._dls_sigma   = self.get_parameter('dls_sigma_threshold').value
        self._joint_names = list(self.get_parameter('joint_names').value)
        self._joint_vlims = list(self.get_parameter('joint_velocity_limits').value)
        robot_desc        = self.get_parameter('robot_description').value

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Publisher: JointTrajectory → uart_bridge (bypasses MoveIt Servo) ───
        self._traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self._control_dt = 1.0 / 30.0   # matches control loop timer

        # ── Shared state ──────────────────────────────────────────────────────
        self._lock              = threading.Lock()
        self._target_pos: Optional[np.ndarray] = None
        self._target_quat: Optional[np.ndarray] = None
        self._last_recv_time: float = 0.0
        self._joint_pos: dict = {}           # name → rad, updated from /joint_states

        # ── Diagnostics counters ───────────────────────────────────────────────
        self._diag_lock        = threading.Lock()
        self._recv_count       = 0
        self._pub_count        = 0
        self._tf_fail_count    = 0
        self._last_diag_time   = time.monotonic()
        self._first_msg_logged = False

        # ── Joint state subscriber ─────────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        # ── KDL Jacobian solver setup ──────────────────────────────────────────
        self._jac_solver = None
        self._kdl_chain  = None
        if robot_desc:
            try:
                robot_model = urdf_parser.URDF.from_xml_string(robot_desc)
                ok, tree = kdl_urdf.treeFromUrdfModel(robot_model)
                if not ok:
                    raise RuntimeError('kdl_urdf.treeFromUrdfModel returned False')
                self._kdl_chain  = tree.getChain(self._base_frame, self._ee_frame)
                self._jac_solver = PyKDL.ChainJntToJacSolver(self._kdl_chain)
                n = self._kdl_chain.getNrOfJoints()
                self.get_logger().info(
                    f'KDL chain ready — {n} joints  '
                    f'{self._base_frame} → {self._ee_frame}  '
                    f'DLS λ_max={self._dls_lam_max}  σ_thresh={self._dls_sigma}')
            except Exception as exc:
                self.get_logger().error(
                    f'KDL setup failed ({exc}) — node will not publish joint commands.')
        else:
            self.get_logger().error(
                'robot_description parameter is empty.  '
                'Add robot_description to the socket_teleop_node parameters in '
                'teleop.launch.py.  Node will not move the arm.')

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
            f'deadband_rot={math.degrees(self._db_rot):.1f}°  '
            f'DLS={"enabled" if self._jac_solver else "DISABLED (no robot_description)"}')

    # ── Joint state subscriber ─────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_pos[name] = pos

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

    # ── DLS Jacobian inverse ───────────────────────────────────────────────────

    def _compute_dls(self, linear: np.ndarray, angular: np.ndarray,
                     joint_pos: dict) -> Optional[np.ndarray]:
        """Compute joint velocities via Singularity-Robust DLS inverse.

        Standard pseudoinverse blows up near singular configurations, sending
        huge joint velocity spikes to the arm controller (→ e-stop).  DLS adds
        an adaptive damping term λ² that keeps joint velocities bounded even AT
        an exact singularity.

        Away from singularity: λ ≈ 0, so DLS ≈ pseudoinverse (no accuracy loss).
        Near singularity (σ_min < dls_sigma_threshold): λ grows, accuracy is
        sacrificed but the arm keeps moving smoothly in available directions.

        Uses the Nakamura–Hanafusa (1986) SR-inverse:
            q̇ = Jᵀ (J Jᵀ + λ²I)⁻¹ v
        with λ = λ_max · (1 − σ_min/σ_thresh)  when σ_min < σ_thresh.
        """
        if self._jac_solver is None:
            return None

        n = len(self._joint_names)
        q = PyKDL.JntArray(n)
        for i, name in enumerate(self._joint_names):
            q[i] = joint_pos.get(name, 0.0)

        jac_kdl = PyKDL.Jacobian(n)
        self._jac_solver.JntToJac(q, jac_kdl)
        J = np.array([[jac_kdl[i, j] for j in range(n)] for i in range(6)])

        if self._ctrl_orient:
            J_use = J                               # 6×n
            v     = np.concatenate([linear, angular])  # 6D
        else:
            J_use = J[:3, :]                        # 3×n  (position-only)
            v     = linear                          # 3D

        m = J_use.shape[0]

        # Adaptive damping: λ ramps from 0 → λ_max as σ_min drops below σ_thresh
        svals     = np.linalg.svd(J_use, compute_uv=False)
        sigma_min = float(svals[-1])
        if sigma_min < self._dls_sigma:
            ratio  = sigma_min / self._dls_sigma
            lam_sq = self._dls_lam_max ** 2 * (1.0 - ratio) ** 2
        else:
            lam_sq = 0.0

        JJT   = J_use @ J_use.T
        q_dot = J_use.T @ np.linalg.solve(JJT + lam_sq * np.eye(m), v)

        # Hard clip to per-joint velocity limits (URDF values)
        for i, lim in enumerate(self._joint_vlims):
            q_dot[i] = float(np.clip(q_dot[i], -lim, lim))

        return q_dot

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        with self._lock:
            target_pos  = self._target_pos
            target_quat = self._target_quat
            age         = time.monotonic() - self._last_recv_time
            joint_pos   = dict(self._joint_pos)

        if target_pos is None or age > self._timeout:
            self._publish_zero()
            return

        # Lookup current EE pose
        try:
            tf = self._tf_buf.lookup_transform(
                self._base_frame, self._ee_frame, Time())
        except Exception:
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
        if self._ctrl_orient:
            rot_err  = q_error_rotvec(target_quat, cur_quat)
            rot_norm = float(np.linalg.norm(rot_err))
            if rot_norm > self._db_rot:
                angular = self._kp_ang * rot_err
                spd = float(np.linalg.norm(angular))
                if spd > self._max_ang:
                    angular *= self._max_ang / spd

        # ── DLS joint velocity control ─────────────────────────────────────────
        q_dot = self._compute_dls(linear, angular, joint_pos)
        if q_dot is None:
            self._publish_zero()
            return

        # Integrate: q_cmd = q_actual + q_dot * dt  (bypass MoveIt Servo)
        q_cmd = [joint_pos.get(n, 0.0) + float(q_dot[i]) * self._control_dt
                 for i, n in enumerate(self._joint_names)]

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names  = list(self._joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = q_cmd
        pt.time_from_start = Duration(sec=0,
                                      nanosec=int(self._control_dt * 1e9))
        msg.points = [pt]
        self._traj_pub.publish(msg)

        with self._diag_lock:
            self._pub_count += 1

    def _publish_zero(self):
        """Hold current position — don't snap to zero joints."""
        with self._lock:
            joint_pos = dict(self._joint_pos)
        if not joint_pos:
            return   # no encoder data yet; nothing safe to send
        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names  = list(self._joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [joint_pos.get(n, 0.0) for n in self._joint_names]
        pt.time_from_start = Duration(sec=0,
                                      nanosec=int(self._control_dt * 1e9))
        msg.points = [pt]
        self._traj_pub.publish(msg)
        with self._diag_lock:
            self._pub_count += 1

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _diag_loop(self):
        now = time.monotonic()
        with self._lock:
            target_pos  = self._target_pos
            target_quat = self._target_quat
            age = now - self._last_recv_time if self._last_recv_time else -1.0
            joint_pos   = dict(self._joint_pos)

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

        pos_err_str = 'TF unavailable'
        rot_err_str = 'TF unavailable'
        singularity_str = 'N/A'
        try:
            tf = self._tf_buf.lookup_transform(
                self._base_frame, self._ee_frame, Time())
            t = tf.transform.translation
            r = tf.transform.rotation
            cur_pos = np.array([t.x, t.y, t.z])
            cur_q   = np.array([r.x, r.y, r.z, r.w])
            pos_err_m = float(np.linalg.norm(target_pos - cur_pos))
            pos_err_str = f'{pos_err_m * 1000:.1f}mm'
            rot_err = q_error_rotvec(target_quat, cur_q)
            rot_err_str = f'{math.degrees(float(np.linalg.norm(rot_err))):.1f}°'

            # Singularity metric: smallest singular value and condition number
            if self._jac_solver is not None:
                n = len(self._joint_names)
                q = PyKDL.JntArray(n)
                for i, name in enumerate(self._joint_names):
                    q[i] = joint_pos.get(name, 0.0)
                jac_kdl = PyKDL.Jacobian(n)
                self._jac_solver.JntToJac(q, jac_kdl)
                J = np.array([[jac_kdl[i, j] for j in range(n)] for i in range(6)])
                svals = np.linalg.svd(J, compute_uv=False)
                cn = svals[0] / max(svals[-1], 1e-9)
                singularity_str = f'σ_min={svals[-1]:.4f}  CN={cn:.1f}'
        except Exception as exc:
            pos_err_str = rot_err_str = f'TF fail: {type(exc).__name__}'

        self.get_logger().info(
            f'[diag] socket={recv_hz:.0f}Hz  pub={pub_hz:.0f}Hz  '
            f'tf_fails={tf_fails}  msg_age={age:.2f}s  '
            f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  '
            f'tgt_rpy=({rpy_tgt[0]:.1f}°,{rpy_tgt[1]:.1f}°,{rpy_tgt[2]:.1f}°)  '
            f'pos_err={pos_err_str}  ori_err={rot_err_str}  {singularity_str}')


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
