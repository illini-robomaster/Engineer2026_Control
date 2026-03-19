#!/usr/bin/env python3
"""
IK-direct teleoperation node.

Receives 6D target poses from arm_vision over TCP.  Solves IK using a custom
DLS (damped least-squares) solver built entirely on numpy.

Two modes (controlled by the control_orientation parameter):
  False (default) -- 3D position-only IK.  Orientation is free; the solver
                     picks the joint-space-nearest solution from the seed.
  True            -- 6D pose IK.  Both position AND orientation are enforced
                     using the geometric Jacobian (linear + angular rows).
                     Only use after the camera->robot orientation frame is
                     validated; an unreachable orientation will diverge.

FK and the Jacobian are computed from the URDF joint transforms using numpy
4x4 homogeneous matrices -- no PyKDL binding runtime issues.

Socket protocol: newline-delimited JSON
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Parameters:
  host                : bind host              (default: "0.0.0.0")
  port                : bind port              (default: 9999)
  base_frame          : robot base TF frame    (default: "base_link")
  ee_frame            : end-effector link name (default: "End_Effector")
  publish_rate_hz     : IK solve + publish rate Hz   (default: 30.0)
  traj_duration_s     : time_from_start in published JointTrajectory (default: 0.05)
  detection_timeout_s : hold if no TCP message for this long (default: 0.4)
  joint_deadband_rad  : suppress publish if joints move less than this (default: 0.001)
  control_orientation : False = position-only (default), True = full 6D pose
  ori_weight          : scale applied to orientation error rows in 6D mode (default: 1.0)
  robot_description   : URDF XML string (passed by launch file)
"""

from __future__ import annotations

import csv
import json
import os
import socket
import subprocess
import threading
import time
from typing import Optional

import numpy as np
from urdf_parser_py import urdf as urdf_parser

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    MotionPlanRequest,
)


# ── Pure-numpy FK + geometric Jacobian ───────────────────────────────────────

def _rpy_xyz_to_mat4(rpy: list, xyz: list) -> np.ndarray:
    """4x4 homogeneous matrix from URDF joint origin (Rz*Ry*Rx, then translate)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,              cp*cr           ],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = xyz
    return T


def _axis_angle_mat4(axis: np.ndarray, angle: float) -> np.ndarray:
    """4x4 homogeneous rotation by `angle` rad around unit `axis`."""
    ax, ay, az = axis
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [t*ax*ax + c,     t*ax*ay - s*az,  t*ax*az + s*ay],
        [t*ax*ay + s*az,  t*ay*ay + c,     t*ay*az - s*ax],
        [t*ax*az - s*ay,  t*ay*az + s*ax,  t*az*az + c   ],
    ])
    return T


def _quat_to_mat3(q: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] -> 3x3 rotation matrix."""
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),      1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
    ])


def _mat3_to_rpy(R: np.ndarray) -> tuple:
    """3x3 rotation matrix → extrinsic XYZ RPY (radians), handles gimbal lock."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-6:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw  = float(np.arctan2(R[1, 0], R[0, 0]))
    else:                                   # gimbal lock
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw  = 0.0
    return roll, pitch, yaw


def _ik_diag_str(chain_data, last_solved, target_pos, target_quat) -> str:
    """Build a compact one-line diagnostic string for IK failure messages.

    Shows: target xyz/RPY, current EE xyz/RPY, orientation error, joint angles,
    and a wrist-singularity hint if J5 is near 0.
    """
    parts = [f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})']

    if target_quat is not None:
        tr, tp, ty = _mat3_to_rpy(_quat_to_mat3(target_quat))
        parts.append(f'tgt_rpy=({np.degrees(tr):.1f},{np.degrees(tp):.1f},{np.degrees(ty):.1f})°')

    if last_solved is not None:
        p_cur, R_cur, _ = _fk_and_jac(chain_data, np.array(last_solved))
        cr, cp, cy = _mat3_to_rpy(R_cur)
        dist = float(np.linalg.norm(target_pos - p_cur))
        parts.append(f'cur=({p_cur[0]:.3f},{p_cur[1]:.3f},{p_cur[2]:.3f})'
                     f'  dist={dist*1000:.1f}mm')
        parts.append(f'cur_rpy=({np.degrees(cr):.1f},{np.degrees(cp):.1f},{np.degrees(cy):.1f})°')

        if target_quat is not None:
            R_tgt = _quat_to_mat3(target_quat)
            R_err = R_tgt @ R_cur.T
            e_r = 0.5 * np.array([R_err[2,1]-R_err[1,2],
                                   R_err[0,2]-R_err[2,0],
                                   R_err[1,0]-R_err[0,1]])
            parts.append(f'ori_err={np.degrees(np.linalg.norm(e_r)):.1f}°')

        j_deg = [float(np.degrees(j)) for j in last_solved]
        parts.append('joints=' + ' '.join(f'J{i+1}:{v:.1f}°' for i, v in enumerate(j_deg)))

        j5 = j_deg[4] if len(j_deg) > 4 else None
        if j5 is not None and abs(j5) < 8.0:
            parts.append(f'!! J5={j5:.1f}° near wrist singularity (J5≈0 → J4/J6 coaxial)')

    return '  |  '.join(parts)


def _rpy_to_quat(r: float, p: float, y: float) -> np.ndarray:
    """Extrinsic XYZ RPY -> quaternion [qx, qy, qz, qw]."""
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [qx, qy, qz, qw]."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _build_chain_data(robot_desc: str, base: str, tip: str) -> list:
    """Parse URDF; return ordered joint dicts from base to tip.

    Each dict: name, type, T_fixed (4x4), axis (unit np.ndarray).
    """
    model = urdf_parser.URDF.from_xml_string(robot_desc.encode('utf-8'))
    joint_from_child = {j.child: j for j in model.joints}

    path = []
    current = tip
    while current != base:
        joint = joint_from_child.get(current)
        if joint is None:
            raise RuntimeError(f'No URDF path from {base!r} to {tip!r}: '
                               f'link {current!r} has no parent joint')
        path.append(joint)
        current = joint.parent
    path.reverse()

    chain_data = []
    for j in path:
        xyz = list(j.origin.xyz) if j.origin and j.origin.xyz else [0., 0., 0.]
        rpy = list(j.origin.rpy) if j.origin and j.origin.rpy else [0., 0., 0.]
        raw_axis = list(j.axis) if j.axis else [1., 0., 0.]
        axis = np.array(raw_axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n > 1e-10:
            axis /= n
        lower = float(j.limit.lower) if j.limit and j.limit.lower is not None else -np.pi
        upper = float(j.limit.upper) if j.limit and j.limit.upper is not None else  np.pi
        chain_data.append({
            'name':    j.name,
            'type':    j.type,
            'T_fixed': _rpy_xyz_to_mat4(rpy, xyz),
            'axis':    axis,
            'lower':   lower,
            'upper':   upper,
        })
    return chain_data


def _joint_limits(chain_data: list) -> tuple:
    """Return (lower, upper) arrays for all revolute joints in chain order."""
    lowers = np.array([s['lower'] for s in chain_data if s['type'] in ('revolute', 'continuous')])
    uppers = np.array([s['upper'] for s in chain_data if s['type'] in ('revolute', 'continuous')])
    return lowers, uppers


def _fk_and_jac(chain_data: list, q: np.ndarray) -> tuple:
    """Compute FK and the 6xN geometric Jacobian.

    Returns:
        p_ee  : np.ndarray (3,)   -- end-effector position
        R_ee  : np.ndarray (3,3)  -- end-effector rotation matrix
        J     : np.ndarray (6,N)  -- geometric Jacobian
                  J[:3, i] = z_i x (p_ee - o_i)   (linear velocity)
                  J[3:, i] = z_i                   (angular velocity)
    """
    T = np.eye(4)
    joint_origins: list = []
    joint_axes_world: list = []
    qi = 0

    for seg in chain_data:
        T = T @ seg['T_fixed']
        if seg['type'] in ('revolute', 'continuous'):
            joint_origins.append(T[:3, 3].copy())
            joint_axes_world.append(T[:3, :3] @ seg['axis'])
            T = T @ _axis_angle_mat4(seg['axis'], q[qi])
            qi += 1

    p_ee = T[:3, 3]
    R_ee = T[:3, :3].copy()
    n = len(q)
    J = np.zeros((6, n))
    for i in range(n):
        zi = joint_axes_world[i]
        J[:3, i] = np.cross(zi, p_ee - joint_origins[i])   # linear
        J[3:, i] = zi                                        # angular
    return p_ee, R_ee, J


# ─────────────────────────────────────────────────────────────────────────────

class IkTeleopNode(Node):

    def __init__(self):
        super().__init__('ik_teleop_node')

        def p(name, default): return self.declare_parameter(name, default).value

        self._host        = p('host',               '0.0.0.0')
        self._port        = p('port',               9999)
        self._base_frame  = p('base_frame',         'base_link')
        self._ee_frame    = p('ee_frame',            'End_Effector')
        self._rate_hz     = p('publish_rate_hz',     30.0)
        self._ik_timeout  = p('ik_timeout_s',        0.005)   # kept for param compat
        self._traj_dur    = p('traj_duration_s',     0.05)
        self._det_timeout = p('detection_timeout_s', 0.4)
        self._ctrl_ori    = p('control_orientation', False)
        self._ori_weight     = p('ori_weight',          1.0)
        self._n_random_seeds = p('n_random_seeds',      15)
        off_r = p('ee_ori_offset_roll',  0.0)
        off_p = p('ee_ori_offset_pitch', 0.0)
        off_y = p('ee_ori_offset_yaw',   0.0)
        self._ee_ori_offset  = _rpy_to_quat(off_r, off_p, off_y)
        robot_desc           = p('robot_description',   '')

        # ── Publisher ────────────────────────────────────────────────────────
        self._traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)

        self._joint_deadband  = p('joint_deadband_rad',  0.001)
        # Reject IK solutions where any joint moved more than this from the
        # current hardware position.  Catches configuration flips (e.g. J4 ±180°)
        # that are geometrically valid but physically undesirable.
        # Set 0.0 to disable.
        self._max_joint_jump  = p('max_joint_jump_rad',  1.5)
        # Joint-space velocity limit (rad/s).  Prevents sudden jumps when IK
        # re-acquires a solution after traversing an unreachable zone — the arm
        # ramps smoothly to the new target instead of snapping.
        # Set 0.0 to disable.
        self._max_joint_vel   = p('max_joint_vel_rad_s', 2.0)
        # Regularization weight for 6D IK — biases the solver toward the seed
        # configuration to prevent wrist-singularity config flips (J4 ±180°).
        # Higher = more conservative (won't deviate from seed), lower = less biased.
        # 0.0 disables regularization.
        self._ik_seed_regularization = p('ik_seed_regularization', 0.02)
        # Singularity-aware DLS damping floor for 6D IK.
        # When sigma_min of the weighted Jacobian falls below sing_threshold,
        # lambda is raised proportionally to prevent jitter near wrist singularity.
        self._sing_threshold = p('sing_threshold', 0.05)   # rad/s per rad
        self._sing_lam_max   = p('sing_lam_max',   0.005)  # max extra damping

        # ── Null-space posture control ────────────────────────────────────────
        # Secondary task: pull joints toward q_preferred without disturbing the
        # primary EE position task.  Only has effect in position-only mode
        # (3D task, 6 joints → 3D null space).  Set 0.0 to disable.
        self._nullspace_gain = p('nullspace_gain', 0.0)
        _q_pref_list         = p('q_preferred',    [0.0] * 6)
        self._q_preferred    = np.array(_q_pref_list, dtype=float)

        # ── Target pose EMA smoothing (velocity-adaptive) ────────────────────
        # pose_alpha:         alpha used when target is stationary (suppress IK noise).
        # pose_alpha_moving:  alpha used when target is moving fast (pass-through, no lag).
        # Blends linearly between the two based on per-tick position delta magnitude.
        # alpha=1.0 → raw (no smoothing); alpha<1.0 → low-pass (lag proportional to 1-alpha).
        self._pose_alpha          = p('pose_alpha',          1.0)
        self._pose_alpha_moving   = p('pose_alpha_moving',   0.9)
        self._pose_alpha_threshold_m = p('pose_alpha_threshold_m', 0.002)
        self._smoothed_pos:  Optional[np.ndarray] = None
        self._smoothed_quat: Optional[np.ndarray] = None
        self._prev_raw_pos:  Optional[np.ndarray] = None

        # ── Shared state ─────────────────────────────────────────────────────
        self._lock               = threading.Lock()
        self._joint_pos: dict    = {}
        self._target_pos: Optional[np.ndarray]  = None
        self._target_quat: Optional[np.ndarray] = None
        self._last_recv_time: float = 0.0
        self._last_solved_joints: Optional[list] = None

        # ── FK feedback (written by control loop, read by feedback sender) ────
        self._fk_feedback: dict = {}
        self._fk_feedback_lock = threading.Lock()

        # ── /joint_states subscriber ──────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        # ── Chain data (numpy FK + Jacobian) ──────────────────────────────────
        self._chain_data:  list = []
        self._joint_names: list = []
        self._n_joints:    int = 0

        if robot_desc:
            try:
                self._chain_data  = _build_chain_data(
                    robot_desc, self._base_frame, self._ee_frame)
                self._joint_names = [
                    s['name'] for s in self._chain_data
                    if s['type'] in ('revolute', 'continuous')]
                self._n_joints    = len(self._joint_names)

                mode_str = '6D pose' if self._ctrl_ori else '3D position-only'
                self.get_logger().info(f'IK: numpy DLS {mode_str}  ori_weight={self._ori_weight}')

                # ── Startup FK sanity-check ───────────────────────────────────
                _p0, _R0, _J0 = _fk_and_jac(self._chain_data, np.zeros(self._n_joints))
                _j_norm = float(np.linalg.norm(_J0[:3, :]))   # position rows
                _ok     = _j_norm > 0.1
                _log    = self.get_logger().info if _ok else self.get_logger().error
                _log(
                    f'Numpy FK@[0]*{self._n_joints}: '
                    f'p=({_p0[0]:.4f},{_p0[1]:.4f},{_p0[2]:.4f})  '
                    f'J_pos_norm={_j_norm:.4f}  '
                    f'{"OK" if _ok else "*** BROKEN -- check URDF base/tip frames ***"}'
                )

                self.get_logger().info(
                    f'Chain ready -- {self._n_joints} joints  '
                    f'{self._base_frame}->{self._ee_frame}  '
                    f'rate={self._rate_hz:.0f}Hz  joints={self._joint_names}')

            except Exception as exc:
                self.get_logger().error(
                    f'Chain setup failed ({exc}) -- node will not publish joint commands.')
        else:
            self.get_logger().error(
                'robot_description parameter is empty -- node will not move the arm.')

        # ── MoveIt joint-space planning ───────────────────────────────────────
        self._joints_plan_time   = p('joints_plan_time_s',    5.0)
        self._joints_vel_scale   = p('joints_vel_scale',      0.3)
        self._joints_accel_scale = p('joints_accel_scale',    0.2)
        self._mg_client = ActionClient(self, MoveGroup, '/move_action')
        # Mutex so concurrent plan_joints commands don't race
        self._plan_joints_lock = threading.Lock()

        # ── Socket server (background thread) ────────────────────────────────
        self._srv_thread = threading.Thread(target=self._socket_server, daemon=True)
        self._srv_thread.start()

        # ── Control loop ─────────────────────────────────────────────────────
        self.create_timer(1.0 / self._rate_hz, self._control_loop)

        # ── Diagnostics ──────────────────────────────────────────────────────
        self.create_timer(5.0, self._diag_loop)
        self._diag_solves          = 0
        self._diag_fails           = 0
        self._diag_snaps           = 0
        self._diag_vel_clamps      = 0
        self._diag_jump_rejects    = 0
        self._diag_deadbands       = 0
        self._diag_pos_only_fb     = 0
        self._diag_solve_ms_sum    = 0.0
        self._diag_solve_ms_max    = 0.0
        self._diag_last_seed       = ''
        self._ik_fail_log_time: float = 0.0

        # ── Per-tick debug CSV log ──────────────────────────────────────────
        debug_log_path = p('debug_log', '')
        self._csv_file = None
        self._csv_writer = None
        if debug_log_path:
            os.makedirs(os.path.dirname(debug_log_path) or '.', exist_ok=True)
            self._csv_file = open(debug_log_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow([
                't_mono',                          # monotonic timestamp (s)
                'dt_ms',                           # time since tick start (ms) — solve time
                'age_ms',                          # time since last socket msg (ms)
                # Target pose from socket (before offset)
                'tgt_x', 'tgt_y', 'tgt_z',
                'tgt_qx', 'tgt_qy', 'tgt_qz', 'tgt_qw',
                # Target quaternion after ee_ori_offset
                'off_qx', 'off_qy', 'off_qz', 'off_qw',
                # IK outcome
                'result',                          # 6D / POS_ONLY / SNAP / FAIL / JUMP_REJECT / JUMP_REJECT/pos_only
                'seed',                            # winning seed label
                'max_jump_rad',                    # largest joint delta from last_solved
                'vel_clamped',                     # 1 if velocity limiter fired
                # Solved joints
                'j1', 'j2', 'j3', 'j4', 'j5', 'j6',
                # Current joint_states (hardware feedback)
                'js1', 'js2', 'js3', 'js4', 'js5', 'js6',
                # FK of solution
                'fk_x', 'fk_y', 'fk_z',
                'fk_err_mm',                       # position error (mm)
                # Orientation error (degrees) — total + per-axis
                'ori_err_deg', 'ori_rx', 'ori_ry', 'ori_rz',
            ])
            self.get_logger().info(f'Debug CSV → {debug_log_path}')

        self.get_logger().info(
            f'ik_teleop_node ready -- listening on {self._host}:{self._port}  '
            f'rate={self._rate_hz:.0f}Hz  traj_dur={self._traj_dur*1000:.0f}ms  '
            f'ctrl_ori={self._ctrl_ori}')

    # ── Joint state subscriber ────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_pos[name] = pos

    # ── TCP socket server ─────────────────────────────────────────────────────

    def _socket_server(self):
        for attempt in range(10):
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self._host, self._port))
                break
            except OSError as e:
                srv.close()
                self.get_logger().warn(
                    f'Socket bind failed ({e}) -- retry {attempt+1}/10 in 1 s')
                time.sleep(1.0)
        else:
            self.get_logger().error(
                f'Could not bind {self._host}:{self._port} after 10 attempts.')
            return

        with srv:
            srv.listen(1)
            srv.settimeout(1.0)
            self.get_logger().info(f'TCP socket listening on {self._host}:{self._port}')
            while rclpy.ok():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                self.get_logger().info(f'arm_vision connected: {addr}')
                self._handle_client(conn, addr)

    def _handle_client(self, conn: socket.socket, addr):
        buf = ''
        # Spawn feedback sender — writes FK state back to the Mac at ~10 Hz
        fb_thread = threading.Thread(
            target=self._feedback_sender, args=(conn,), daemon=True)
        fb_thread.start()
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
        # Clear stale feedback so next client doesn't see old state
        with self._fk_feedback_lock:
            self._fk_feedback = {}
        self.get_logger().info(f'arm_vision disconnected: {addr}')
        with self._lock:
            self._target_pos         = None
            self._target_quat        = None
            self._last_solved_joints = None

    def _feedback_sender(self, conn: socket.socket):
        """Send FK feedback to the Mac at ~10 Hz on the existing TCP connection.

        Runs in a daemon thread spawned per client connection.  Dies silently
        when the connection is closed (OSError from sendall).
        """
        while True:
            time.sleep(0.1)
            with self._fk_feedback_lock:
                fb = dict(self._fk_feedback)
            if not fb:
                continue
            try:
                conn.sendall((json.dumps(fb) + '\n').encode('utf-8'))
            except OSError:
                break

    def _parse_message(self, line: str):
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Bad socket message: {exc}')
            return

        # ── Command messages ─────────────────────────────────────────────────
        if 'cmd' in msg:
            if msg['cmd'] == 'home':
                self.get_logger().info('Homing command received via TCP')
                with self._lock:
                    self._target_pos  = None   # stop IK loop from commanding old target
                    self._target_quat = None
                threading.Thread(target=self._run_homing, daemon=True).start()
            elif msg['cmd'] == 'joints':
                import math
                positions_rad = [math.radians(d) for d in msg.get('positions', [])]
                duration_s    = float(msg.get('duration', 2.0))
                if len(positions_rad) == len(self._joint_names):
                    self._publish_raw_joints(positions_rad, duration_s)
            elif msg['cmd'] == 'plan_joints':
                import math
                positions_rad = [math.radians(d) for d in msg.get('positions', [])]
                if len(positions_rad) == len(self._joint_names):
                    if self._plan_joints_lock.acquire(blocking=False):
                        threading.Thread(
                            target=self._run_plan_joints,
                            args=(positions_rad,),
                            daemon=True,
                        ).start()
                    else:
                        self.get_logger().warn('plan_joints: already active, ignoring')
            return   # not a pose message

        # ── Pose message ─────────────────────────────────────────────────────
        try:
            pos  = np.array([float(msg['x']),  float(msg['y']),  float(msg['z'])])
            quat = np.array([float(msg['qx']), float(msg['qy']),
                             float(msg['qz']), float(msg['qw'])])
            n = np.linalg.norm(quat)
            if n < 1e-6:
                return
            quat /= n
            with self._lock:
                self._target_pos     = pos
                self._target_quat    = quat
                self._last_recv_time = time.monotonic()
        except (KeyError, ValueError) as exc:
            self.get_logger().warn(f'Bad pose message: {exc}')

    def _build_joints_goal(self, positions_rad: list) -> MoveGroup.Goal:
        """Build a MoveGroup joint-space goal from a list of joint angles (rad)."""
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name                      = 'arm'
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = float(self._joints_plan_time)
        req.max_velocity_scaling_factor     = float(self._joints_vel_scale)
        req.max_acceleration_scaling_factor = float(self._joints_accel_scale)

        constraints = Constraints()
        for name, pos in zip(self._joint_names, positions_rad):
            jc = JointConstraint()
            jc.joint_name     = name
            jc.position       = float(pos)
            jc.tolerance_above = 0.05   # ~3°
            jc.tolerance_below = 0.05
            jc.weight         = 1.0
            constraints.joint_constraints.append(jc)
        req.goal_constraints = [constraints]

        goal.request = req
        goal.planning_options.plan_only                          = False
        goal.planning_options.replan                             = False
        goal.planning_options.planning_scene_diff.is_diff        = True
        return goal

    def _run_plan_joints(self, positions_rad: list) -> None:
        """Plan and execute a joint-space goal via MoveIt (runs in a daemon thread).

        Sets planning_active=True in FK feedback while running so chassis_fsm
        can poll for completion using the same homing-flag pattern.
        Releases _plan_joints_lock when done.
        """
        self.get_logger().info(
            f'plan_joints: planning to [{", ".join(f"{r:.3f}" for r in positions_rad)}]')

        with self._fk_feedback_lock:
            fb = dict(self._fk_feedback)
            fb['planning_active'] = True
            fb.pop('planning_ok', None)
            fb.pop('planning_error_code', None)
            self._fk_feedback = fb

        try:
            if not self._mg_client.server_is_ready():
                self.get_logger().warn(
                    'plan_joints: /move_action not ready — falling back to raw joints')
                self._publish_raw_joints(positions_rad, self._joints_plan_time)
                return

            done   = threading.Event()
            result = [False]

            def on_goal_accepted(future):
                handle = future.result()
                if not handle.accepted:
                    self.get_logger().warn('plan_joints: goal rejected by MoveGroup')
                    with self._fk_feedback_lock:
                        fb = dict(self._fk_feedback)
                        fb['planning_ok'] = False
                        fb['planning_error_code'] = -1
                        self._fk_feedback = fb
                    done.set()
                    return
                handle.get_result_async().add_done_callback(on_result)

            def on_result(future):
                try:
                    code = future.result().result.error_code.val
                    result[0] = (code == MoveItErrorCodes.SUCCESS)
                    with self._fk_feedback_lock:
                        fb = dict(self._fk_feedback)
                        fb['planning_ok'] = result[0]
                        fb['planning_error_code'] = int(code)
                        self._fk_feedback = fb
                    if result[0]:
                        self.get_logger().info('plan_joints: motion complete')
                    else:
                        self.get_logger().warn(f'plan_joints: MoveGroup failed (code={code})')
                except Exception as exc:
                    self.get_logger().warn(f'plan_joints: result exception: {exc}')
                    with self._fk_feedback_lock:
                        fb = dict(self._fk_feedback)
                        fb['planning_ok'] = False
                        fb['planning_error_code'] = -2
                        self._fk_feedback = fb
                done.set()

            future = self._mg_client.send_goal_async(self._build_joints_goal(positions_rad))
            future.add_done_callback(on_goal_accepted)

            if not done.wait(timeout=30.0):
                self.get_logger().warn('plan_joints: timed out after 30 s')
        finally:
            self._plan_joints_lock.release()
            with self._fk_feedback_lock:
                fb = dict(self._fk_feedback)
                fb.pop('planning_active', None)
                self._fk_feedback = fb

    def _publish_raw_joints(self, positions: list, duration_s: float) -> None:
        """Publish a JointTrajectory directly from joint-space positions (radians)."""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = list(self._joint_names)
        pt = JointTrajectoryPoint()
        pt.positions      = list(positions)
        pt.time_from_start = Duration(
            sec=int(duration_s),
            nanosec=int((duration_s % 1) * 1e9),
        )
        traj.points = [pt]
        self._traj_pub.publish(traj)

    def _run_homing(self):
        """Spawn homing_node locally (Linux side) and report status via FK feedback."""
        self.get_logger().info('Homing: starting...')
        # Pre-empt any in-flight IK trajectory so the arm doesn't drift toward
        # the previous target during the 50 ms traj_duration window before the
        # homing subprocess takes control.
        with self._lock:
            js = dict(self._joint_pos)
        if len(js) == len(self._joint_names):
            hold = [js.get(n, 0.0) for n in self._joint_names]
            self._publish_raw_joints(hold, 0.15)   # 150 ms covers the 50 ms traj_dur
        # Mark homing active in feedback so Mac side pauses streaming
        with self._fk_feedback_lock:
            fb = dict(self._fk_feedback)
            fb['homing'] = True
            self._fk_feedback = fb
        try:
            result = subprocess.run(
                ['ros2', 'run', 'arm_hardware', 'homing_node'],
                timeout=30,
            )
            self.get_logger().info(f'Homing done (exit {result.returncode})')
        except subprocess.TimeoutExpired:
            self.get_logger().error('Homing timed out (30 s)')
        except Exception as exc:
            self.get_logger().error(f'Homing failed: {exc}')
        finally:
            # Read current joint state (after homing) to compute post-home FK
            with self._lock:
                js = dict(self._joint_pos)
            # Reset IK control-loop state so the first solve after homing seeds
            # from the actual post-home hardware joints, not the stale pre-home
            # last_solved.  Without this, pick_best can select the old pre-home
            # configuration (if it happens to be close to q_preferred) and the
            # velocity limiter then smoothly moves the arm AWAY from home.
            if js and len(js) == len(self._joint_names):
                post_home_q = [js.get(n, 0.0) for n in self._joint_names]
                with self._lock:
                    self._last_solved_joints = post_home_q
            # Also flush the EMA smoother — its buffered values are from before
            # homing and would cause lag or drift on the first post-home target.
            self._smoothed_pos  = None
            self._smoothed_quat = None
            self._prev_raw_pos  = None
            with self._fk_feedback_lock:
                fb = dict(self._fk_feedback)
                fb.pop('homing', None)
                # Provide post-home FK so Mac can resync ee_pos immediately
                if self._chain_data and js:
                    try:
                        q = np.array([js.get(n, 0.0) for n in self._joint_names])
                        p_home, _, _ = _fk_and_jac(self._chain_data, q)
                        fb.update({
                            'fk_x': float(p_home[0]),
                            'fk_y': float(p_home[1]),
                            'fk_z': float(p_home[2]),
                            'ik_ok': True,
                        })
                    except Exception:
                        pass
                self._fk_feedback = fb

    # ── IK solvers ────────────────────────────────────────────────────────────

    def _solve_ik_pos_only(
        self,
        target_pos: np.ndarray,
        seed: list,
        seed_label: str = '',
        log_fail: bool = False,
        max_iters: int = 500,
    ) -> Optional[list]:
        """3D position-only DLS IK.  Orientation is unconstrained."""
        q = np.array(seed, dtype=float)
        lowers, uppers = _joint_limits(self._chain_data)

        init_err_norm: Optional[float] = None
        init_fk_pos: Optional[tuple] = None
        J_norm_first: Optional[float] = None
        iters_used: int = 0
        stop_reason: str = 'max_iters'

        lam = 1e-3
        prev_err = float('inf')

        for it in range(max_iters):
            iters_used = it
            p_ee, _R, J_full = _fk_and_jac(self._chain_data, q)
            J    = J_full[:3, :]   # use only linear rows
            err  = target_pos - p_ee
            err_norm = float(np.linalg.norm(err))

            if init_err_norm is None:
                init_err_norm = err_norm
                init_fk_pos   = (float(p_ee[0]), float(p_ee[1]), float(p_ee[2]))
            if J_norm_first is None:
                J_norm_first = float(np.linalg.norm(J))

            if err_norm < 1e-3:
                return list(q)

            if err_norm >= prev_err:
                lam = min(lam * 10.0, 1e-1)
            else:
                lam = max(lam * 0.5, 1e-7)
            prev_err = err_norm

            JJT_reg = J @ J.T + lam * np.eye(3)     # 3×3, reused below
            J_dls   = J.T @ np.linalg.inv(JJT_reg)  # 6×3 DLS pseudoinverse
            dq      = J_dls @ err                    # primary task (identical to before)

            # Secondary: pull joints toward preferred config via null-space projection.
            # N = I - J†J has rank 3 in position-only mode; negligible in 6D.
            if self._nullspace_gain > 0.0:
                N   = np.eye(len(q)) - J_dls @ J    # 6×6 null-space projector
                dq += N @ (self._nullspace_gain * (self._q_preferred[:len(q)] - q))

            dq_norm = float(np.linalg.norm(dq))
            if dq_norm < 1e-9:
                stop_reason = 'stuck'
                break
            if dq_norm > 0.15:
                dq *= 0.15 / dq_norm
            q = np.clip(q + dq, lowers, uppers)

        if log_fail:
            p_final, _, _ = _fk_and_jac(self._chain_data, q)
            final_err = float(np.linalg.norm(target_pos - p_final))
            label = f'[{seed_label}]' if seed_label else ''
            ifk   = init_fk_pos or (0.0, 0.0, 0.0)
            self.get_logger().warn(
                f'IK fail {label}  '
                f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  '
                f'seed=[{", ".join(f"{v:+.2f}" for v in seed)}]  '
                f'seed_fk=({ifk[0]:.3f},{ifk[1]:.3f},{ifk[2]:.3f})  '
                f'init_err={init_err_norm:.4f}m  final_err={final_err:.4f}m  '
                f'iters={iters_used}  reason={stop_reason}  '
                f'J_norm={J_norm_first:.4f}')
        return None

    def _solve_ik_6d(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        seed: list,
        seed_label: str = '',
        log_fail: bool = False,
        max_iters: int = 500,
    ) -> Optional[list]:
        """6D pose DLS IK.  Minimises position + orientation error simultaneously.

        Orientation error uses the geometric (rotation-matrix) residual:
            R_err = R_tgt @ R_ee.T
            e_rot = 0.5 * [R_err[2,1]-R_err[1,2],
                           R_err[0,2]-R_err[2,0],
                           R_err[1,0]-R_err[0,1]]
        This is the axis-angle linearisation, exact for small errors.

        J4 180-degree configuration flips (wrist singularity when J5≈0) are
        handled by the control loop's max_joint_jump_rad guard, not by
        constraining the solver itself — which needs freedom to converge.

        ori_weight scales the orientation rows so you can tune position vs
        orientation tracking independently (default 1.0).
        """
        R_tgt = _quat_to_mat3(target_quat)
        q = np.array(seed, dtype=float)
        w = float(self._ori_weight)
        lowers, uppers = _joint_limits(self._chain_data)

        init_pos_err: Optional[float] = None
        init_ori_err: Optional[float] = None
        iters_used: int = 0
        stop_reason: str = 'max_iters'

        lam = 1e-3
        prev_err = float('inf')

        for it in range(max_iters):
            iters_used = it
            p_ee, R_ee, J = _fk_and_jac(self._chain_data, q)

            err_pos = target_pos - p_ee

            # Geometric orientation error (rotation-matrix residual)
            R_err = R_tgt @ R_ee.T
            err_rot = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])

            pos_norm = float(np.linalg.norm(err_pos))
            ori_norm = float(np.linalg.norm(err_rot))

            if init_pos_err is None:
                init_pos_err = pos_norm
                init_ori_err = ori_norm

            if pos_norm < 1e-3 and ori_norm < 0.035:   # 1 mm + ~2.0 deg
                return list(q)

            # Build weighted 6D error and scale angular Jacobian rows
            err_6 = np.concatenate([err_pos, w * err_rot])
            J_w   = J.copy()
            J_w[3:, :] *= w

            err_6d_norm = float(np.linalg.norm(err_6))
            if err_6d_norm >= prev_err:
                lam = min(lam * 10.0, 1e-1)
            else:
                lam = max(lam * 0.5, 1e-7)
            prev_err = err_6d_norm

            # Singularity-aware damping floor: when the Jacobian is ill-conditioned
            # (J5 ≈ 0 / EE z ≈ world z wrist singularity), enforce a minimum lambda
            # proportional to singularity depth to prevent numerically unstable dq.
            sigma_min = float(np.linalg.svd(J_w, compute_uv=False)[-1])
            if sigma_min < self._sing_threshold:
                sing_lam = self._sing_lam_max * (1.0 - sigma_min / self._sing_threshold) ** 2
                lam = max(lam, sing_lam)
                self.get_logger().debug(f'sing sigma_min={sigma_min:.4f} lam={lam:.2e}')

            dq = J_w.T @ np.linalg.solve(J_w @ J_w.T + lam * np.eye(6), err_6)
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm < 1e-9:
                stop_reason = 'stuck'
                break
            if dq_norm > 0.20:
                dq *= 0.20 / dq_norm
            q = np.clip(q + dq, lowers, uppers)

        # Near-miss acceptance: if the solver ran out of iterations but got
        # close, accept the result.  2mm + 3° is tight enough that orientation
        # tracking looks correct but loose enough to catch cases that stall
        # just above the primary convergence threshold.
        if stop_reason == 'max_iters':
            p_f, R_f, _ = _fk_and_jac(self._chain_data, q)
            R_ef = R_tgt @ R_f.T
            e_rf = 0.5 * np.array([R_ef[2,1]-R_ef[1,2], R_ef[0,2]-R_ef[2,0], R_ef[1,0]-R_ef[0,1]])
            final_pos = float(np.linalg.norm(target_pos - p_f))
            final_ori = float(np.linalg.norm(e_rf))
            if final_pos < 2e-3 and final_ori < 0.052:   # 2 mm + ~3.0 deg
                return list(q)

        if log_fail:
            p_f, R_f, _ = _fk_and_jac(self._chain_data, q)
            R_ef = R_tgt @ R_f.T
            e_rf = 0.5 * np.array([R_ef[2,1]-R_ef[1,2], R_ef[0,2]-R_ef[2,0], R_ef[1,0]-R_ef[0,1]])
            label = f'[{seed_label}]' if seed_label else ''
            self.get_logger().warn(
                f'IK6D fail {label}  '
                f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  '
                f'init_pos_err={init_pos_err:.4f}m  init_ori_err={init_ori_err:.4f}rad  '
                f'final_pos_err={np.linalg.norm(target_pos-p_f):.4f}m  '
                f'final_ori_err={np.linalg.norm(e_rf):.4f}rad  '
                f'iters={iters_used}  reason={stop_reason}')
        return None

    # ── IK helper: try a list of seeds ───────────────────────────────────────

    def _try_ik(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray],
        seeds: list,
        log_fail: bool = False,
        max_iters: int = 500,
        time_budget: float = 0.0,
        pick_best: bool = False,
    ) -> tuple[Optional[list], str]:
        """Try each (label, seed) pair; return (solution, winning_seed_label).

        time_budget: if > 0, stop trying seeds after this many seconds elapsed.
        pick_best:   if True, try ALL seeds within the time budget and return
                     the solution whose joints are closest to q_preferred
                     (L2 distance).  This prevents the solver from settling on
                     a J4=±180° solution when a J4≈0° solution also exists.
                     If False (default), return the first valid solution found.
        """
        t0 = time.monotonic() if time_budget > 0.0 else 0.0

        best_sol:   Optional[list] = None
        best_label: str            = ''
        best_dist:  float          = float('inf')

        for label, s in seeds:
            if time_budget > 0.0 and (time.monotonic() - t0) > time_budget:
                break
            if self._ctrl_ori and target_quat is not None:
                result = self._solve_ik_6d(
                    target_pos, target_quat, s, label, log_fail, max_iters=max_iters)
            else:
                result = self._solve_ik_pos_only(
                    target_pos, s, label, log_fail, max_iters=max_iters)
            if result is not None:
                if not pick_best:
                    return result, label
                dist = float(np.linalg.norm(
                    np.array(result) - self._q_preferred[:len(result)]))
                if dist < best_dist:
                    best_dist  = dist
                    best_sol   = result
                    best_label = label

        return best_sol, best_label

    # ── Snap-to-reachable fallback ────────────────────────────────────────────

    def _snap_to_reachable(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray],
        det_seeds: list,
        time_budget: float = 0.005,
    ) -> Optional[list]:
        """Binary-search along ray from current EE to target; return IK solution
        for the furthest reachable point found.

        Performance-critical: uses only 4 bisection steps (1/16 precision) with
        reduced IK iterations (100 max) and a strict time budget to prevent
        blocking the control loop.
        """
        if self._last_solved_joints is None:
            return None
        p_cur, _, _ = _fk_and_jac(self._chain_data,
                                   np.array(self._last_solved_joints))
        direction = target_pos - p_cur
        dist = float(np.linalg.norm(direction))
        if dist < 1e-4:
            return None   # already at (or past) the workspace boundary

        t0 = time.monotonic()

        # Use only last_solved seed for snap search (fastest, most relevant)
        snap_seeds = [('last_solved', list(self._last_solved_joints))]

        lo, hi = 0.0, 1.0
        best: Optional[list] = None
        for _ in range(4):
            if (time.monotonic() - t0) > time_budget:
                break
            mid = (lo + hi) / 2.0
            cand = p_cur + mid * direction
            sol, _ = self._try_ik(cand, target_quat, snap_seeds,
                                  log_fail=False, max_iters=100)
            if sol is not None:
                best = sol
                lo = mid   # can go further
            else:
                hi = mid   # too far, back off
        return best

    # ── CSV tick logger ──────────────────────────────────────────────────────

    def _log_csv_row(
        self, t_tick: float, solve_ms: float, age: float,
        tgt_pos, raw_quat, off_quat,
        result_type: str, seed: str,
        max_jump, vel_clamped: bool,
        result_joints, js_list,
    ):
        if self._csv_writer is None:
            return
        fk_x = fk_y = fk_z = fk_err = ''
        ori_err = ori_rx = ori_ry = ori_rz = ''
        if result_joints is not None and tgt_pos is not None:
            p_sol, R_sol, _ = _fk_and_jac(self._chain_data, np.array(result_joints))
            fk_x, fk_y, fk_z = f'{p_sol[0]:.5f}', f'{p_sol[1]:.5f}', f'{p_sol[2]:.5f}'
            fk_err = f'{np.linalg.norm(tgt_pos - p_sol)*1000:.2f}'
            if off_quat is not None and self._ctrl_ori:
                R_tgt = _quat_to_mat3(off_quat)
                R_e = R_tgt @ R_sol.T
                e_r = 0.5 * np.array([R_e[2,1]-R_e[1,2], R_e[0,2]-R_e[2,0], R_e[1,0]-R_e[0,1]])
                e_deg = e_r * 180.0 / np.pi
                ori_err = f'{float(np.linalg.norm(e_r))*180/np.pi:.2f}'
                ori_rx, ori_ry, ori_rz = f'{e_deg[0]:.2f}', f'{e_deg[1]:.2f}', f'{e_deg[2]:.2f}'

        def _qf(q):
            return [f'{q[i]:.6f}' for i in range(4)] if q is not None else [''] * 4
        def _jf(j):
            return [f'{v:.5f}' for v in j] if j is not None else [''] * 6
        def _pf(p):
            return [f'{p[i]:.5f}' for i in range(3)] if p is not None else [''] * 3

        self._csv_writer.writerow([
            f'{t_tick:.4f}', f'{solve_ms:.2f}', f'{age*1000:.1f}',
            *_pf(tgt_pos), *_qf(raw_quat), *_qf(off_quat),
            result_type, seed,
            f'{max_jump:.4f}' if max_jump is not None else '',
            '1' if vel_clamped else '0',
            *_jf(result_joints), *_jf(js_list),
            fk_x, fk_y, fk_z, fk_err,
            ori_err, ori_rx, ori_ry, ori_rz,
        ])

    # ── Control loop (30 Hz) ─────────────────────────────────────────────────

    def _control_loop(self):
        if not self._chain_data:
            return

        t_tick_start = time.monotonic()

        with self._lock:
            target_pos   = self._target_pos
            target_quat  = self._target_quat
            age          = t_tick_start - self._last_recv_time
            joint_pos    = dict(self._joint_pos)

        if target_pos is None or age > self._det_timeout:
            # Reset smoother on timeout so it doesn't lag when target reappears
            if age > self._det_timeout:
                self._smoothed_pos  = None
                self._smoothed_quat = None
                self._prev_raw_pos  = None
            return

        # ── EMA smoothing on target pose (velocity-adaptive) ─────────────────
        # When the target is moving fast → alpha ramps toward pose_alpha_moving
        #   (near 1.0) so the arm tracks without lag.
        # When the target is stationary → alpha = pose_alpha (heavy smoothing)
        #   to suppress IK-noise jitter.
        # Blend is linear in per-tick position delta scaled by pose_alpha_threshold_m.
        raw_delta = (float(np.linalg.norm(target_pos - self._prev_raw_pos))
                     if self._prev_raw_pos is not None else 0.0)
        self._prev_raw_pos = target_pos.copy()
        if self._pose_alpha < 1.0 or self._pose_alpha_moving < 1.0:
            t = min(raw_delta / max(self._pose_alpha_threshold_m, 1e-9), 1.0)
            a = (1.0 - t) * self._pose_alpha + t * self._pose_alpha_moving
            if self._smoothed_pos is None:
                self._smoothed_pos  = target_pos.copy()
                self._smoothed_quat = target_quat.copy() if target_quat is not None else None
            else:
                self._smoothed_pos = a * target_pos + (1.0 - a) * self._smoothed_pos
                if target_quat is not None and self._smoothed_quat is not None:
                    sq = a * target_quat + (1.0 - a) * self._smoothed_quat
                    n  = float(np.linalg.norm(sq))
                    self._smoothed_quat = sq / n if n > 1e-9 else target_quat
            target_pos  = self._smoothed_pos
            target_quat = self._smoothed_quat

        # Apply static EE frame offset (right-multiply) to compensate for the
        # mismatch between the teleop's zero orientation and the URDF EE frame.
        raw_quat = target_quat.copy() if target_quat is not None else None
        if target_quat is not None:
            target_quat = _quat_mul(target_quat, self._ee_ori_offset)
            n = float(np.linalg.norm(target_quat))
            if n > 1e-9:
                target_quat = target_quat / n

        if len(joint_pos) < self._n_joints:
            self.get_logger().warn_once(
                f'Waiting for /joint_states ({len(joint_pos)}/{self._n_joints})')
            return

        with self._lock:
            last_solved = self._last_solved_joints

        js_seed   = [joint_pos.get(n, 0.0) for n in self._joint_names]
        zero_seed = [0.0] * self._n_joints

        # Deterministic seeds (also used as fallback seeds inside snap search)
        det_seeds: list = []
        if last_solved is not None:
            det_seeds.append(('last_solved', list(last_solved)))
            # J4-reset seed: copy of last_solved with only J4 snapped to
            # q_preferred[3]=0.  Starts near the real robot pose (high
            # convergence probability) but with J4=0°, giving the solver a
            # reliable path to a J4≈0 solution that pick_best can then prefer.
            j4_reset    = list(last_solved)
            j4_reset[3] = float(self._q_preferred[3])
            det_seeds.append(('j4_reset', j4_reset))
        det_seeds.append(('joint_states', js_seed))
        det_seeds.append(('zero/home',    zero_seed))

        # Full seed list: deterministic + random
        seeds_to_try = list(det_seeds)
        lowers, uppers = _joint_limits(self._chain_data)
        for _ in range(self._n_random_seeds):
            seeds_to_try.append(('random', list(np.random.uniform(lowers, uppers))))

        should_log = (t_tick_start - self._ik_fail_log_time) > 2.0

        # Time budget: at 60Hz each tick has ~16ms.  Reserve ~4ms for snap fallback.
        tick_budget = 1.0 / self._rate_hz
        # Cap per-seed iterations for 6D IK so multiple seeds fit within
        # the time budget.  200 iters is enough for frame-to-frame deltas
        # with the relaxed convergence threshold (0.035 rad).
        ik_iters = 200 if (self._ctrl_ori and target_quat is not None) else 500
        result, winning_seed = self._try_ik(
            target_pos, target_quat, seeds_to_try, log_fail=False,
            max_iters=ik_iters, time_budget=tick_budget * 0.7,
            pick_best=True)

        # Reject solutions that require a large jump from the last commanded position.
        # Catches configuration flips (e.g. J4 ±180°) that are geometrically valid
        # but would violently snap the arm mid-motion.
        # Only active after the first solve — initial positioning is unrestricted.
        #
        # Exception: when control_orientation is active, 6D IK solutions are
        # allowed through even with large jumps — the velocity limiter
        # (max_joint_vel_rad_s) smooths the transition.  Hard-rejecting forces
        # pos-only fallback, which locks the arm in a configuration that can
        # never match the target orientation.
        jump_rejected = False
        is_6d_solve = self._ctrl_ori and target_quat is not None
        if result is not None and self._max_joint_jump > 0.0 and last_solved is not None:
            max_jump = max(abs(r - c) for r, c in zip(result, last_solved))
            if max_jump > self._max_joint_jump:
                if is_6d_solve:
                    self.get_logger().debug(
                        f'[ik] 6D jump allowed  max_jump={max_jump:.3f}rad  '
                        f'seed={winning_seed}  vel_limiter will smooth')
                else:
                    self.get_logger().debug(
                        f'[ik] jump_reject  max_jump={max_jump:.3f}rad  '
                        f'seed={winning_seed}  threshold={self._max_joint_jump}')
                    self._log_csv_row(
                        t_tick_start, (time.monotonic() - t_tick_start) * 1000, age,
                        target_pos, raw_quat, target_quat,
                        'JUMP_REJECT', winning_seed, max_jump, False,
                        list(result), js_seed)
                    result = None
                    jump_rejected = True
                    self._diag_jump_rejects += 1

        # ── Position-only fallback when 6D IK fails ───────────────────────
        # The most common 6D failure is orientation unreachable (wrist singularity
        # or cube tilted to an unachievable angle).  Position is almost always
        # reachable.  Falling back to pos-only keeps the arm tracking smoothly
        # instead of going to 488mm fk_err.
        pos_only_fallback = False
        if result is None and self._ctrl_ori:
            fb_seeds = det_seeds[:2]   # last_solved + joint_states only (fast)
            result, winning_seed = self._try_ik(
                target_pos, None, fb_seeds, log_fail=False,
                time_budget=tick_budget * 0.2)
            if result is not None:
                pos_only_fallback = True
                winning_seed = winning_seed + '/pos_only'
                self._diag_pos_only_fb += 1
                # Warn periodically when pos-only fallback fires — the user
                # may not realise orientation is being silently dropped.
                if self._diag_pos_only_fb in (1, 10, 50) or self._diag_pos_only_fb % 100 == 0:
                    self.get_logger().warn(
                        f'[ik] ORI_UNREACHABLE → pos-only fallback  count={self._diag_pos_only_fb}'
                        f'  |  {_ik_diag_str(self._chain_data, last_solved, target_pos, target_quat)}')
                # Re-check jump guard for the pos-only solution
                if self._max_joint_jump > 0.0 and last_solved is not None:
                    max_jump = max(abs(r - c) for r, c in zip(result, last_solved))
                    if max_jump > self._max_joint_jump:
                        self._log_csv_row(
                            t_tick_start, (time.monotonic() - t_tick_start) * 1000, age,
                            target_pos, raw_quat, target_quat,
                            'JUMP_REJECT/pos_only', winning_seed, max_jump, False,
                            list(result), js_seed)
                        result = None
                        jump_rejected = True
                        self._diag_jump_rejects += 1

        used_snap = False
        if result is None:
            result = self._snap_to_reachable(target_pos, target_quat, det_seeds)
            if result is not None:
                used_snap = True
                winning_seed = 'snap'
                self._diag_snaps += 1
                p_snap, _, _ = _fk_and_jac(self._chain_data, np.array(result))
                snap_err_mm = float(np.linalg.norm(target_pos - p_snap) * 1000.0)
                self.get_logger().warn(
                    f'[ik] SNAP: position partially unreachable — moved to workspace edge'
                    f'  |  snap_err={snap_err_mm:.1f}mm from target'
                    f'  |  {_ik_diag_str(self._chain_data, last_solved, target_pos, target_quat)}')

        if result is None:
            if should_log:
                self._ik_fail_log_time = t_tick_start
                cause = ('jump_guard blocked all solutions' if jump_rejected
                         else 'position outside workspace' if not self._ctrl_ori
                         else 'position AND orientation both unreachable')
                self.get_logger().error(
                    f'[ik] FAIL: truly unreachable  cause={cause}'
                    f'  |  {_ik_diag_str(self._chain_data, last_solved, target_pos, target_quat)}')
            self._diag_fails += 1
            self._log_csv_row(
                t_tick_start, (time.monotonic() - t_tick_start) * 1000, age,
                target_pos, raw_quat, target_quat,
                'FAIL', '', None, False, None, js_seed)
            # Feedback: IK failed — include last known FK so Mac can resync boundary
            _fb: dict = {'ik_ok': False}
            if last_solved is not None:
                _p, _, _ = _fk_and_jac(self._chain_data, np.array(last_solved))
                _fb.update({'fk_x': float(_p[0]), 'fk_y': float(_p[1]),
                            'fk_z': float(_p[2])})
            with self._fk_feedback_lock:
                self._fk_feedback = _fb
            return

        # ── Joint-space velocity limiting ─────────────────────────────────
        # Smooths transitions when IK re-acquires after traversing an
        # unreachable zone.  Each joint is clamped to max_joint_vel per tick.
        pre_vel_max_jump = (max(abs(r - c) for r, c in zip(result, last_solved))
                            if last_solved is not None else None)
        vel_clamped = False
        if last_solved is not None and self._max_joint_vel > 0.0:
            max_step = self._max_joint_vel / self._rate_hz
            for i in range(len(result)):
                delta = result[i] - last_solved[i]
                if abs(delta) > max_step:
                    result[i] = last_solved[i] + max_step * (1.0 if delta > 0 else -1.0)
                    vel_clamped = True
        if vel_clamped:
            self._diag_vel_clamps += 1

        if last_solved is not None:
            max_delta = max(abs(r - c) for r, c in zip(result, last_solved))
            if max_delta < self._joint_deadband:
                self._diag_deadbands += 1
                # Don't log DEADBAND ticks — they're noise (no motion)
                return

        # Track IK solve time
        solve_ms = (time.monotonic() - t_tick_start) * 1000.0
        self._diag_solve_ms_sum += solve_ms
        self._diag_solve_ms_max = max(self._diag_solve_ms_max, solve_ms)
        self._diag_last_seed = winning_seed

        # Per-tick debug log (visible with `ros2 run --log-level debug`)
        self.get_logger().debug(
            f'[ik] solve={solve_ms:.1f}ms  seed={winning_seed}  '
            f'vel_clamp={vel_clamped}  snap={used_snap}  age={age*1000:.0f}ms  '
            f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})')

        self._publish_raw_joints(list(result), self._traj_dur)

        with self._lock:
            self._last_solved_joints = list(result)
        self._diag_solves += 1

        # Feedback: FK position of solved joints + IK quality flag + joint angles
        _p_sol, _, _ = _fk_and_jac(self._chain_data, np.array(result))
        with self._fk_feedback_lock:
            self._fk_feedback = {
                'fk_x': float(_p_sol[0]),
                'fk_y': float(_p_sol[1]),
                'fk_z': float(_p_sol[2]),
                'ik_ok': not used_snap,   # False when arm snapped to nearest reachable
                **{f'j{i+1}': float(result[i]) for i in range(len(result))},
            }

        # Determine result type label for CSV
        if pos_only_fallback:
            csv_type = 'POS_ONLY'
        elif used_snap:
            csv_type = 'SNAP'
        elif is_6d_solve:
            csv_type = '6D'
        else:
            csv_type = '3D'
        self._log_csv_row(
            t_tick_start, solve_ms, age,
            target_pos, raw_quat, target_quat,
            csv_type, winning_seed, pre_vel_max_jump, vel_clamped,
            result, js_seed)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _diag_loop(self):
        with self._lock:
            target_pos  = self._target_pos
            last_joints = self._last_solved_joints
            age = time.monotonic() - self._last_recv_time if self._last_recv_time else -1.0

        solves       = self._diag_solves;       self._diag_solves       = 0
        fails        = self._diag_fails;        self._diag_fails        = 0
        snaps        = self._diag_snaps;        self._diag_snaps        = 0
        vel_clamps   = self._diag_vel_clamps;   self._diag_vel_clamps   = 0
        jump_rejects = self._diag_jump_rejects; self._diag_jump_rejects = 0
        deadbands    = self._diag_deadbands;    self._diag_deadbands    = 0
        pos_only_fb  = self._diag_pos_only_fb;  self._diag_pos_only_fb  = 0
        solve_ms_sum = self._diag_solve_ms_sum; self._diag_solve_ms_sum = 0.0
        solve_ms_max = self._diag_solve_ms_max; self._diag_solve_ms_max = 0.0
        last_seed    = self._diag_last_seed

        if target_pos is None:
            self.get_logger().info('[diag] No socket data yet')
            return

        avg_ms = solve_ms_sum / max(solves, 1)
        total = solves + fails
        fail_pct = (fails / max(total, 1)) * 100

        # FK to check position + orientation error between target and solved pose
        fk_err_str = ''
        ori_err_str = ''
        if last_joints is not None:
            p_solved, R_solved, _ = _fk_and_jac(self._chain_data, np.array(last_joints))
            fk_err = float(np.linalg.norm(target_pos - p_solved))
            fk_err_str = f'  fk_err={fk_err*1000:.1f}mm'

            # Orientation error (only meaningful when ctrl_ori is true)
            with self._lock:
                tgt_quat = self._target_quat
            if self._ctrl_ori and tgt_quat is not None:
                tgt_q = tgt_quat.copy()
                if self._ee_ori_offset is not None:
                    tgt_q = _quat_mul(tgt_q, self._ee_ori_offset)
                    n = float(np.linalg.norm(tgt_q))
                    if n > 1e-9:
                        tgt_q = tgt_q / n
                R_tgt = _quat_to_mat3(tgt_q)
                R_e = R_tgt @ R_solved.T
                e_r = 0.5 * np.array([R_e[2,1]-R_e[1,2], R_e[0,2]-R_e[2,0], R_e[1,0]-R_e[0,1]])
                ori_err_deg = float(np.linalg.norm(e_r)) * 180.0 / np.pi
                # Per-axis breakdown helps diagnose constant offsets (e.g. pitch=-30°)
                e_deg = e_r * 180.0 / np.pi
                ori_err_str = (f'  ori_err={ori_err_deg:.1f}°'
                               f'(rx={e_deg[0]:.1f} ry={e_deg[1]:.1f} rz={e_deg[2]:.1f})')

        self.get_logger().info(
            f'[diag] age={age:.2f}s  '
            f'solves={solves} fails={fails}({fail_pct:.0f}%)  '
            f'ik_avg={avg_ms:.1f}ms ik_max={solve_ms_max:.1f}ms  '
            f'snaps={snaps} pos_only={pos_only_fb} vel_clamps={vel_clamps} '
            f'jump_rej={jump_rejects} deadband={deadbands}  '
            f'seed={last_seed}{fk_err_str}{ori_err_str}  '
            f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = IkTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._csv_file is not None:
            node._csv_file.close()
            node.get_logger().info('Debug CSV closed.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
