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

import json
import socket
import threading
import time
from typing import Optional

import numpy as np
from urdf_parser_py import urdf as urdf_parser

import rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.node import Node


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

        # ── Shared state ─────────────────────────────────────────────────────
        self._lock               = threading.Lock()
        self._joint_pos: dict    = {}
        self._target_pos: Optional[np.ndarray]  = None
        self._target_quat: Optional[np.ndarray] = None
        self._last_recv_time: float = 0.0
        self._last_solved_joints: Optional[list] = None

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
        self.get_logger().info(f'arm_vision disconnected: {addr}')
        with self._lock:
            self._target_pos         = None
            self._target_quat        = None
            self._last_solved_joints = None

    def _parse_message(self, line: str):
        if not line:
            return
        try:
            msg  = json.loads(line)
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
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'Bad socket message: {exc}')

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

            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
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

            if pos_norm < 1e-3 and ori_norm < 0.01:   # 1 mm + ~0.6 deg
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

            dq = J_w.T @ np.linalg.solve(J_w @ J_w.T + lam * np.eye(6), err_6)
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm < 1e-9:
                stop_reason = 'stuck'
                break
            if dq_norm > 0.15:
                dq *= 0.15 / dq_norm
            q = np.clip(q + dq, lowers, uppers)

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
    ) -> tuple[Optional[list], str]:
        """Try each (label, seed) pair; return (solution, winning_seed_label).

        time_budget: if > 0, stop trying seeds after this many seconds elapsed.
        """
        t0 = time.monotonic() if time_budget > 0.0 else 0.0
        for label, s in seeds:
            if time_budget > 0.0 and (time.monotonic() - t0) > time_budget:
                return None, ''
            if self._ctrl_ori and target_quat is not None:
                result = self._solve_ik_6d(
                    target_pos, target_quat, s, label, log_fail, max_iters=max_iters)
            else:
                result = self._solve_ik_pos_only(
                    target_pos, s, label, log_fail, max_iters=max_iters)
            if result is not None:
                return result, label
        return None, ''

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
            return

        # Apply static EE frame offset (right-multiply) to compensate for the
        # mismatch between the teleop's zero orientation and the URDF EE frame.
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
        result, winning_seed = self._try_ik(
            target_pos, target_quat, seeds_to_try, log_fail=False,
            time_budget=tick_budget * 0.7)

        # Reject solutions that require a large jump from the last commanded position.
        # Catches configuration flips (e.g. J4 ±180°) that are geometrically valid
        # but would violently snap the arm mid-motion.
        # Only active after the first solve — initial positioning is unrestricted.
        jump_rejected = False
        if result is not None and self._max_joint_jump > 0.0 and last_solved is not None:
            max_jump = max(abs(r - c) for r, c in zip(result, last_solved))
            if max_jump > self._max_joint_jump:
                self.get_logger().debug(
                    f'[ik] jump_reject  max_jump={max_jump:.3f}rad  '
                    f'seed={winning_seed}  threshold={self._max_joint_jump}')
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
                # Re-check jump guard for the pos-only solution
                if self._max_joint_jump > 0.0 and last_solved is not None:
                    max_jump = max(abs(r - c) for r, c in zip(result, last_solved))
                    if max_jump > self._max_joint_jump:
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

        if result is None:
            if should_log:
                self._ik_fail_log_time = t_tick_start
                self.get_logger().error(
                    f'IK: target truly unreachable  '
                    f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  '
                    f'ctrl_ori={self._ctrl_ori}  '
                    f'jump_rejected={jump_rejected}  '
                    f'Check: (1) target in workspace?  (2) orientation reachable?  '
                    f'(3) increase ori_weight if pos converges but ori does not')
            self._diag_fails += 1
            return

        # ── Joint-space velocity limiting ─────────────────────────────────
        # Smooths transitions when IK re-acquires after traversing an
        # unreachable zone.  Each joint is clamped to max_joint_vel per tick.
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

        traj              = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = list(self._joint_names)
        pt                = JointTrajectoryPoint()
        pt.positions      = list(result)
        pt.time_from_start = Duration(sec=0, nanosec=int(self._traj_dur * 1e9))
        traj.points = [pt]
        self._traj_pub.publish(traj)

        with self._lock:
            self._last_solved_joints = list(result)
        self._diag_solves += 1

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

        # FK to check position error between target and actual solved pose
        fk_err_str = ''
        if last_joints is not None:
            p_solved, _, _ = _fk_and_jac(self._chain_data, np.array(last_joints))
            fk_err = float(np.linalg.norm(target_pos - p_solved))
            fk_err_str = f'  fk_err={fk_err*1000:.1f}mm'

        joints_str = (f'[{", ".join(f"{p:+.3f}" for p in last_joints)}]'
                      if last_joints else 'none')
        self.get_logger().info(
            f'[diag] age={age:.2f}s  '
            f'solves={solves} fails={fails}({fail_pct:.0f}%)  '
            f'ik_avg={avg_ms:.1f}ms ik_max={solve_ms_max:.1f}ms  '
            f'snaps={snaps} pos_only={pos_only_fb} vel_clamps={vel_clamps} '
            f'jump_rej={jump_rejects} deadband={deadbands}  '
            f'seed={last_seed}{fk_err_str}  '
            f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = IkTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
