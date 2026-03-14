#!/usr/bin/env python3
"""
IK-direct teleoperation node.

Receives 6D target poses from arm_vision over TCP.  Solves IK using PyKDL's
Levenberg-Marquardt solver (ChainIkSolverPos_LMA), seeded from /joint_states
for solution continuity.  No external IK package required — PyKDL ships with
ROS 2 Humble on all architectures including arm64.

Socket protocol: newline-delimited JSON
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Parameters:
  host                : bind host              (default: "0.0.0.0")
  port                : bind port              (default: 9999)
  base_frame          : robot base TF frame    (default: "base_link")
  ee_frame            : end-effector link name (default: "End_Effector")
  publish_rate_hz     : IK solve + publish rate Hz   (default: 30.0)
  ik_timeout_s        : unused (kept for param compatibility)
  traj_duration_s     : time_from_start in published JointTrajectory (default: 0.05)
  detection_timeout_s : hold if no TCP message for this long (default: 0.4)
  control_orientation : if False (default), use FK orientation from seed as target so
                        LMA only minimises position error — far more robust.  Set True
                        to enforce the incoming quaternion (full 6D), which requires
                        the target orientation to be consistent with the URDF model.
  robot_description   : URDF XML string (passed by launch file)
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional

import numpy as np
import PyKDL
from urdf_parser_py import urdf as urdf_parser

import rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.node import Node


# ── URDF → KDL chain (same helpers as socket_teleop_node) ────────────────────

def _pose_to_kdl_frame(pose) -> PyKDL.Frame:
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if pose is not None:
        if pose.xyz is not None:
            xyz = [float(v) for v in pose.xyz]
        if pose.rpy is not None:
            rpy = [float(v) for v in pose.rpy]
    return PyKDL.Frame(PyKDL.Rotation.RPY(*rpy), PyKDL.Vector(*xyz))


def _urdf_joint_to_kdl_joint(joint, joint_frame: PyKDL.Frame) -> PyKDL.Joint:
    axis = joint.axis if joint.axis is not None else [1.0, 0.0, 0.0]
    axis_vec = joint_frame.M * PyKDL.Vector(*[float(v) for v in axis])
    if joint.type in ('revolute', 'continuous'):
        return PyKDL.Joint(joint.name, joint_frame.p, axis_vec, PyKDL.Joint.RotAxis)
    if joint.type == 'prismatic':
        return PyKDL.Joint(joint.name, joint_frame.p, axis_vec, PyKDL.Joint.TransAxis)
    return PyKDL.Joint(joint.name, PyKDL.Joint.Fixed)


def _build_kdl_tree(robot_model) -> PyKDL.Tree:
    child_links  = {j.child  for j in robot_model.joints}
    parent_links = {j.parent for j in robot_model.joints}
    root_links   = sorted(parent_links - child_links)
    if len(root_links) != 1:
        raise RuntimeError(f'Expected one root link, got: {root_links}')
    root_link = root_links[0]
    tree = PyKDL.Tree(root_link)
    child_to_joint = {j.child: j for j in robot_model.joints}
    remaining = {link.name for link in robot_model.links if link.name != root_link}
    while remaining:
        progressed = False
        for name in list(remaining):
            joint = child_to_joint.get(name)
            if joint is None or joint.parent in remaining:
                continue
            frame   = _pose_to_kdl_frame(joint.origin)
            segment = PyKDL.Segment(name, _urdf_joint_to_kdl_joint(joint, frame), frame)
            tree.addSegment(segment, joint.parent)
            remaining.remove(name)
            progressed = True
        if not progressed:
            raise RuntimeError(f'Could not resolve URDF tree for: {remaining}')
    return tree


def _build_kdl_chain(robot_desc: str, base: str, tip: str) -> PyKDL.Chain:
    model = urdf_parser.URDF.from_xml_string(robot_desc.encode('utf-8'))
    return _build_kdl_tree(model).getChain(base, tip)


def _chain_joint_names(chain: PyKDL.Chain) -> list[str]:
    """Return names of non-fixed joints in chain order."""
    names = []
    for i in range(chain.getNrOfSegments()):
        seg = chain.getSegment(i)
        if seg.getJoint().getType() != PyKDL.Joint.Fixed:
            names.append(seg.getJoint().getName())
    return names


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
        robot_desc        = p('robot_description',   '')

        # ── Publisher ────────────────────────────────────────────────────────
        self._traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)

        self._joint_deadband = p('joint_deadband_rad', 0.001)   # suppress if change < 1 mrad

        # ── Shared state ─────────────────────────────────────────────────────
        self._lock               = threading.Lock()
        self._joint_pos: dict    = {}
        self._target_pos: Optional[np.ndarray]  = None
        self._target_quat: Optional[np.ndarray] = None
        self._last_recv_time: float = 0.0
        self._last_solved_joints: Optional[list] = None

        # ── /joint_states subscriber ──────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        # ── PyKDL solvers ─────────────────────────────────────────────────────
        # Primary (ctrl_ori=False): custom DLS iterative position-only IK.
        #   Uses ChainFkSolverPos_recursive + ChainJntToJacSolver.
        #   Minimises ||FK(q).p - target_pos||² with orientation unconstrained.
        #   Seeded from last solved joints → minimum joint motion each step.
        #
        # Fallback (ctrl_ori=True): PyKDL LMA with relaxed epsilon (1e-3).
        #   Enforces incoming quaternion.  Only use once camera orientation is
        #   calibrated to match the URDF kinematic model.
        #
        # NOTE: PyKDL LMA L-weights constructor crashes (SIGSEGV) on this
        #   platform regardless of argument type (list or numpy array).  Do NOT
        #   call ChainIkSolverPos_LMA(chain, L, ...) here.
        self._fk_solver:   Optional[PyKDL.ChainFkSolverPos_recursive] = None
        self._jac_solver:  Optional[PyKDL.ChainJntToJacSolver] = None
        self._ik_solver:   Optional[PyKDL.ChainIkSolverPos_LMA] = None  # 6D only
        self._joint_names: list[str] = []
        self._n_joints:    int = 0

        if robot_desc:
            try:
                chain = _build_kdl_chain(robot_desc, self._base_frame, self._ee_frame)
                self._fk_solver   = PyKDL.ChainFkSolverPos_recursive(chain)
                self._jac_solver  = PyKDL.ChainJntToJacSolver(chain)
                self._joint_names = _chain_joint_names(chain)
                self._n_joints    = chain.getNrOfJoints()

                if self._ctrl_ori:
                    # LMA with relaxed epsilon so 1 mm position accuracy is
                    # enough to declare convergence (default 1e-5 is 10 µm —
                    # too tight, causes maxiter exhaustion).
                    self._ik_solver = PyKDL.ChainIkSolverPos_LMA(chain, 1e-3, 500, 1e-15)
                    self.get_logger().info('IK: LMA 6D (eps=1e-3)')
                else:
                    self.get_logger().info('IK: custom DLS position-only')

                self.get_logger().info(
                    f'KDL chain ready — {self._n_joints} joints  '
                    f'{self._base_frame}→{self._ee_frame}  '
                    f'rate={self._rate_hz:.0f}Hz  joints={self._joint_names}')
            except Exception as exc:
                self.get_logger().error(
                    f'KDL setup failed ({exc}) — node will not publish joint commands.')
        else:
            self.get_logger().error(
                'robot_description parameter is empty — node will not move the arm.')

        # ── Socket server (background thread) ────────────────────────────────
        self._srv_thread = threading.Thread(target=self._socket_server, daemon=True)
        self._srv_thread.start()

        # ── Control loop ─────────────────────────────────────────────────────
        self.create_timer(1.0 / self._rate_hz, self._control_loop)

        # ── Diagnostics ──────────────────────────────────────────────────────
        self.create_timer(5.0, self._diag_loop)
        self._diag_solves = 0
        self._diag_fails  = 0

        self.get_logger().info(
            f'ik_teleop_node ready — listening on {self._host}:{self._port}  '
            f'rate={self._rate_hz:.0f}Hz  traj_dur={self._traj_dur*1000:.0f}ms  '
            f'ctrl_ori={self._ctrl_ori}')

    # ── Joint state subscriber ────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_pos[name] = pos

    # ── TCP socket server ─────────────────────────────────────────────────────

    def _socket_server(self):
        # Retry bind in case a stale process is still releasing the port.
        for attempt in range(10):
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind((self._host, self._port))
                break
            except OSError as e:
                srv.close()
                self.get_logger().warn(
                    f'Socket bind failed ({e}) — retry {attempt+1}/10 in 1 s')
                time.sleep(1.0)
        else:
            self.get_logger().error(
                f'Could not bind {self._host}:{self._port} after 10 attempts — '
                f'kill any process holding port {self._port} and restart.')
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
            self._last_solved_joints = None   # reset seed so next session starts fresh

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

    # ── IK solve ─────────────────────────────────────────────────────────────

    def _solve_ik_pos_only(
        self,
        target_pos: np.ndarray,
        seed: list[float],
    ) -> Optional[list[float]]:
        """Position-only IK via damped Jacobian (DLS) iteration.

        Minimises the 3D position error ||FK(q).p - target_pos|| using the
        3×N position rows of the Jacobian.  Orientation is UNCONSTRAINED —
        the solver is free to choose any EE orientation, and because it is
        seeded from the current joint state it naturally picks the nearest
        configuration (minimum joint motion).

        This avoids the failure mode of the old approach (setting orientation
        target = FK(seed).M) which over-constrained the problem: as position
        moves away from the seed, maintaining the seed orientation often
        becomes geometrically infeasible.
        """
        q = np.array(seed, dtype=float)
        q_kdl  = PyKDL.JntArray(self._n_joints)
        jac_kdl = PyKDL.Jacobian(self._n_joints)
        fk_frame = PyKDL.Frame()

        for _ in range(100):
            for i, v in enumerate(q):
                q_kdl[i] = float(v)

            # Current EE position
            self._fk_solver.JntToCart(q_kdl, fk_frame)
            err = np.array([
                target_pos[0] - fk_frame.p.x(),
                target_pos[1] - fk_frame.p.y(),
                target_pos[2] - fk_frame.p.z(),
            ])

            if float(np.linalg.norm(err)) < 1e-3:   # 1 mm convergence
                return list(q)

            # Position Jacobian: 3 rows (linear velocity) × N cols (joints)
            self._jac_solver.JntToJac(q_kdl, jac_kdl)
            J = np.array([[jac_kdl[r, c] for c in range(self._n_joints)]
                          for r in range(3)])

            # DLS pseudoinverse: Δq = Jᵀ(JJᵀ + λI)⁻¹ Δp
            # λ=1e-4 gives near-exact pseudoinverse away from singularity
            # while bounding joint velocities near it.
            lam_sq = 1e-4
            dq = J.T @ np.linalg.solve(J @ J.T + lam_sq * np.eye(3), err)

            # Limit step size to avoid overshoot on large initial errors
            step = float(np.linalg.norm(dq))
            if step > 0.5:
                dq *= 0.5 / step

            q += dq

        return None   # did not converge in 100 iterations

    def _solve_ik(
        self,
        pos:  np.ndarray,
        quat: np.ndarray,
        seed: list[float],
    ) -> Optional[list[float]]:
        if not self._ctrl_ori:
            return self._solve_ik_pos_only(pos, seed)

        # Full 6D: enforce incoming quaternion via LMA
        if self._ik_solver is None:
            return None
        q_seed   = PyKDL.JntArray(self._n_joints)
        q_result = PyKDL.JntArray(self._n_joints)
        for i, v in enumerate(seed):
            q_seed[i] = float(v)
        rot    = PyKDL.Rotation.Quaternion(
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        target = PyKDL.Frame(
            rot, PyKDL.Vector(float(pos[0]), float(pos[1]), float(pos[2])))
        ret = self._ik_solver.CartToJnt(q_seed, target, q_result)
        if ret < 0:
            return None
        return [q_result[i] for i in range(self._n_joints)]

    # ── Control loop (30 Hz) ─────────────────────────────────────────────────

    def _control_loop(self):
        if self._fk_solver is None or self._jac_solver is None:
            return

        with self._lock:
            target_pos   = self._target_pos
            target_quat  = self._target_quat
            age          = time.monotonic() - self._last_recv_time
            joint_pos    = dict(self._joint_pos)

        if target_pos is None or age > self._det_timeout:
            return  # no target or stale — hold last commanded position

        if len(joint_pos) < self._n_joints:
            self.get_logger().warn_once(
                f'Waiting for /joint_states ({len(joint_pos)}/{self._n_joints})')
            return

        # Seed from last solved joints (better continuity) or fall back to /joint_states
        with self._lock:
            last_solved = self._last_solved_joints

        if last_solved is not None:
            seed = list(last_solved)
        else:
            seed = [joint_pos.get(n, 0.0) for n in self._joint_names]

        result = self._solve_ik(target_pos, target_quat, seed)

        # Retry with /joint_states seed if last-solved seed failed
        if result is None and last_solved is not None:
            seed = [joint_pos.get(n, 0.0) for n in self._joint_names]
            result = self._solve_ik(target_pos, target_quat, seed)

        if result is None:
            self._diag_fails += 1
            return

        # Suppress publish if every joint barely moved (absorbs detection jitter
        # when at target without sending redundant identical commands).
        if last_solved is not None:
            max_delta = max(abs(r - c) for r, c in zip(result, last_solved))
            if max_delta < self._joint_deadband:
                return

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

        solves, self._diag_solves = self._diag_solves, 0
        fails,  self._diag_fails  = self._diag_fails,  0

        if target_pos is None:
            self.get_logger().info('[diag] No socket data yet')
            return

        joints_str = (f'[{", ".join(f"{p:+.3f}" for p in last_joints)}]'
                      if last_joints else 'none')
        self.get_logger().info(
            f'[diag] socket_age={age:.1f}s  '
            f'solves={solves}  fails={fails}  '
            f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  '
            f'last_joints={joints_str}')


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
