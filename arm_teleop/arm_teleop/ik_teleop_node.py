#!/usr/bin/env python3
"""
IK-direct teleoperation node.

Receives 6D target poses from arm_vision over TCP.  Solves IK using TRAC-IK
(SQP + random restarts with "Distance" mode) which reliably finds solutions
across the full workspace — including large initial displacements where KDL
Newton-Raphson fails.  Seeds each solve from /joint_states for solution
continuity.

Requires: sudo apt install ros-$ROS_DISTRO-trac-ik

Socket protocol: newline-delimited JSON
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Parameters:
  host                : bind host              (default: "0.0.0.0")
  port                : bind port              (default: 9999)
  base_frame          : robot base TF frame    (default: "base_link")
  ee_frame            : end-effector link name (default: "End_Effector")
  publish_rate_hz     : IK solve + publish rate Hz   (default: 30.0)
  ik_timeout_s        : time budget per IK solve     (default: 0.005)
  traj_duration_s     : time_from_start in published JointTrajectory (default: 0.05)
  detection_timeout_s : hold if no TCP message for this long (default: 0.4)
  robot_description   : URDF XML string (passed by launch file)
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional

import numpy as np
from trac_ik_python.trac_ik import IK as TracIK

import rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.node import Node


class IkTeleopNode(Node):

    def __init__(self):
        super().__init__('ik_teleop_node')

        def p(name, default): return self.declare_parameter(name, default).value

        self._host        = p('host',               '0.0.0.0')
        self._port        = p('port',               9999)
        self._base_frame  = p('base_frame',         'base_link')
        self._ee_frame    = p('ee_frame',            'End_Effector')
        self._rate_hz     = p('publish_rate_hz',     30.0)
        self._ik_timeout  = p('ik_timeout_s',        0.005)
        self._traj_dur    = p('traj_duration_s',     0.05)
        self._det_timeout = p('detection_timeout_s', 0.4)
        robot_desc        = p('robot_description',   '')

        # ── Publisher ────────────────────────────────────────────────────────
        self._traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)

        # ── Shared state ─────────────────────────────────────────────────────
        self._lock               = threading.Lock()
        self._joint_pos: dict    = {}
        self._target_pos: Optional[np.ndarray]  = None
        self._target_quat: Optional[np.ndarray] = None
        self._last_recv_time: float = 0.0
        self._last_solved_joints: Optional[list] = None    # diagnostics only

        # ── /joint_states subscriber ──────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        # ── TRAC-IK solver ────────────────────────────────────────────────────
        # "Distance" mode: returns the solution closest in joint space to the seed,
        # giving smooth motion during continuous tracking.
        self._ik_solver   = None
        self._joint_names = []   # derived from solver (guaranteed correct order)
        if robot_desc:
            try:
                self._ik_solver = TracIK(
                    self._base_frame,
                    self._ee_frame,
                    urdf_string=robot_desc,
                    timeout=self._ik_timeout,
                    epsilon=1e-5,
                    solve_type='Distance',
                )
                self._joint_names = list(self._ik_solver.joint_names)
                self.get_logger().info(
                    f'TRAC-IK solver ready — {len(self._joint_names)} joints  '
                    f'{self._base_frame}→{self._ee_frame}  '
                    f'rate={self._rate_hz:.0f}Hz  timeout={self._ik_timeout*1000:.0f}ms  '
                    f'joints={self._joint_names}')
            except Exception as exc:
                self.get_logger().error(
                    f'TRAC-IK setup failed ({exc}) — node will not publish joint commands.')
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
            f'rate={self._rate_hz:.0f}Hz  traj_dur={self._traj_dur*1000:.0f}ms')

    # ── Joint state subscriber ────────────────────────────────────────────────

    def _joint_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._joint_pos[name] = pos

    # ── TCP socket server ─────────────────────────────────────────────────────

    def _socket_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self._host, self._port))
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
            self._target_pos  = None
            self._target_quat = None

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

    # ── Control loop (30 Hz) ─────────────────────────────────────────────────

    def _control_loop(self):
        if self._ik_solver is None:
            return

        with self._lock:
            target_pos   = self._target_pos
            target_quat  = self._target_quat
            age          = time.monotonic() - self._last_recv_time
            joint_pos    = dict(self._joint_pos)

        if target_pos is None or age > self._det_timeout:
            return  # no target or stale — hold last commanded position

        # ── IK seed from /joint_states (solver order) ────────────────────────
        if len(joint_pos) < len(self._joint_names):
            self.get_logger().warn_once(
                f'Waiting for /joint_states ({len(joint_pos)}/'
                f'{len(self._joint_names)}) for IK seed')
            return
        seed = [joint_pos.get(n, 0.0) for n in self._joint_names]

        # ── Solve (TRAC-IK quaternion: x, y, z, w) ───────────────────────────
        result = self._ik_solver.get_ik(
            seed,
            float(target_pos[0]),  float(target_pos[1]),  float(target_pos[2]),
            float(target_quat[0]), float(target_quat[1]),
            float(target_quat[2]), float(target_quat[3]),
        )

        if result is None:
            self._diag_fails += 1
            self.get_logger().warn(
                f'TRAC-IK no solution  '
                f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})')
            return

        # ── Publish ──────────────────────────────────────────────────────────
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
