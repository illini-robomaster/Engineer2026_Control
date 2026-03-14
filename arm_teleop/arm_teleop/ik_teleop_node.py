#!/usr/bin/env python3
"""
IK-direct teleoperation node.

Listens on the same TCP socket as socket_teleop_node.
Instead of running a continuous Cartesian velocity controller via servo_node,
calls MoveIt's /compute_ik service and publishes joint positions directly to
/arm_controller/joint_trajectory.  No servo_node needed.

Socket protocol: same as socket_teleop_node — newline-delimited JSON
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Parameters:
  host                : bind host              (default: "0.0.0.0")
  port                : bind port              (default: 9999)
  base_frame          : robot base TF frame    (default: "base_link")
  ee_frame            : end-effector link name (default: "End_Effector")
  ik_group_name       : MoveIt planning group  (default: "arm")
  publish_rate_hz     : max IK solve rate Hz   (default: 10.0)
  position_deadband_m : skip re-solve if target moved less than this (default: 0.005)
  ik_timeout_s        : IK service timeout     (default: 0.2)
  traj_duration_s     : time_from_start for the published JointTrajectory (default: 0.2)
  detection_timeout_s : hold position if no TCP message for this long   (default: 0.4)
  joint_names         : ordered joint list     (default: Joint1…Joint6)
"""

import json
import socket
import threading
import time
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class IkTeleopNode(Node):

    def __init__(self):
        super().__init__('ik_teleop_node')

        def p(name, default): return self.declare_parameter(name, default).value

        self._host        = p('host', '0.0.0.0')
        self._port        = p('port', 9999)
        self._base_frame  = p('base_frame', 'base_link')
        self._ee_frame    = p('ee_frame', 'End_Effector')
        self._group       = p('ik_group_name', 'arm')
        self._rate_hz     = p('publish_rate_hz', 10.0)
        self._deadband    = p('position_deadband_m', 0.005)
        self._ik_timeout  = p('ik_timeout_s', 0.2)
        self._traj_dur    = p('traj_duration_s', 0.2)
        self._det_timeout = p('detection_timeout_s', 0.4)
        self._joint_names = list(p('joint_names',
            ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']))

        # ── Publisher ────────────────────────────────────────────────────────
        self._traj_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)

        # ── IK service client ────────────────────────────────────────────────
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')

        # ── Shared state ─────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._joint_pos: dict = {}                        # name → rad (IK seed)
        self._target_pos: Optional[np.ndarray] = None     # [x, y, z]
        self._target_quat: Optional[np.ndarray] = None    # [x, y, z, w]
        self._last_recv_time: float = 0.0
        self._last_solved_pos: Optional[np.ndarray] = None  # target at last IK call
        self._last_solved_joints: Optional[list] = None   # joints from last IK solution
        self._ik_pending: bool = False
        self._ik_pending_since: float = 0.0              # monotonic time when pending set

        # ── Subscriber ───────────────────────────────────────────────────────
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)

        # ── Socket server (background thread) ────────────────────────────────
        self._srv_thread = threading.Thread(target=self._socket_server, daemon=True)
        self._srv_thread.start()

        # ── Control loop ─────────────────────────────────────────────────────
        self.create_timer(1.0 / self._rate_hz, self._control_loop)

        # ── Diagnostics ──────────────────────────────────────────────────────
        self.create_timer(5.0, self._diag_loop)

        self.get_logger().info(
            f'ik_teleop_node ready — listening on {self._host}:{self._port}  '
            f'group={self._group}  rate={self._rate_hz:.0f}Hz  '
            f'deadband={self._deadband * 1000:.0f}mm')

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
            self._target_pos = None
            self._target_quat = None
            self._last_solved_pos = None   # force fresh IK on next connect

    def _parse_message(self, line: str):
        if not line:
            return
        try:
            msg = json.loads(line)
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

    # ── Control loop (10 Hz) ──────────────────────────────────────────────────

    def _control_loop(self):
        with self._lock:
            target_pos   = self._target_pos
            target_quat  = self._target_quat
            age          = time.monotonic() - self._last_recv_time
            last_solved  = self._last_solved_pos
            joint_pos    = dict(self._joint_pos)

        if target_pos is None or age > self._det_timeout:
            return  # no target or stale — hold last commanded position

        if self._ik_pending:
            return  # previous IK call in flight

        # Skip if target didn't move enough since last solve
        if last_solved is not None:
            delta = float(np.linalg.norm(target_pos - last_solved))
            if delta < self._deadband:
                return
            self.get_logger().debug(f'target moved {delta*1000:.1f}mm → re-solving IK')

        if not self._ik_client.service_is_ready():
            self.get_logger().warn_once('/compute_ik service not yet available')
            return

        # Block until we have real joint feedback — seeding KDL at all-zero is a
        # singularity for most arm configurations and will reliably fail IK.
        if len(joint_pos) < len(self._joint_names):
            self.get_logger().warn_once(
                f'Waiting for /joint_states ({len(joint_pos)}/{len(self._joint_names)} joints) '
                f'before solving IK — seeding at 0 causes singularity errors')
            return

        # ── Build IK request ─────────────────────────────────────────────────
        req = GetPositionIK.Request()
        req.ik_request.group_name       = self._group
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.nanosec  = int(self._ik_timeout * 1e9)
        req.ik_request.attempts         = 10
        req.ik_request.ik_link_name     = self._ee_frame

        # Seed with current joint positions. All joints must be present (checked
        # above) so we never fall back to 0.0 which is near/at singularity.
        seed = req.ik_request.robot_state.joint_state
        seed.name     = list(self._joint_names)
        seed.position = [joint_pos[n] for n in self._joint_names]

        # Target pose
        ps = PoseStamped()
        ps.header.frame_id    = self._base_frame
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = float(target_pos[0])
        ps.pose.position.y    = float(target_pos[1])
        ps.pose.position.z    = float(target_pos[2])
        ps.pose.orientation.x = float(target_quat[0])
        ps.pose.orientation.y = float(target_quat[1])
        ps.pose.orientation.z = float(target_quat[2])
        ps.pose.orientation.w = float(target_quat[3])
        req.ik_request.pose_stamped = ps

        self.get_logger().info(
            f'IK request  tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  '
            f'seed=[{", ".join(f"{v:+.3f}" for v in seed.position)}]')
        self._ik_pending = True
        self._ik_pending_since = time.monotonic()
        future = self._ik_client.call_async(req)
        future.add_done_callback(lambda f: self._ik_done(f, target_pos.copy()))

    def _ik_done(self, future, solved_for: np.ndarray):
        self._ik_pending = False
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f'IK service call failed: {exc}')
            return

        if resp is None:
            return

        if resp.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().warn(
                f'IK no solution (code={resp.error_code.val}) for '
                f'({solved_for[0]:.3f}, {solved_for[1]:.3f}, {solved_for[2]:.3f})')
            return

        # Extract joint positions in the declared order
        sol = dict(zip(resp.solution.joint_state.name,
                       resp.solution.joint_state.position))
        positions = [sol.get(n, 0.0) for n in self._joint_names]

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names  = list(self._joint_names)
        pt = JointTrajectoryPoint()
        pt.positions       = positions
        pt.time_from_start = Duration(sec=0, nanosec=int(self._traj_dur * 1e9))
        traj.points = [pt]
        self._traj_pub.publish(traj)

        with self._lock:
            self._last_solved_pos = solved_for
            self._last_solved_joints = positions

        self.get_logger().info(
            f'IK solved  [{", ".join(f"{p:+.3f}" for p in positions)}] rad  '
            f'tgt=({solved_for[0]:.3f},{solved_for[1]:.3f},{solved_for[2]:.3f})')

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _diag_loop(self):
        with self._lock:
            target_pos        = self._target_pos
            last_solved_pos   = self._last_solved_pos
            last_joints       = self._last_solved_joints
            age = time.monotonic() - self._last_recv_time if self._last_recv_time else -1.0
        pending = self._ik_pending
        pending_age = (time.monotonic() - self._ik_pending_since) if pending else 0.0

        if pending and pending_age > 2.0:
            self.get_logger().warn(
                f'[diag] _ik_pending has been True for {pending_age:.1f}s — '
                f'callback may not be firing!')

        if target_pos is None:
            self.get_logger().info('[diag] No socket data yet')
        else:
            svc = 'ready' if self._ik_client.service_is_ready() else 'NOT READY'
            solved_str = (f'({last_solved_pos[0]:.3f},{last_solved_pos[1]:.3f},'
                          f'{last_solved_pos[2]:.3f})') if last_solved_pos is not None else 'none'
            joints_str = (f'[{", ".join(f"{p:+.3f}" for p in last_joints)}]'
                          ) if last_joints is not None else 'none'
            self.get_logger().info(
                f'[diag] socket_age={age:.1f}s  /compute_ik={svc}  pending={pending}  '
                f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  '
                f'last_solved_tgt={solved_str}  last_joints={joints_str}')


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
