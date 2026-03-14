#!/usr/bin/env python3
"""
MoveIt-planning teleoperation node.

Receives 6D target poses from arm_vision over TCP (same JSON protocol as
socket_teleop_node), then uses the /move_action server (MoveIt2 MoveGroup
action) to plan (OMPL) and execute collision-free trajectories.

USE CASE: Collision-aware reach to a STATIC or slowly-changing target.
          NOT suitable for fast real-time tracking — use ik_direct or servo
          mode instead.

IMPORTANT: kinematics.yaml must have  mode: global  (random restarts) so that
OMPL goal-tree sampling succeeds.  "local" (gradient descent) almost always
diverges from the random seeds OMPL uses and produces "Unable to sample any
valid states for goal tree".

Replanning is triggered when:
  - The target moves more than `replan_threshold_m` from the last plan goal, OR
  - No plan is in flight and the arm has not yet reached the target

Once a plan executes successfully, replanning is suppressed until the target
moves — this prevents OMPL from finding a different IK branch on the next
periodic replan and causing the arm to jump to a different joint configuration
for the same Cartesian pose.

Socket protocol: newline-delimited JSON
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Parameters:
  host                : bind host              (default: "0.0.0.0")
  port                : bind port              (default: 9999)
  base_frame          : planning frame         (default: "base_link")
  ee_link             : end-effector link      (default: "End_Effector")
  move_group_name     : MoveIt group name      (default: "arm")
  detection_timeout_s : cancel if no msg for N s  (default: 0.4)
  replan_threshold_m  : replan if target moves > N m  (default: 0.03)
  planning_time_s     : OMPL time budget per plan   (default: 1.0)
  velocity_scaling    : [0,1] scale on max joint vel (default: 0.5)
  acceleration_scaling: [0,1] scale on max joint acc (default: 0.3)
  pos_tolerance_m     : position goal tolerance     (default: 0.005)
  ori_tolerance_rad   : orientation goal tolerance  (default: 0.05)
  control_orientation : if False (default), send position-only goal — more
                        reliable since incoming quaternion may not be reachable.
                        Set True only after camera→robot orientation is validated.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive


class MoveitTeleopNode(Node):

    def __init__(self):
        super().__init__('moveit_teleop_node')

        def p(name, default): return self.declare_parameter(name, default).value

        self._host          = p('host',                '0.0.0.0')
        self._port          = p('port',                9999)
        self._base_frame    = p('base_frame',          'base_link')
        self._ee_link       = p('ee_link',             'End_Effector')
        self._group         = p('move_group_name',     'arm')
        self._det_timeout   = p('detection_timeout_s', 0.4)
        self._replan_thresh = p('replan_threshold_m',  0.03)
        self._plan_time     = p('planning_time_s',     1.0)
        self._vel_scale     = p('velocity_scaling',    0.5)
        self._accel_scale   = p('acceleration_scaling', 0.3)
        self._pos_tol       = p('pos_tolerance_m',     0.005)
        self._ori_tol       = p('ori_tolerance_rad',   0.05)
        self._ctrl_ori      = p('control_orientation', False)

        # ── MoveGroup action client ───────────────────────────────────────────
        self._mg_client = ActionClient(self, MoveGroup, '/move_action')

        # ── Shared state (guarded by _lock) ──────────────────────────────────
        self._lock               = threading.Lock()
        self._target_pos:  Optional[np.ndarray] = None
        self._target_quat: Optional[np.ndarray] = None
        self._last_recv_time: float = 0.0

        # Planning bookkeeping (only touched in the timer callback)
        self._planned_pos:   Optional[np.ndarray] = None
        self._goal_handle    = None
        self._goal_in_flight = False
        self._reached_target = False   # True after success; reset when target moves
        self._last_server_warn: float = 0.0

        # ── TCP socket server ─────────────────────────────────────────────────
        self._srv_thread = threading.Thread(target=self._socket_server, daemon=True)
        self._srv_thread.start()

        # ── Planning loop (2 Hz) ──────────────────────────────────────────────
        self.create_timer(0.5, self._planning_loop)

        # ── Diagnostics ───────────────────────────────────────────────────────
        self.create_timer(5.0, self._diag_loop)
        self._diag_plans = 0
        self._diag_ok    = 0
        self._diag_fail  = 0

        self.get_logger().info(
            f'moveit_teleop_node ready — {self._host}:{self._port}  '
            f'group={self._group}  ee={self._ee_link}  '
            f'plan_time={self._plan_time:.1f}s  '
            f'vel={self._vel_scale:.1f}  ctrl_ori={self._ctrl_ori}')

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
                    f'Socket bind failed ({e}) — retry {attempt+1}/10 in 1 s')
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
                self.get_logger().info(f'Client connected: {addr}')
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
        self.get_logger().info(f'Client disconnected: {addr}')
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

    # ── Planning loop (runs in ROS executor — safe to call action client) ─────

    def _planning_loop(self):
        with self._lock:
            target_pos  = self._target_pos
            target_quat = self._target_quat
            age         = time.monotonic() - self._last_recv_time

        if target_pos is None or age > self._det_timeout:
            self._cancel_current_goal()
            return

        target_moved = (
            self._planned_pos is None or
            np.linalg.norm(target_pos - self._planned_pos) > self._replan_thresh
        )

        if target_moved:
            # New target — reset "reached" flag and plan immediately.
            self._reached_target = False
        elif self._reached_target:
            # Already at this target.  Don't replan: OMPL is stochastic and would
            # find a different IK solution each run, causing the arm to jump between
            # joint configurations for the same Cartesian pose.
            return
        elif self._goal_in_flight:
            # Execution in progress for this target — don't interrupt.
            return

        self._cancel_current_goal()
        self._send_goal(target_pos, target_quat)
        self._planned_pos = target_pos.copy()
        self._diag_plans += 1

    # ── Goal construction ─────────────────────────────────────────────────────

    def _build_goal(self, pos: np.ndarray, quat: np.ndarray) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()

        req = MotionPlanRequest()
        req.group_name                      = self._group
        req.num_planning_attempts           = 5
        req.allowed_planning_time           = self._plan_time
        req.max_velocity_scaling_factor     = float(self._vel_scale)
        req.max_acceleration_scaling_factor = float(self._accel_scale)

        # Position constraint: sphere around target
        sphere = SolidPrimitive(type=SolidPrimitive.SPHERE,
                                dimensions=[self._pos_tol])
        center = Pose()
        center.position.x = float(pos[0])
        center.position.y = float(pos[1])
        center.position.z = float(pos[2])
        center.orientation.w = 1.0

        pc = PositionConstraint()
        pc.header.frame_id   = self._base_frame
        pc.link_name         = self._ee_link
        pc.constraint_region = BoundingVolume(primitives=[sphere],
                                              primitive_poses=[center])
        pc.weight = 1.0

        if self._ctrl_ori:
            oc = OrientationConstraint()
            oc.header.frame_id           = self._base_frame
            oc.link_name                 = self._ee_link
            oc.orientation.x             = float(quat[0])
            oc.orientation.y             = float(quat[1])
            oc.orientation.z             = float(quat[2])
            oc.orientation.w             = float(quat[3])
            oc.absolute_x_axis_tolerance = float(self._ori_tol)
            oc.absolute_y_axis_tolerance = float(self._ori_tol)
            oc.absolute_z_axis_tolerance = float(self._ori_tol)
            oc.weight                    = 0.5
            req.goal_constraints = [
                Constraints(position_constraints=[pc], orientation_constraints=[oc])
            ]
        else:
            req.goal_constraints = [Constraints(position_constraints=[pc])]

        goal.request = req
        goal.planning_options.plan_only          = False
        goal.planning_options.replan             = False
        goal.planning_options.planning_scene_diff.is_diff = True
        return goal

    def _send_goal(self, pos: np.ndarray, quat: np.ndarray):
        if not self._mg_client.server_is_ready():
            now = time.monotonic()
            if now - self._last_server_warn > 5.0:
                self.get_logger().warn('MoveGroup action server not available — waiting...')
                self._last_server_warn = now
            return

        future = self._mg_client.send_goal_async(self._build_goal(pos, quat))
        future.add_done_callback(self._on_goal_accepted)
        self._goal_in_flight = True

    def _on_goal_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('MoveGroup goal rejected')
            self._goal_in_flight = False
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        self._goal_in_flight = False
        self._goal_handle    = None
        try:
            code = future.result().result.error_code.val
        except Exception as exc:
            self.get_logger().warn(f'MoveGroup result exception: {exc}')
            self._diag_fail += 1
            return

        if code == MoveItErrorCodes.SUCCESS:
            self._diag_ok      += 1
            self._reached_target = True
        elif code in (MoveItErrorCodes.PREEMPTED, MoveItErrorCodes.CONTROL_FAILED):
            pass
        else:
            self._diag_fail += 1
            tgt = self._planned_pos
            self.get_logger().warn(
                f'MoveGroup failed  error_code={code}' +
                (f'  tgt=({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f})' if tgt is not None else ''))

    def _cancel_current_goal(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle    = None
            self._goal_in_flight = False
        self._reached_target = False

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def _diag_loop(self):
        with self._lock:
            target_pos = self._target_pos
            age = time.monotonic() - self._last_recv_time if self._last_recv_time else -1.0

        plans, self._diag_plans = self._diag_plans, 0
        ok,    self._diag_ok    = self._diag_ok,    0
        fail,  self._diag_fail  = self._diag_fail,  0

        if target_pos is None:
            self.get_logger().info('[diag] No socket data')
            return

        self.get_logger().info(
            f'[diag] age={age:.1f}s  plans={plans}  ok={ok}  fail={fail}  '
            f'in_flight={self._goal_in_flight}  reached={self._reached_target}  '
            f'tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})')


def main(args=None):
    rclpy.init(args=args)
    node = MoveitTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
