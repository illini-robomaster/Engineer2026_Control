#!/usr/bin/env python3
"""
homing_node — moves each arm joint to 0 rad in a configurable sequential order.

Sends a FollowJointTrajectory action goal for ALL joints per step (required by
the controller since allow_partial_joints_goal=false), moving only ONE joint
toward 0 at a time while holding all others at their current positions.  Waits
for action completion before proceeding to the next joint.

Default homing order: Joint2 → Joint1 → Joint3 → Joint4 → Joint5 → Joint6
Rationale: Joint2 (shoulder lift) is zeroed first so the arm tucks in, then
Joint1 (base rotation) returns to forward, then distal joints fold.

Parameters:
  homing_order       — joint names in desired order (default: [J2,J1,J3,J4,J5,J6])
  joint_speed_rad_s  — max joint speed used to compute step duration (default: 0.3)
  min_duration_s     — minimum time for any single step (default: 2.0 s)
  settle_time_s      — extra wait after each step for mechanical settling (default: 0.5 s)

Usage:
  # Standalone (recommended — run before starting teleop/vision):
  ros2 run arm_hardware homing_node

  # Via launch argument (starts right after arm_controller is active):
  ros2 launch arm_bringup arm_bringup.launch.py run_homing:=true
"""

import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

# ── Canonical joint order expected by the controller ──────────────────────────
_ALL_JOINTS = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']

# ── Default homing order ───────────────────────────────────────────────────────
_DEFAULT_ORDER = ['Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6', 'Joint1']

# ── Action server (JointTrajectoryController standard naming) ──────────────────
_ACTION_NAME = '/arm_controller/follow_joint_trajectory'


class HomingNode(Node):

    def __init__(self) -> None:
        super().__init__('homing_node')

        self.declare_parameter('homing_order', _DEFAULT_ORDER)
        self.declare_parameter('joint_speed_rad_s', 0.3)
        self.declare_parameter('min_duration_s', 2.0)
        self.declare_parameter('settle_time_s', 0.5)

        self._order   = list(self.get_parameter('homing_order').value)
        self._speed   = float(self.get_parameter('joint_speed_rad_s').value)
        self._min_dur = float(self.get_parameter('min_duration_s').value)
        self._settle  = float(self.get_parameter('settle_time_s').value)

        # Current joint positions (updated from /joint_states)
        self._current: dict[str, float] = {}

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10)

        self._action_client = ActionClient(
            self, FollowJointTrajectory, _ACTION_NAME)

    # ── Joint state callback ───────────────────────────────────────────────────

    def _js_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._current[name] = pos

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _wait_for_joint_states(self, timeout_s: float = 10.0) -> None:
        self.get_logger().info('Waiting for /joint_states…')
        deadline = time.monotonic() + timeout_s
        while not all(j in self._current for j in _ALL_JOINTS):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.monotonic() > deadline:
                raise RuntimeError(
                    'Timeout: /joint_states not received. '
                    'Is ros2_control running?')
        self.get_logger().info(
            'Current joints (deg): ' +
            '  '.join(f'{j}={math.degrees(self._current[j]):+.1f}°'
                      for j in _ALL_JOINTS))

    def _duration_for(self, delta_rad: float) -> float:
        """Compute step duration from angle distance and speed limit."""
        return max(abs(delta_rad) / self._speed, self._min_dur)

    # ── Single-joint homing step ───────────────────────────────────────────────

    def _home_one(self, joint_name: str) -> bool:
        """Move `joint_name` to 0, hold all others. Returns True on success."""
        current_pos = {j: self._current.get(j, 0.0) for j in _ALL_JOINTS}
        delta = current_pos[joint_name]            # distance to zero

        if abs(delta) < 0.005:                     # ~0.3° — already at zero
            self.get_logger().info(
                f'  {joint_name} already at zero ({math.degrees(delta):+.2f}°), skip.')
            return True

        duration_s = self._duration_for(delta)
        self.get_logger().info(
            f'  Homing {joint_name}: '
            f'{math.degrees(delta):+.1f}° → 0°  '
            f'(duration={duration_s:.1f} s)')

        # Build goal — all joints, only this one moves to 0
        target_pos = dict(current_pos)
        target_pos[joint_name] = 0.0

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(_ALL_JOINTS)

        # Point 0: current position, zero velocity, t=0
        p0 = JointTrajectoryPoint()
        p0.positions  = [current_pos[j] for j in _ALL_JOINTS]
        p0.velocities = [0.0] * 6
        p0.time_from_start = Duration(sec=0, nanosec=0)

        # Point 1: goal position, zero velocity, t=duration
        p1 = JointTrajectoryPoint()
        p1.positions  = [target_pos[j] for j in _ALL_JOINTS]
        p1.velocities = [0.0] * 6
        sec  = int(duration_s)
        nsec = int((duration_s - sec) * 1_000_000_000)
        p1.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal.trajectory.points = [p0, p1]

        # Send goal and wait for acceptance
        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                f'Goal rejected for {joint_name}! '
                'Is arm_controller active and not preempted?')
            return False

        # Wait for execution to finish
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().warn(
                f'  {joint_name}: action finished with error_code={result.error_code} '
                f'({result.error_string})')

        # Settle: let the hardware physically stop vibrating
        time.sleep(self._settle)

        # Refresh current positions
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info(
            f'  {joint_name} → done  '
            f'(now {math.degrees(self._current.get(joint_name, 0.0)):+.1f}°)')
        return True

    # ── Main sequence ──────────────────────────────────────────────────────────

    def run(self) -> None:
        self.get_logger().info('=' * 50)
        self.get_logger().info('Homing sequence starting')
        self.get_logger().info(f'Order: {" → ".join(self._order)}')
        self.get_logger().info('=' * 50)

        # Wait for joint state feedback
        self._wait_for_joint_states()

        # Wait for action server
        self.get_logger().info(f'Waiting for {_ACTION_NAME}…')
        if not self._action_client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error(
                f'Action server {_ACTION_NAME} not available. '
                'Is arm_controller running and active?')
            return

        # Sequential homing
        for joint in self._order:
            if joint not in _ALL_JOINTS:
                self.get_logger().warn(
                    f'Unknown joint "{joint}" in homing_order, skipping.')
                continue
            ok = self._home_one(joint)
            if not ok:
                self.get_logger().error(
                    f'Homing aborted at {joint}.')
                return

        self.get_logger().info('=' * 50)
        self.get_logger().info('Homing complete — all joints at 0 rad')
        self.get_logger().info('=' * 50)


def main() -> None:
    rclpy.init()
    node = HomingNode()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn('Homing interrupted by user.')
    except Exception as e:
        node.get_logger().error(f'Homing failed: {e}')
        raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
