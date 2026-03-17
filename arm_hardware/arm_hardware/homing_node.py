#!/usr/bin/env python3
"""
homing_node — moves arm joints to 0 rad in a safe, collision-aware order.

Homing groups are chosen dynamically based on Joint2's current angle:

  J2 ≥ 0 (arm leaning forward):
    [J3+J4+J5+J6]  →  [J2]  →  [J1]
    Tuck all distal joints simultaneously first, then retract shoulder,
    finally rotate base home.

  J2 < 0 (arm leaning back):
    [J2+J4+J5+J6]  →  [J3]  →  [J1]
    Retract shoulder + wrist simultaneously, then fold elbow,
    finally rotate base home.

Within each group all listed joints move simultaneously; groups execute
sequentially.

Parameters:
  joint_speed_rad_s  — max joint speed used to compute step duration (default: 0.5)
  min_duration_s     — minimum time for any single group (default: 1.0 s)
  settle_time_s      — extra wait after each group for mechanical settling (default: 0.2 s)

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
from moveit_msgs.msg import AllowedCollisionMatrix, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

# ── Canonical joint order expected by the controller ──────────────────────────
_ALL_JOINTS = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']

# ── Homing group sequences (chosen at runtime based on J2 angle) ─────────────
_GROUPS_J2_NEG = [
    ['Joint2', 'Joint4', 'Joint5', 'Joint6'],  # retract shoulder + wrist simultaneously
    ['Joint3'],                                 # fold elbow
    ['Joint1'],                                 # base rotation last
]
_GROUPS_J2_POS = [
    ['Joint3', 'Joint4', 'Joint5', 'Joint6'],  # tuck all distal joints simultaneously
    ['Joint2'],                                 # retract shoulder
    ['Joint1'],                                 # base rotation last
]

# ── Action server (JointTrajectoryController standard naming) ──────────────────
_ACTION_NAME = '/arm_controller/follow_joint_trajectory'


class HomingNode(Node):

    def __init__(self) -> None:
        super().__init__('homing_node')

        self.declare_parameter('joint_speed_rad_s', 0.5)
        self.declare_parameter('min_duration_s', 1.0)
        self.declare_parameter('settle_time_s', 0.2)

        self._speed   = float(self.get_parameter('joint_speed_rad_s').value)
        self._min_dur = float(self.get_parameter('min_duration_s').value)
        self._settle  = float(self.get_parameter('settle_time_s').value)

        # Current joint positions (updated from /joint_states)
        self._current: dict[str, float] = {}
        self._homed: set[str] = set()   # joints already homed → always command 0
        self._saved_acm = None

        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10)

        self._action_client = ActionClient(
            self, FollowJointTrajectory, _ACTION_NAME)

        self._get_scene_cli   = self.create_client(GetPlanningScene,   '/get_planning_scene')
        self._apply_scene_cli = self.create_client(ApplyPlanningScene, '/apply_planning_scene')

    # ── Collision ACM helpers ─────────────────────────────────────────────────

    def _disable_collisions(self) -> None:
        """Save the current planning-scene ACM and replace it with an all-allow matrix."""
        if not self._get_scene_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('GetPlanningScene unavailable — skipping ACM save.')
            return
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        fut = self._get_scene_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if fut.result():
            self._saved_acm = fut.result().scene.allowed_collision_matrix
            self.get_logger().info(
                f'[ACM] Saved ({len(self._saved_acm.entry_names)} explicit entries). '
                'Disabling collision checking for pre-homing phase.')

        all_links = ['base_link', 'Link0', 'Link1', 'J3_J4', 'Link2', 'J5_J6', 'End_Effector']
        acm = AllowedCollisionMatrix()
        acm.default_entry_names  = all_links
        acm.default_entry_values = [True] * len(all_links)
        scene = PlanningScene(is_diff=True)
        scene.allowed_collision_matrix = acm

        if not self._apply_scene_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('ApplyPlanningScene unavailable — collision checking stays on.')
            return
        apply_fut = self._apply_scene_cli.call_async(ApplyPlanningScene.Request(scene=scene))
        rclpy.spin_until_future_complete(self, apply_fut, timeout_sec=5.0)
        self.get_logger().info('[ACM] All-allow matrix applied.')

    def _restore_collisions(self) -> None:
        """Restore the ACM saved by _disable_collisions."""
        if self._saved_acm is None:
            self.get_logger().warn('[ACM] No saved ACM — cannot restore collision checking.')
            return
        scene = PlanningScene(is_diff=True)
        scene.allowed_collision_matrix = self._saved_acm
        apply_fut = self._apply_scene_cli.call_async(ApplyPlanningScene.Request(scene=scene))
        rclpy.spin_until_future_complete(self, apply_fut, timeout_sec=5.0)
        self.get_logger().info('[ACM] Collision checking restored.')
        self._saved_acm = None

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

    # ── Group homing step ──────────────────────────────────────────────────────

    def _home_group(self, joint_names: list[str]) -> bool:
        """Move all joints in `joint_names` to 0 simultaneously, hold all others.
        Returns True on success."""
        current_pos = {j: self._current.get(j, 0.0) for j in _ALL_JOINTS}

        # Filter out joints that are already at zero
        to_move = [j for j in joint_names if abs(current_pos[j]) >= 0.005]
        if not to_move:
            self.get_logger().info(
                f'  {" + ".join(joint_names)} already at zero, skip.')
            return True

        # Duration driven by the joint that has the farthest to travel
        duration_s = max(self._duration_for(current_pos[j]) for j in to_move)

        label = ' + '.join(joint_names)
        self.get_logger().info(
            f'  Homing [{label}]  duration={duration_s:.1f}s  ' +
            '  '.join(f'{j}: {math.degrees(current_pos[j]):+.1f}°→0°' for j in to_move))

        target_pos = dict(current_pos)
        # Zero the joints in this group AND all previously-homed joints.
        # Without this, encoder lag from earlier groups (motor hasn't fully
        # converged to 0) would be re-commanded as the hold position,
        # effectively undoing earlier homing.
        for j in to_move:
            target_pos[j] = 0.0
        for j in self._homed:
            target_pos[j] = 0.0

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(_ALL_JOINTS)

        p0 = JointTrajectoryPoint()
        p0.positions  = [current_pos[j] for j in _ALL_JOINTS]
        p0.velocities = [0.0] * 6
        p0.time_from_start = Duration(sec=0, nanosec=0)

        p1 = JointTrajectoryPoint()
        p1.positions  = [target_pos[j] for j in _ALL_JOINTS]
        p1.velocities = [0.0] * 6
        sec  = int(duration_s)
        nsec = int((duration_s - sec) * 1_000_000_000)
        p1.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal.trajectory.points = [p0, p1]

        future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                f'Goal rejected for [{label}]! '
                'Is arm_controller active and not preempted?')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().warn(
                f'  [{label}]: error_code={result.error_code} ({result.error_string})')

        time.sleep(self._settle)

        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)

        # Mark these joints as homed so subsequent groups keep them at zero.
        self._homed.update(to_move)

        self.get_logger().info(
            '  done  ' + '  '.join(
                f'{j}={math.degrees(self._current.get(j, 0.0)):+.1f}°'
                for j in joint_names))
        return True

    # ── Main sequence ──────────────────────────────────────────────────────────

    def run(self) -> None:
        self.get_logger().info('=' * 50)
        self.get_logger().info('Homing sequence starting')
        self.get_logger().info('=' * 50)

        self._disable_collisions()
        try:
            self._wait_for_joint_states()

            # Choose group sequence based on J2 angle.
            j2_rad = self._current.get('Joint2', 0.0)
            j2_deg = math.degrees(j2_rad)
            if j2_rad >= 0.0:
                groups = _GROUPS_J2_POS
                self.get_logger().info(
                    f'J2 = {j2_deg:+.1f}° (≥ 0) → retract shoulder first')
            else:
                groups = _GROUPS_J2_NEG
                self.get_logger().info(
                    f'J2 = {j2_deg:+.1f}° (< 0) → fold elbow first')

            group_strs = [' + '.join(g) for g in groups]
            self.get_logger().info(
                'Groups: ' + '  →  '.join(f'[{s}]' for s in group_strs))

            self.get_logger().info(f'Waiting for {_ACTION_NAME}…')
            if not self._action_client.wait_for_server(timeout_sec=15.0):
                self.get_logger().error(
                    f'Action server {_ACTION_NAME} not available. '
                    'Is arm_controller running and active?')
                return

            for group in groups:
                ok = self._home_group(group)
                if not ok:
                    self.get_logger().error(
                        f'Homing aborted at group [{" + ".join(group)}].')
                    return

            self.get_logger().info('=' * 50)
            self.get_logger().info('Homing complete — all joints at 0 rad')
            self.get_logger().info('=' * 50)
        finally:
            self._restore_collisions()


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
