#!/usr/bin/env python3
"""
joint_monitor_node — prints joint angles live while the arm moves or a MoveIt plan is computed.

Subscribes:
  /joint_states                      (sensor_msgs/JointState)       → current arm state
  /move_group/display_planned_path   (moveit_msgs/DisplayTrajectory) → planned goal state

Output (stdout):
  [CURRENT]  Joint1=+xx.xx°  ...  Joint6=+xx.xx°   (updates in-place at each JointState msg)
  [PLAN GOAL] ...                                    (printed on its own line when a plan arrives)

Usage:
  # Standalone (recommended — run in its own terminal while bringup is active):
  ros2 run arm_teleop joint_monitor_node

  # Or via bringup launch argument (shares terminal output with other nodes):
  ros2 launch arm_bringup arm_bringup.launch.py print_joints:=true
"""

import math

import rclpy
from moveit_msgs.msg import DisplayTrajectory
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Ordered joint names matching the SRDF "arm" group
_JOINTS = ['Joint1', 'Joint2', 'Joint3', 'Joint4', 'Joint5', 'Joint6']

# ANSI colour helpers
_RESET  = '\033[0m'
_BOLD   = '\033[1m'
_CYAN   = '\033[96m'
_GREEN  = '\033[92m'


def _row(names: list[str], values_rad: list[float]) -> str:
    return '  '.join(
        f'{name}={math.degrees(v):+7.2f}\u00b0'
        for name, v in zip(names, values_rad)
    )


class JointMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__('joint_monitor_node')

        # Cache: joint_name → position (rad)
        self._current: dict[str, float] = {}
        # Track whether the live line has been printed at least once
        self._live_started = False

        self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10
        )
        self.create_subscription(
            DisplayTrajectory,
            '/move_group/display_planned_path',
            self._on_display_path,
            5,
        )

        print(f'\n{_BOLD}{_CYAN}=== Joint Monitor ==={_RESET}')
        print('Listening on  /joint_states  +  /move_group/display_planned_path')
        print(f'Press {_BOLD}Ctrl-C{_RESET} to stop.\n')

    # ------------------------------------------------------------------
    def _on_joint_states(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._current[name] = pos

        # Only print once all 6 tracked joints have been seen
        if not all(j in self._current for j in _JOINTS):
            return

        vals = [self._current[j] for j in _JOINTS]
        line = f'{_CYAN}[CURRENT]{_RESET}  {_row(_JOINTS, vals)}'

        if self._live_started:
            # Overwrite the previous live line in-place
            print(f'\r\033[K{line}', end='', flush=True)
        else:
            print(line, end='', flush=True)
            self._live_started = True

    # ------------------------------------------------------------------
    def _on_display_path(self, msg: DisplayTrajectory) -> None:
        if not msg.trajectory:
            return

        jt = msg.trajectory[-1].joint_trajectory
        if not jt.points:
            return

        last = jt.points[-1]
        name_to_pos = dict(zip(jt.joint_names, last.positions))

        if not all(j in name_to_pos for j in _JOINTS):
            return

        vals = [name_to_pos[j] for j in _JOINTS]

        # Print on a new line so it doesn't clobber the live line
        print(
            f'\n{_BOLD}{_GREEN}[PLAN GOAL]{_RESET}  {_row(_JOINTS, vals)}\n',
            flush=True,
        )
        # Next current-state print should start fresh (live line was interrupted)
        self._live_started = False


def main() -> None:
    rclpy.init()
    node = JointMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
