#!/usr/bin/env python3
"""
Keyboard teleoperation node (debug / manual override).

Uses raw terminal input (no external deps). Run in a dedicated terminal.

── Cartesian mode (default) ─────────────────────────────────────────────────
  W / S       +X / -X  (forward / back)
  A / D       +Y / -Y  (left / right)
  Q / E       +Z / -Z  (up / down)
  I / K       rotate +Y / -Y
  J / L       rotate +Z / -Z
  U / O       rotate +X / -X
  [ / ]       decrease / increase speed

── Joint mode (press M to toggle) ───────────────────────────────────────────
  1 – 6       select joint (shown in status line)
  + / -       jog selected joint

── General ──────────────────────────────────────────────────────────────────
  M           toggle Cartesian / Joint mode
  Space       emergency stop (publish zero)
  H           go to named 'home' state via move_group
  Ctrl+C      quit
"""

import math
import sys
import termios
import threading
import tty

import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, WorkspaceParameters, RobotState
from rclpy.action import ActionClient
from rclpy.node import Node


HELP = """
── Cartesian (C) ──────────────  ── Joint (J) ───────────────────
  W/S  ±X   A/D  ±Y   Q/E  ±Z    1-6  select joint
  I/K  ±Ry  J/L  ±Rz  U/O  ±Rx   +/-  jog joint
  [/]  speed ±10 %
──────────────────────────────────────────────────────────────────
  M    toggle mode   Space  stop   H  home   Ctrl+C  quit
"""


class KeyboardTeleopNode(Node):

    def __init__(self):
        super().__init__('keyboard_teleop_node')

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('linear_speed', 0.08)
        self.declare_parameter('angular_speed', 0.3)
        self.declare_parameter('joint_speed', 0.3)
        self.declare_parameter('joint_names',
                               ['Joint1', 'Joint2', 'Joint3',
                                'Joint4', 'Joint5', 'Joint6'])

        self._base_frame  = self.get_parameter('base_frame').value
        self._lin_speed   = self.get_parameter('linear_speed').value
        self._ang_speed   = self.get_parameter('angular_speed').value
        self._jnt_speed   = self.get_parameter('joint_speed').value
        self._joint_names = self.get_parameter('joint_names').value

        self._twist_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self._jog_pub = self.create_publisher(
            JointJog, '/servo_node/delta_joint_cmds', 10)

        self._cart_mode = True      # True = Cartesian, False = Joint
        self._selected_joint = 0   # index into joint_names
        self._speed_scale = 1.0    # multiplier on speed params

        self._running = True
        self._key_thread = threading.Thread(target=self._read_keys, daemon=True)
        self._key_thread.start()

        # Zero-twist publisher at 10 Hz keeps servo alive when no key pressed
        self._zero_timer = self.create_timer(0.1, self._publish_zero)

        print(HELP)
        self._print_status()

    # ── Key reading ───────────────────────────────────────────────────────────

    def _read_keys(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self._running:
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                self._handle_key(ch)
        except Exception as e:
            self.get_logger().error(f'Key reader error: {e}')
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _handle_key(self, key: str):
        key = key.lower()

        # Quit
        if key == '\x03':   # Ctrl+C
            self._running = False
            rclpy.shutdown()
            return

        # Mode toggle
        if key == 'm':
            self._cart_mode = not self._cart_mode
            self._print_status()
            return

        # Emergency stop
        if key == ' ':
            self._publish_zero()
            return

        # Home (joint mode, uses move_group via CLI – just logs a hint)
        if key == 'h':
            self.get_logger().info(
                'Home: run  ros2 action send_goal /move_group '
                'moveit_msgs/action/MoveGroup '
                '"{request: {group_name: arm, goal_constraints: [...]}}"  '
                'or use the MoveIt RViz panel.')
            return

        # Speed scaling
        if key == '[':
            self._speed_scale = max(0.1, self._speed_scale - 0.1)
            self._print_status()
            return
        if key == ']':
            self._speed_scale = min(3.0, self._speed_scale + 0.1)
            self._print_status()
            return

        if self._cart_mode:
            self._cartesian_key(key)
        else:
            self._joint_key(key)

    # ── Cartesian commands ────────────────────────────────────────────────────

    _CART_MAP = {
        'w': ('linear',  'x',  1),
        's': ('linear',  'x', -1),
        'a': ('linear',  'y',  1),
        'd': ('linear',  'y', -1),
        'q': ('linear',  'z',  1),
        'e': ('linear',  'z', -1),
        'i': ('angular', 'y',  1),
        'k': ('angular', 'y', -1),
        'j': ('angular', 'z',  1),
        'l': ('angular', 'z', -1),
        'u': ('angular', 'x',  1),
        'o': ('angular', 'x', -1),
    }

    def _cartesian_key(self, key: str):
        if key not in self._CART_MAP:
            return
        kind, axis, sign = self._CART_MAP[key]
        speed = (self._lin_speed if kind == 'linear'
                 else self._ang_speed) * self._speed_scale * sign

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        setattr(getattr(msg.twist, kind), axis, speed)
        self._twist_pub.publish(msg)

    # ── Joint jog commands ────────────────────────────────────────────────────

    def _joint_key(self, key: str):
        if key in '123456':
            self._selected_joint = int(key) - 1
            self._print_status()
            return

        if key in ('+', '=', '-', '_'):
            sign = 1 if key in ('+', '=') else -1
            speed = self._jnt_speed * self._speed_scale * sign
            msg = JointJog()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._base_frame
            msg.joint_names = [self._joint_names[self._selected_joint]]
            msg.velocities = [speed]
            msg.duration = 0.1
            self._jog_pub.publish(msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_zero(self):
        if not self._cart_mode:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        self._twist_pub.publish(msg)

    def _print_status(self):
        mode = 'Cartesian' if self._cart_mode else (
            f'Joint  [Joint{self._selected_joint+1} selected]')
        print(f'\r[{mode}]  speed×{self._speed_scale:.1f}   ', end='', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
