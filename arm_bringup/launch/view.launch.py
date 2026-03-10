#!/usr/bin/env python3
"""
view.launch.py — minimal read-only stack: view real arm joint angles in RViz.

Starts only:
  • robot_state_publisher   — broadcasts TF from /joint_states
  • uart_bridge_node        — publishes /joint_states from STM32 UART feedback
  • rviz2                   — shows the arm model following real angles

Does NOT start: arm_controller, move_group, servo_node, socket_teleop.
Use this mode to verify UART feedback and check homing without any motion
commands being sent to the hardware (uart_bridge TX is silenced in this
mode because no arm_controller is running to produce controller_state).

Launch arguments:
  uart_port   : serial device path  (default: /dev/ttyS3)
  baud_rate   : serial baud rate    (default: 115200)

Quick-start:
  ros2 launch arm_bringup view.launch.py uart_port:=/dev/ttyS4

  # Enable raw-frame debug logging:
  ros2 launch arm_bringup view.launch.py uart_port:=/dev/ttyCH341USB0 debug_rx:=true
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    urdf_pkg  = Path(get_package_share_directory('robotic_arm_v4_urdf'))
    hw_pkg    = Path(get_package_share_directory('arm_hardware'))

    urdf_text   = (urdf_pkg / 'urdf' / 'robotic_arm_v4_urdf.urdf').read_text()
    hw_params   = str(hw_pkg / 'config' / 'hardware_params.yaml')
    # Use the simple display.rviz (RobotModel + TF only, no MotionPlanning plugin).
    # moveit.rviz has the MotionPlanning display which freezes the model when
    # move_group is not running — that is the case in view-only mode.
    rviz_config = str(urdf_pkg / 'config' / 'display.rviz')

    uart_port    = LaunchConfiguration('uart_port')
    baud_rate    = LaunchConfiguration('baud_rate')
    debug_rx     = LaunchConfiguration('debug_rx')
    use_crc      = LaunchConfiguration('use_crc_framing')

    return LaunchDescription([
        DeclareLaunchArgument('uart_port',      default_value='/dev/ttyS3'),
        DeclareLaunchArgument('baud_rate',      default_value='115200'),
        DeclareLaunchArgument('debug_rx',       default_value='false'),
        DeclareLaunchArgument('use_crc_framing', default_value='true'),   # MCU uses $...*CCCC

        # TF broadcaster: /joint_states → /tf
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': urdf_text}],
            output='screen',
        ),

        # UART receiver: STM32 feedback → /joint_states + /real_joint_states
        # TX is silenced because no arm_controller publishes controller_state.
        Node(
            package='arm_hardware',
            executable='uart_bridge_node',
            name='uart_bridge_node',
            output='screen',
            parameters=[
                hw_params,
                {
                    'port':                  uart_port,
                    'baud_rate':             baud_rate,
                    'override_joint_states': True,   # own /joint_states
                    'debug_rx':              debug_rx,
                    'use_crc_framing':       use_crc,
                },
            ],
        ),

        # RViz: shows robot model following /joint_states (real arm angles)
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
    ])
