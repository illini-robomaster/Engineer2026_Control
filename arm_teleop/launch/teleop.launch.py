#!/usr/bin/env python3
"""
Teleop launch: starts the socket_teleop_node.

The node listens on a TCP socket for absolute 6D end-effector target poses
sent by the standalone arm_vision client (see arm_vision/ in the repo root).

Args:
  host : TCP bind address  (default: 0.0.0.0)
  port : TCP bind port     (default: 9999)

NOTE — keyboard teleop must be run in a separate terminal (it needs a real TTY):
  source install/setup.bash
  ros2 run arm_teleop keyboard_teleop_node
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = Path(get_package_share_directory('arm_teleop'))
    teleop_params = str(pkg / 'config' / 'teleop_params.yaml')

    host = LaunchConfiguration('host')
    port = LaunchConfiguration('port')

    socket_teleop = Node(
        package='arm_teleop',
        executable='socket_teleop_node',
        name='socket_teleop_node',
        output='screen',
        parameters=[
            teleop_params,
            {'host': host, 'port': port},
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('host', default_value='0.0.0.0',
                              description='TCP bind address for pose socket'),
        DeclareLaunchArgument('port', default_value='9999',
                              description='TCP bind port for pose socket'),
        socket_teleop,
    ])
