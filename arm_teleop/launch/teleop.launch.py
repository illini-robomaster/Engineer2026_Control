#!/usr/bin/env python3
"""
Teleop launch: starts socket_teleop_node (servo mode) or ik_teleop_node (ik_direct mode).

Args:
  host         : TCP bind address  (default: 0.0.0.0)
  port         : TCP bind port     (default: 9999)
  teleop_mode  : "servo" (default) or "ik_direct"

NOTE — keyboard teleop must be run in a separate terminal (it needs a real TTY):
  source install/setup.bash
  ros2 run arm_teleop keyboard_teleop_node
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = Path(get_package_share_directory('arm_teleop'))
    teleop_params = str(pkg / 'config' / 'teleop_params.yaml')

    host        = LaunchConfiguration('host')
    port        = LaunchConfiguration('port')
    teleop_mode = LaunchConfiguration('teleop_mode')

    is_servo    = PythonExpression(["'", teleop_mode, "' != 'ik_direct'"])
    is_ik       = PythonExpression(["'", teleop_mode, "' == 'ik_direct'"])

    socket_teleop = Node(
        package='arm_teleop',
        executable='socket_teleop_node',
        name='socket_teleop_node',
        output='screen',
        parameters=[
            teleop_params,
            {'host': host, 'port': port},
        ],
        condition=IfCondition(is_servo),
    )

    ik_teleop = Node(
        package='arm_teleop',
        executable='ik_teleop_node',
        name='ik_teleop_node',
        output='screen',
        parameters=[
            teleop_params,
            {'host': host, 'port': port},
        ],
        condition=IfCondition(is_ik),
    )

    return LaunchDescription([
        DeclareLaunchArgument('host', default_value='0.0.0.0',
                              description='TCP bind address for pose socket'),
        DeclareLaunchArgument('port', default_value='9999',
                              description='TCP bind port for pose socket'),
        DeclareLaunchArgument('teleop_mode', default_value='servo',
                              description='"servo" or "ik_direct"'),
        socket_teleop,
        ik_teleop,
    ])
