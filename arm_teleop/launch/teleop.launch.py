#!/usr/bin/env python3
"""
Teleop launch: starts one of three teleop nodes based on teleop_mode.

  servo     (default) — socket_teleop_node: DLS Jacobian velocity control
  ik_direct           — ik_teleop_node:     inline PyKDL LMA IK, direct JointTrajectory
  moveit              — moveit_teleop_node:  MoveGroup action (OMPL plan + execute)

Args:
  host         : TCP bind address  (default: 0.0.0.0)
  port         : TCP bind port     (default: 9999)
  teleop_mode  : "servo" | "ik_direct" | "moveit"  (default: "servo")
  debug_log    : ik_direct CSV log path (default: "")

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

    urdf_path = (
        Path(get_package_share_directory('robotic_arm_v4_urdf'))
        / 'urdf' / 'robotic_arm_v4_urdf.urdf'
    )
    robot_description = {'robot_description': urdf_path.read_text()}

    host            = LaunchConfiguration('host')
    port            = LaunchConfiguration('port')
    teleop_mode     = LaunchConfiguration('teleop_mode')
    ctrl_ori        = LaunchConfiguration('control_orientation')
    ori_weight      = LaunchConfiguration('ori_weight')
    debug_log       = LaunchConfiguration('debug_log')
    use_moveit_ik   = LaunchConfiguration('use_moveit_ik')

    is_servo  = PythonExpression(["'", teleop_mode, "' == 'servo'"])
    is_ik     = PythonExpression(["'", teleop_mode, "' == 'ik_direct'"])
    is_moveit = PythonExpression(["'", teleop_mode, "' == 'moveit'"])

    socket_teleop = Node(
        package='arm_teleop',
        executable='socket_teleop_node',
        name='socket_teleop_node',
        output='screen',
        parameters=[
            teleop_params,
            robot_description,
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
            robot_description,
            {'host': host, 'port': port,
             'control_orientation': ctrl_ori,
             'ori_weight': ori_weight,
             'debug_log': debug_log,
             'use_moveit_ik': use_moveit_ik},
        ],
        condition=IfCondition(is_ik),
    )

    # moveit_teleop_node does not need robot_description — it talks to
    # move_group which already has the URDF/SRDF loaded.
    moveit_teleop = Node(
        package='arm_teleop',
        executable='moveit_teleop_node',
        name='moveit_teleop_node',
        output='screen',
        parameters=[
            teleop_params,
            {'host': host, 'port': port},
        ],
        condition=IfCondition(is_moveit),
    )

    return LaunchDescription([
        DeclareLaunchArgument('host', default_value='0.0.0.0',
                              description='TCP bind address for pose socket'),
        DeclareLaunchArgument('port', default_value='9999',
                              description='TCP bind port for pose socket'),
        DeclareLaunchArgument('teleop_mode', default_value='ik_direct',
                              description='"ik_direct" (default) | "servo" | "moveit"'),
        DeclareLaunchArgument('control_orientation', default_value='false',
                              description='ik_direct: true=6D pose IK, false=position-only'),
        DeclareLaunchArgument('ori_weight', default_value='0.5',
                              description='ik_direct 6D mode: orientation error weight'),
        DeclareLaunchArgument('debug_log', default_value='',
                              description='ik_direct CSV log path; empty disables file logging'),
        DeclareLaunchArgument('use_moveit_ik', default_value='false',
                              description='ik_direct: true=use MoveIt IK service, false=custom DLS'),
        socket_teleop,
        ik_teleop,
        moveit_teleop,
    ])
