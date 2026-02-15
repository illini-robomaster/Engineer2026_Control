#!/usr/bin/env python3

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("robotic_arm_v4_urdf"))
    urdf_path = pkg_share / "urdf" / "robotic_arm_v4_urdf.urdf"
    controllers_path = pkg_share / "config" / "controllers.yaml"
    rviz_path = pkg_share / "config" / "display.rviz"

    robot_description = {"robot_description": urdf_path.read_text()}
    use_rviz = LaunchConfiguration("use_rviz")

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, str(controllers_path)],
        output="screen",
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "120",
        ],
        output="screen",
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "arm_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "120",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[robot_description],
                output="screen",
            ),
            control_node,
            joint_state_broadcaster_spawner,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_broadcaster_spawner,
                    on_exit=[arm_controller_spawner],
                )
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(rviz_path)],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
