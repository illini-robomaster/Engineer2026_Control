#!/usr/bin/env python3

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("robotic_arm_v4_urdf"))
    gazebo_share = Path(get_package_share_directory("gazebo_ros"))
    urdf_path = pkg_share / "urdf" / "robotic_arm_v4_urdf.urdf"

    robot_description = {"robot_description": urdf_path.read_text()}

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(gazebo_share / "launch" / "gazebo.launch.py"))
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[robot_description],
                output="screen",
            ),
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=["-topic", "robot_description", "-entity", "robotic_arm_v4_urdf"],
                output="screen",
            ),
        ]
    )
