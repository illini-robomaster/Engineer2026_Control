#!/usr/bin/env python3

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_text(path: Path) -> str:
    return path.read_text()


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("robotic_arm_v4_urdf"))

    urdf_path = pkg_share / "urdf" / "robotic_arm_v4_urdf.urdf"
    srdf_path = pkg_share / "config" / "robotic_arm_v4.srdf"
    kinematics_path = pkg_share / "config" / "kinematics.yaml"
    joint_limits_path = pkg_share / "config" / "joint_limits.yaml"
    ompl_path = pkg_share / "config" / "ompl_planning.yaml"
    moveit_controllers_path = pkg_share / "config" / "moveit_controllers.yaml"
    moveit_rviz_path = pkg_share / "config" / "moveit.rviz"

    robot_description = {"robot_description": _load_text(urdf_path)}
    robot_description_semantic = {"robot_description_semantic": _load_text(srdf_path)}
    robot_description_kinematics = {"robot_description_kinematics": _load_yaml(kinematics_path)}
    robot_description_planning = {"robot_description_planning": _load_yaml(joint_limits_path)}

    planning_pipeline_config = _load_yaml(ompl_path)
    planning_pipeline_config["planning_pipelines"] = ["ompl"]
    planning_pipeline_config["default_planning_pipeline"] = "ompl"

    trajectory_execution = {
        "allow_trajectory_execution": True,
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    use_moveit_rviz = LaunchConfiguration("use_moveit_rviz")

    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(pkg_share / "launch" / "control.launch.py")),
        launch_arguments={"use_rviz": "false"}.items(),
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            planning_pipeline_config,
            trajectory_execution,
            planning_scene_monitor_parameters,
            _load_yaml(moveit_controllers_path),
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        condition=IfCondition(use_moveit_rviz),
        arguments=["-d", str(moveit_rviz_path)],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_moveit_rviz", default_value="true"),
            control_launch,
            TimerAction(period=2.0, actions=[move_group_node]),
            TimerAction(period=2.0, actions=[rviz_node]),
        ]
    )
