#!/usr/bin/env python3
"""
Top-level bringup: starts the complete teleoperation system.

┌─────────────────────────────────────────────────────────────────────┐
│  INPUT LAYER                                                         │
│  socket_teleop_node  ───────────────────────────────────────────────│
│    (receives 6D target pose from arm_vision via TCP socket)          │
│  keyboard_teleop_node  ─┐                                           │
│                          ├──► /servo_node/delta_twist_cmds           │
│                          └──► /servo_node/delta_joint_cmds          │
├─────────────────────────────────────────────────────────────────────┤
│  PLANNING LAYER (MoveIt2)                                            │
│  servo_node ──────────────► /arm_controller/joint_trajectory        │
│  move_group                                                          │
├─────────────────────────────────────────────────────────────────────┤
│  OUTPUT LAYER                                                        │
│  use_real_robot=false → mock_components (existing ros2_control)     │
│  use_real_robot=true  → uart_bridge_node → STM32 via UART           │
└─────────────────────────────────────────────────────────────────────┘

Launch arguments:
  use_real_robot   : true → start UART bridge; false → sim only (default: false)
  uart_port        : UART device path                         (default: /dev/ttyCH341USB0)
  baud_rate        : UART baud rate                           (default: 115200)
  use_teleop       : start socket_teleop_node (arm_vision pipeline) (default: true)
                     set false to use MoveIt Plan+Execute from RViz directly
  socket_host      : TCP bind address for pose socket         (default: 0.0.0.0)
  socket_port      : TCP bind port for pose socket            (default: 9999)
  use_moveit_rviz  : show MoveIt RViz panel                   (default: true)
  print_joints     : print live joint angles to terminal      (default: false)
  run_homing       : run parallel-group homing on startup (default: false)
                     only effective when use_real_robot:=true
                     groups: [Joint2+Joint3] → [Joint4+Joint5] → [Joint6+Joint1]

Quick-start (simulation only, Plan+Execute via RViz):
  ros2 launch arm_bringup arm_bringup.launch.py use_teleop:=false

With real robot, Plan+Execute only (no arm_vision):
  ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true use_teleop:=false

With real robot + arm_vision socket teleop:
  ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true

Print joint angles while using MoveIt (run in a SEPARATE terminal — cleaner output):
  source install/setup.bash && ros2 run arm_teleop joint_monitor_node
  # or inline with bringup:
  ros2 launch arm_bringup arm_bringup.launch.py print_joints:=true

Keyboard debug (run in a SEPARATE terminal after bringup is up):
  source install/setup.bash && ros2 run arm_teleop keyboard_teleop_node

Start arm_vision client (no ROS needed, separate terminal):
  cd arm_vision && python main.py run
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arm_urdf_share = Path(get_package_share_directory('robotic_arm_v4_urdf'))
    arm_teleop_share = Path(get_package_share_directory('arm_teleop'))
    arm_hw_share = Path(get_package_share_directory('arm_hardware'))

    use_real_robot  = LaunchConfiguration('use_real_robot')
    uart_port       = LaunchConfiguration('uart_port')
    baud_rate       = LaunchConfiguration('baud_rate')
    use_teleop      = LaunchConfiguration('use_teleop')
    socket_host     = LaunchConfiguration('socket_host')
    socket_port     = LaunchConfiguration('socket_port')
    use_moveit_rviz = LaunchConfiguration('use_moveit_rviz')
    print_joints    = LaunchConfiguration('print_joints')
    run_homing      = LaunchConfiguration('run_homing')
    debug_tx        = LaunchConfiguration('debug_tx')
    debug_rx        = LaunchConfiguration('debug_rx')

    # ── MoveIt2 + ros2_control + servo_node + RViz ───────────────────────────
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(arm_urdf_share / 'launch' / 'moveit_control.launch.py')),
        launch_arguments={
            'use_moveit_rviz': use_moveit_rviz,            'use_real_robot':  use_real_robot,        }.items(),
    )

    # ── Socket teleop node (receives poses from arm_vision client) ────────────
    teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(arm_teleop_share / 'launch' / 'teleop.launch.py')),
        launch_arguments={
            'host': socket_host,
            'port': socket_port,
        }.items(),
        condition=IfCondition(use_teleop),
    )

    # ── Joint monitor (optional — prints live joint angles and planned goal) ────
    joint_monitor_node = Node(
        package='arm_teleop',
        executable='joint_monitor_node',
        output='screen',
        condition=IfCondition(print_joints),
    )

    # ── UART bridge (real robot only) ─────────────────────────────────────────
    hardware_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(arm_hw_share / 'launch' / 'hardware.launch.py')),
        launch_arguments={
            'uart_port':  uart_port,
            'baud_rate':  baud_rate,
            'run_homing': run_homing,
            'debug_tx':   debug_tx,
            'debug_rx':   debug_rx,
        }.items(),
        condition=IfCondition(use_real_robot),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_real_robot',  default_value='false'),
        DeclareLaunchArgument('uart_port',       default_value=''),
        DeclareLaunchArgument('baud_rate',       default_value='115200'),
        DeclareLaunchArgument('use_teleop',      default_value='true'),
        DeclareLaunchArgument('socket_host',     default_value='0.0.0.0'),
        DeclareLaunchArgument('socket_port',     default_value='9999'),
        DeclareLaunchArgument('use_moveit_rviz', default_value='true'),
        DeclareLaunchArgument('print_joints',    default_value='false'),
        DeclareLaunchArgument('run_homing',      default_value='false'),
        DeclareLaunchArgument('debug_tx',        default_value='false'),
        DeclareLaunchArgument('debug_rx',        default_value='false'),

        moveit_launch,
        teleop_launch,
        hardware_launch,
        joint_monitor_node,
    ])
