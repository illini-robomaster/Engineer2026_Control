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

import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _kill_stale_ros_nodes(context):
    """Kill any lingering ROS 2 nodes from a previous session.

    Uses process-name matching (pkill without -f) for C++ executables so we
    don't accidentally kill the current launch process (whose full cmdline
    contains these names as arguments).  Python nodes are matched by their
    script name with -f, excluding our own process tree.
    """
    import os

    own_pgid = os.getpgrp()

    # C++ executables — safe to match by process name (no -f)
    native_targets = [
        'move_group',
        'servo_node_main',
        'robot_state_pub',   # truncated to 15 chars for comm matching
        'ros2_control_no',   # truncated to 15 chars
        'rviz2',
    ]
    for name in native_targets:
        subprocess.run(
            ['pkill', '-9', name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    # Python nodes — must use -f but exclude our own process group
    python_targets = [
        'uart_bridge_node',
        'socket_teleop_node',
        'ik_teleop_node',
        'moveit_teleop_node',
        'joint_monitor_node',
        'homing_node',
        'spawner',
    ]
    for name in python_targets:
        result = subprocess.run(
            ['pgrep', '-f', name], capture_output=True, text=True,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.strip().splitlines():
            pid = int(line)
            # Skip any process in our own process group (the new launch)
            try:
                if os.getpgid(pid) == own_pgid:
                    continue
                os.kill(pid, 9)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    # Stop the ROS 2 daemon so topic/service caches are fresh
    subprocess.run(
        ['ros2', 'daemon', 'stop'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # Purge FastRTPS shared-memory segments that cache stale DDS messages
    # (e.g. last-published /joint_states with transient-local durability).
    import glob as globmod
    for shm in globmod.glob('/dev/shm/fastrtps_*'):
        try:
            os.remove(shm)
        except OSError:
            pass

    return []


def generate_launch_description():
    arm_urdf_share = Path(get_package_share_directory('robotic_arm_v4_urdf'))
    arm_teleop_share = Path(get_package_share_directory('arm_teleop'))
    arm_hw_share = Path(get_package_share_directory('arm_hardware'))

    use_real_robot  = LaunchConfiguration('use_real_robot')
    uart_port       = LaunchConfiguration('uart_port')
    baud_rate       = LaunchConfiguration('baud_rate')
    use_teleop      = LaunchConfiguration('use_teleop')
    teleop_mode     = LaunchConfiguration('teleop_mode')
    socket_host     = LaunchConfiguration('socket_host')
    socket_port     = LaunchConfiguration('socket_port')
    use_moveit_rviz = LaunchConfiguration('use_moveit_rviz')
    print_joints    = LaunchConfiguration('print_joints')
    run_homing      = LaunchConfiguration('run_homing')
    debug_tx        = LaunchConfiguration('debug_tx')
    debug_rx        = LaunchConfiguration('debug_rx')
    control_orientation = LaunchConfiguration('control_orientation')
    ori_weight          = LaunchConfiguration('ori_weight')
    debug_log           = LaunchConfiguration('debug_log')

    # servo_node is only needed when teleop_mode=='servo'; ik_direct and moveit use
    # /arm_controller/joint_trajectory directly, so servo_node is not required.
    use_servo = PythonExpression(["'", teleop_mode, "' == 'servo'"])

    # ── MoveIt2 + ros2_control + servo_node + RViz ───────────────────────────
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(arm_urdf_share / 'launch' / 'moveit_control.launch.py')),
        launch_arguments={
            'use_moveit_rviz': use_moveit_rviz,
            'use_real_robot':  use_real_robot,
            'use_servo':       use_servo,
        }.items(),
    )

    # ── Socket teleop node (receives poses from arm_vision client) ────────────
    teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(arm_teleop_share / 'launch' / 'teleop.launch.py')),
        launch_arguments={
            'host':                socket_host,
            'port':                socket_port,
            'teleop_mode':         teleop_mode,
            'control_orientation': control_orientation,
            'ori_weight':          ori_weight,
            'debug_log':           debug_log,
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

    # ── Kill stale nodes from previous session before launching ────────────
    cleanup = OpaqueFunction(function=_kill_stale_ros_nodes)

    return LaunchDescription([
        cleanup,

        DeclareLaunchArgument('use_real_robot',  default_value='false'),
        DeclareLaunchArgument('uart_port',       default_value=''),
        DeclareLaunchArgument('baud_rate',       default_value='115200'),
        DeclareLaunchArgument('use_teleop',      default_value='true'),
        DeclareLaunchArgument('teleop_mode',     default_value='ik_direct',
            description='"ik_direct" (default, PyKDL LMA) | "servo" (DLS Jacobian) | "moveit" (OMPL, not for teleop)'),
        DeclareLaunchArgument('socket_host',     default_value='0.0.0.0'),
        DeclareLaunchArgument('socket_port',     default_value='9999'),
        DeclareLaunchArgument('use_moveit_rviz', default_value='true'),
        DeclareLaunchArgument('print_joints',    default_value='false'),
        DeclareLaunchArgument('run_homing',      default_value='false'),
        DeclareLaunchArgument('debug_tx',        default_value='false'),
        DeclareLaunchArgument('debug_rx',        default_value='false'),
        DeclareLaunchArgument('control_orientation', default_value='false',
            description='ik_direct mode: true=6D pose IK, false=position-only (default)'),
        DeclareLaunchArgument('ori_weight',      default_value='0.5',
            description='ik_direct 6D mode: orientation error weight (tune if pitch lags)'),
        DeclareLaunchArgument('debug_log',       default_value='',
            description='ik_direct CSV log path; forwarded to arm_teleop when teleop is enabled'),

        moveit_launch,
        teleop_launch,
        hardware_launch,
        joint_monitor_node,
    ])
