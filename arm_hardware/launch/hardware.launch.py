#!/usr/bin/env python3
"""
Hardware launch: starts the UART bridge to the STM32 controller.

Args:
  uart_port   : serial device path, default /dev/ttyS3
  baud_rate   : serial baud rate, default 115200
  run_homing  : if true, run the sequential homing sequence on startup (default: false)
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = Path(get_package_share_directory('arm_hardware'))
    hw_params = str(pkg / 'config' / 'hardware_params.yaml')

    uart_port  = LaunchConfiguration('uart_port')
    baud_rate  = LaunchConfiguration('baud_rate')
    run_homing = LaunchConfiguration('run_homing')
    debug_tx   = LaunchConfiguration('debug_tx')
    debug_rx   = LaunchConfiguration('debug_rx')

    uart_bridge = Node(
        package='arm_hardware',
        executable='uart_bridge_node',
        name='uart_bridge_node',
        output='screen',
        parameters=[
            hw_params,
            {
                'port':                    uart_port,
                'baud_rate':               baud_rate,
                # Publish real encoder angles to /joint_states so MoveIt and
                # robot_state_publisher reflect actual hardware positions.
                # Requires control_node to remap its /joint_states output to
                # /mock_joint_states (handled in control.launch.py when
                # use_real_robot=true).
                'override_joint_states':   True,
                'debug_rx':                debug_rx,
                'debug_tx':                debug_tx,
            },
        ],
    )

    # Homing node: runs once then exits.
    # Delayed by 3 s to ensure arm_controller is fully active before sending goals.
    homing_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='arm_hardware',
                executable='homing_node',
                name='homing_node',
                output='screen',
                condition=IfCondition(run_homing),
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('uart_port',  default_value=''),
        DeclareLaunchArgument('baud_rate',  default_value='115200'),
        DeclareLaunchArgument('run_homing', default_value='false'),
        DeclareLaunchArgument('debug_rx',   default_value='false'),
        DeclareLaunchArgument('debug_tx',   default_value='false'),
        uart_bridge,
        homing_node,
    ])
