#!/usr/bin/env python3
"""
Hardware launch: starts the UART bridge to the STM32 controller.

Args:
  uart_port   : serial device path, default /dev/ttyS3
  baud_rate   : serial baud rate, default 115200
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = Path(get_package_share_directory('arm_hardware'))
    hw_params = str(pkg / 'config' / 'hardware_params.yaml')

    uart_port = LaunchConfiguration('uart_port')
    baud_rate = LaunchConfiguration('baud_rate')

    uart_bridge = Node(
        package='arm_hardware',
        executable='uart_bridge_node',
        name='uart_bridge_node',
        output='screen',
        parameters=[
            hw_params,
            {
                'port':      uart_port,
                'baud_rate': baud_rate,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('uart_port', default_value='/dev/ttyS3'),
        DeclareLaunchArgument('baud_rate', default_value='115200'),
        uart_bridge,
    ])
