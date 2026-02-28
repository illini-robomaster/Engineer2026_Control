#!/usr/bin/env python3
"""
Teleop launch: starts apriltag_ros (camera → tag detection), the
AprilTag cube teleop node, and optionally the keyboard teleop node.

Args:
  camera_device   : USB video device, default /dev/video0
  camera_frame    : TF frame for the camera, default camera_link
  tag_size        : AprilTag size in metres, default 0.05
  use_keyboard    : also launch keyboard_teleop_node, default false
  use_rviz        : show camera image in RViz (requires usb_cam image_view), default false
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = Path(get_package_share_directory('arm_teleop'))
    teleop_params = str(pkg / 'config' / 'teleop_params.yaml')

    camera_device = LaunchConfiguration('camera_device')
    camera_frame  = LaunchConfiguration('camera_frame')
    tag_size      = LaunchConfiguration('tag_size')
    use_keyboard  = LaunchConfiguration('use_keyboard')

    # ── Camera node (usb_cam) ─────────────────────────────────────────────────
    # Publishes /image_raw and /camera_info.
    # Install: apt-get install ros-humble-usb-cam
    camera_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[{
            'video_device': camera_device,
            'camera_frame_id': camera_frame,
            'image_width':  640,
            'image_height': 480,
            'framerate':    30.0,
            'pixel_format': 'yuyv',
        }],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
        ],
    )

    # Static TF: world → camera_link.
    # Adjust xyz (metres) and rpy (radians) to where the camera is physically
    # mounted relative to the world origin.  Defaults put the camera 1 m above
    # origin, facing in the -Z direction (down-looking).
    # For a table-mounted camera facing toward you, set the actual position and
    # orientation.  Example: 1 m in front of the robot base, 0.5 m up, facing -X.
    camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_static_tf',
        arguments=[
            '--x',   '1.0',
            '--y',   '0.0',
            '--z',   '0.5',
            '--roll',  '0.0',
            '--pitch', '1.5708',   # 90° down (camera Z → world -Z)
            '--yaw',   '3.1416',   # 180° (camera X → world -X)
            '--frame-id', 'world',
            '--child-frame-id', camera_frame,
        ],
    )

    # ── AprilTag detection node ───────────────────────────────────────────────
    # Publishes /detections (AprilTagDetectionArray) and TF for each tag.
    # Install: apt-get install ros-humble-apriltag-ros
    apriltag_node = Node(
        package='apriltag_ros',
        executable='apriltag_node',
        name='apriltag_node',
        output='screen',
        remappings=[
            ('image',       '/camera/image_raw'),
            ('camera_info', '/camera/camera_info'),
        ],
        parameters=[{
            'family':     '36h11',
            'size':       tag_size,
            'publish_tf': True,
        }],
    )

    # ── AprilTag cube teleop node ─────────────────────────────────────────────
    apriltag_teleop = Node(
        package='arm_teleop',
        executable='apriltag_teleop_node',
        name='apriltag_teleop_node',
        output='screen',
        parameters=[teleop_params, {'camera_frame': camera_frame}],
    )

    # ── Keyboard teleop node (optional) ──────────────────────────────────────
    keyboard_teleop = Node(
        package='arm_teleop',
        executable='keyboard_teleop_node',
        name='keyboard_teleop_node',
        output='screen',
        condition=IfCondition(use_keyboard),
        parameters=[teleop_params],
        prefix='xterm -e',   # launch in its own terminal window
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument('camera_frame',  default_value='camera_link'),
        DeclareLaunchArgument('tag_size',      default_value='0.05'),
        DeclareLaunchArgument('use_keyboard',  default_value='false'),

        camera_tf_node,
        camera_node,
        apriltag_node,
        apriltag_teleop,
        keyboard_teleop,
    ])
