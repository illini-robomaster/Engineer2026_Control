#!/usr/bin/env python3
"""MoveIt IK service wrapper for teleoperation.

Provides async/sync access to /compute_ik service with fallback behavior.
Designed to wrap MoveIt's GetPositionIK service for use in ik_teleop_node.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest
from sensor_msgs.msg import JointState


class IkServiceWrapper:
    """Async wrapper around MoveIt's GetPositionIK service.
    
    Handles:
      - Service availability checking
      - Request/response marshalling
      - Timeouts and error handling
      - Diagnostics (success rate, timing)
    """

    def __init__(
        self,
        node: Node,
        service_timeout_sec: float = 0.1,
        service_name: str = '/compute_ik',
    ):
        """
        Initialize IK service wrapper.
        
        Args:
            node: ROS 2 node for creating client
            service_timeout_sec: Time to wait for service availability
            service_name: ROS service name (default: /compute_ik)
        """
        self._node = node
        self._service_timeout = service_timeout_sec
        self._service_name = service_name
        self._available = False
        self._client = None
        
        # Diagnostics
        self._attempts = 0
        self._successes = 0
        self._failures = 0
        self._time_sum_ms = 0.0
        self._time_max_ms = 0.0
        
        # Try to connect
        try:
            self._client = node.create_client(GetPositionIK, service_name)
            self._available = self._client.wait_for_service(
                timeout_sec=service_timeout_sec
            )
            if self._available:
                node.get_logger().info(
                    f'IK service wrapper initialized: {service_name}'
                )
            else:
                node.get_logger().warn(
                    f'IK service {service_name} not available within '
                    f'{service_timeout_sec}s (will use fallback)'
                )
        except Exception as e:
            node.get_logger().warn(
                f'IK service wrapper init failed: {e} (will use fallback)'
            )
            self._available = False

    def is_available(self) -> bool:
        """Check if IK service is available."""
        return self._available

    def solve_ik_blocking(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        seed: list,
        joint_names: list[str],
        group_name: str = 'arm',
        base_frame: str = 'base_link',
        timeout_override_sec: Optional[float] = None,
        max_wall_time_sec: float = 0.005,
    ) -> Optional[list]:
        """
        Solve IK synchronously with wall-time limit to prevent blocking control loop.
        
        IMPORTANT: Call only from a non-critical path or with max_wall_time_sec << 
        the control loop period (e.g., 60 Hz → 16 ms). Falling back to custom DLS 
        is expected and normal.
        
        Args:
            target_pos: Target EE position [x, y, z] (meters)
            target_quat: Target EE orientation [qx, qy, qz, qw]
            seed: Seed joint configuration (list of N floats)
            joint_names: List of joint names in order
            group_name: MoveIt group name (default: 'arm')
            base_frame: Planning frame ID (default: 'base_link')
            timeout_override_sec: Override kinematics timeout (optional)
            max_wall_time_sec: Hard deadline — give up if elapsed > this (default: 5ms)
        
        Returns:
            List of joint angles on success, None on failure or timeout.
        """
        if not self._available or self._client is None:
            return None
        
        self._attempts += 1
        t0 = time.time()
        
        try:
            # Build robot state with seed
            robot_state = JointState()
            robot_state.name = joint_names
            robot_state.position = [float(x) for x in seed]
            robot_state.velocity = [0.0] * len(seed)
            
            # Build target pose
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = base_frame
            pose_stamped.pose.position.x = float(target_pos[0])
            pose_stamped.pose.position.y = float(target_pos[1])
            pose_stamped.pose.position.z = float(target_pos[2])
            pose_stamped.pose.orientation.x = float(target_quat[0])
            pose_stamped.pose.orientation.y = float(target_quat[1])
            pose_stamped.pose.orientation.z = float(target_quat[2])
            pose_stamped.pose.orientation.w = float(target_quat[3])
            
            # Build IK request
            request = GetPositionIK.Request()
            request.ik_request.group_name = group_name
            request.ik_request.robot_state.joint_state = robot_state
            request.ik_request.pose_stamped = pose_stamped
            
            # Set timeout if override provided
            if timeout_override_sec is not None:
                request.ik_request.timeout.sec = int(timeout_override_sec)
                request.ik_request.timeout.nanosec = int(
                    (timeout_override_sec % 1.0) * 1e9
                )
            
            # Call service with wall-time protection
            # If we exceed max_wall_time_sec, abort and let control loop use fallback
            response = self._client.call(request)
            
            elapsed_ms = (time.time() - t0) * 1000.0
            
            # Check if we exceeded the hard deadline
            if elapsed_ms > max_wall_time_sec * 1000.0:
                self._failures += 1
                self._node.get_logger().debug(
                    f'IK service exceeded wall time ({elapsed_ms:.2f}ms > {max_wall_time_sec*1000:.1f}ms)'
                )
                return None
            
            self._time_sum_ms += elapsed_ms
            self._time_max_ms = max(self._time_max_ms, elapsed_ms)
            
            # Check success
            if response.error_code.val == 1:  # MoveItErrorCodes.SUCCESS
                solution = response.solution.joint_state
                if solution.name and solution.position:
                    self._successes += 1
                    self._node.get_logger().debug(
                        f'IK solved in {elapsed_ms:.2f}ms'
                    )
                    return list(solution.position)
            
            self._failures += 1
            self._node.get_logger().debug(
                f'IK failed (code={response.error_code.val}) in {elapsed_ms:.2f}ms'
            )
            return None
            
        except Exception as e:
            self._failures += 1
            self._node.get_logger().debug(f'IK service call raised: {e}')
            return None

    def get_diagnostics(self) -> dict:
        """
        Get diagnostic statistics.
        
        Returns:
            Dict with keys: attempts, successes, failures, success_rate, 
                           avg_time_ms, max_time_ms
        """
        success_rate = (
            self._successes / max(self._attempts, 1) * 100.0
        )
        avg_time = (
            self._time_sum_ms / max(self._successes, 1)
        )
        return {
            'attempts': self._attempts,
            'successes': self._successes,
            'failures': self._failures,
            'success_rate': success_rate,
            'avg_time_ms': avg_time,
            'max_time_ms': self._time_max_ms,
        }

    def reset_diagnostics(self):
        """Reset diagnostic counters."""
        self._attempts = 0
        self._successes = 0
        self._failures = 0
        self._time_sum_ms = 0.0
        self._time_max_ms = 0.0
