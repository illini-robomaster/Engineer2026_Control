#!/usr/bin/env python3
"""
AprilTag cube teleoperation node.

Detects an AprilTag cube held by the operator in front of a table-mounted USB
camera.  The cube's 6-DOF pose in camera space is mapped to a desired
end-effector position in the robot's base_link frame, then MoveIt Servo is
driven toward that target via TwistStamped.

Cube face → tag ID default (configurable in cube_geometry.yaml):
  Tag 0  +X face    Tag 1  -X face
  Tag 2  +Y face    Tag 3  -Y face
  Tag 4  +Z face    Tag 5  -Z face

Multiple visible tags are fused (position average, rotation SLERP) for a
more stable estimate.  When no tags are visible the servo is halted.
"""

import math
import threading
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from apriltag_msgs.msg import AprilTagDetectionArray
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
import yaml


# ── Pure-numpy quaternion helpers (xyzw convention) ──────────────────────────

def qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by unit quaternion q."""
    qv = np.array([v[0], v[1], v[2], 0.0])
    return qmul(qmul(q, qv), qconj(q))[:3]


def q_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion [xyzw] → 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def rot_to_q(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion [xyzw]."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def q_error_rotvec(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Rotation vector (axis*angle) from current to target orientation."""
    q_err = qmul(q_target, qconj(q_current))
    if q_err[3] < 0:            # keep shortest-path
        q_err = -q_err
    vec = q_err[:3]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-7:
        return np.zeros(3)
    angle = 2.0 * math.atan2(vec_norm, float(q_err[3]))
    return (angle / vec_norm) * vec


def average_quaternions(quats: list) -> np.ndarray:
    """Average a list of unit quaternions [xyzw] via rotation-matrix averaging."""
    R_sum = np.zeros((3, 3))
    for q in quats:
        R_sum += q_to_rot(q)
    # Nearest rotation matrix via SVD
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return rot_to_q(R_avg)


# ─────────────────────────────────────────────────────────────────────────────

class AprilTagTeleopNode(Node):

    def __init__(self):
        super().__init__('apriltag_teleop_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('ee_frame', 'End_Effector')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tag_frame_prefix', '36h11:')
        self.declare_parameter('kp_linear', 1.5)
        self.declare_parameter('kp_angular', 2.0)
        self.declare_parameter('max_linear_speed', 0.25)
        self.declare_parameter('max_angular_speed', 0.5)
        self.declare_parameter('deadband_pos_m', 0.008)
        self.declare_parameter('deadband_rot_rad', 0.05)
        self.declare_parameter('detection_timeout_s', 0.4)
        self.declare_parameter('control_orientation', False)
        self.declare_parameter('workspace.cam_origin', [0.0, 0.0, 0.5])
        self.declare_parameter('workspace.ee_origin', [0.2, 0.0, 0.3])
        self.declare_parameter('workspace.scale', [1.0, 1.0, 1.0])
        self.declare_parameter('workspace.cam_to_robot',
                               [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0])

        self._cam_frame = self.get_parameter('camera_frame').value
        self._ee_frame = self.get_parameter('ee_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._tag_prefix = self.get_parameter('tag_frame_prefix').value
        self._kp_lin = self.get_parameter('kp_linear').value
        self._kp_ang = self.get_parameter('kp_angular').value
        self._max_lin = self.get_parameter('max_linear_speed').value
        self._max_ang = self.get_parameter('max_angular_speed').value
        self._db_pos = self.get_parameter('deadband_pos_m').value
        self._db_rot = self.get_parameter('deadband_rot_rad').value
        self._timeout = self.get_parameter('detection_timeout_s').value
        self._ctrl_orient = self.get_parameter('control_orientation').value

        ws = {
            'cam_origin': np.array(self.get_parameter('workspace.cam_origin').value),
            'ee_origin':  np.array(self.get_parameter('workspace.ee_origin').value),
            'scale':      np.array(self.get_parameter('workspace.scale').value),
            'R':          np.array(self.get_parameter('workspace.cam_to_robot').value
                                   ).reshape(3, 3),
        }
        self._ws = ws

        # ── Cube geometry (from YAML file) ────────────────────────────────────
        pkg = get_package_share_directory('arm_teleop')
        with open(f'{pkg}/config/cube_geometry.yaml') as f:
            cube_cfg = yaml.safe_load(f)['cube']

        # face_data[tag_id] = {'pos': np.array, 'quat': np.array [xyzw]}
        self._face_data: dict = {}
        for face in cube_cfg['faces']:
            tag_id = face['id']
            pos = np.array(face['position'], dtype=float)
            q = np.array(face['orientation_xyzw'], dtype=float)
            q /= np.linalg.norm(q)
            self._face_data[tag_id] = {'pos': pos, 'q': q}

        self.get_logger().info(
            f'Cube geometry loaded: {len(self._face_data)} faces, '
            f'side={cube_cfg["side_length"]*100:.0f} cm')

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Publishers / subscribers ──────────────────────────────────────────
        self._twist_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)

        self._detection_sub = self.create_subscription(
            AprilTagDetectionArray, '/detections',
            self._detection_cb, 10)

        # Shared state (detection callback → control timer)
        self._lock = threading.Lock()
        self._last_detection_stamp: Optional[float] = None  # wall-clock seconds
        self._visible_ids: list = []

        # Control loop at ~30 Hz
        self._timer = self.create_timer(0.033, self._control_loop)

        self.get_logger().info('AprilTag teleop node ready.')

    # ── Detection callback ────────────────────────────────────────────────────

    def _detection_cb(self, msg: AprilTagDetectionArray):
        ids = [d.id for d in msg.detections if d.id in self._face_data]
        with self._lock:
            self._visible_ids = ids
            if ids:
                self._last_detection_stamp = self.get_clock().now().nanoseconds / 1e9

    # ── Main control loop ─────────────────────────────────────────────────────

    def _control_loop(self):
        now_s = self.get_clock().now().nanoseconds / 1e9

        with self._lock:
            ids = list(self._visible_ids)
            last_t = self._last_detection_stamp

        # Halt if no recent detection
        if not ids or last_t is None or (now_s - last_t) > self._timeout:
            self._publish_zero()
            return

        # ── Compute cube-centre pose in camera frame ──────────────────────────
        positions_cam = []
        rotations_cam = []

        for tag_id in ids:
            face = self._face_data[tag_id]
            tag_frame = f'{self._tag_prefix}{tag_id}'

            try:
                tf = self._tf_buf.lookup_transform(
                    self._cam_frame, tag_frame, Time())
            except Exception:
                continue

            t = tf.transform.translation
            r = tf.transform.rotation
            tag_pos_cam = np.array([t.x, t.y, t.z])
            tag_q_cam   = np.array([r.x, r.y, r.z, r.w])

            # Recover cube centre:
            # T_cube_in_cam = T_tag_in_cam * inv(T_tag_in_cube)
            tag_pos_cube = face['pos']
            tag_q_cube   = face['q']

            # inv(T_tag_in_cube): rotation = conj(tag_q_cube), pos = -R_inv * tag_pos_cube
            cube_pos_in_tag = qrot(qconj(tag_q_cube), -tag_pos_cube)

            cube_pos_cam = tag_pos_cam + qrot(tag_q_cam, cube_pos_in_tag)
            cube_q_cam   = qmul(tag_q_cam, qconj(tag_q_cube))

            positions_cam.append(cube_pos_cam)
            rotations_cam.append(cube_q_cam / np.linalg.norm(cube_q_cam))

        if not positions_cam:
            self._publish_zero()
            return

        # Average estimates
        cube_pos_cam = np.mean(positions_cam, axis=0)
        cube_q_cam = (rotations_cam[0] if len(rotations_cam) == 1
                      else average_quaternions(rotations_cam))

        # ── Map cube pose to desired EE pose in base_link ─────────────────────
        ws = self._ws
        delta_cam = cube_pos_cam - ws['cam_origin']
        delta_robot = ws['R'] @ (ws['scale'] * delta_cam)
        target_pos = ws['ee_origin'] + delta_robot

        # ── Get current EE pose from TF ───────────────────────────────────────
        try:
            tf_ee = self._tf_buf.lookup_transform(
                self._base_frame, self._ee_frame, Time())
        except Exception:
            self._publish_zero()
            return

        t = tf_ee.transform.translation
        r = tf_ee.transform.rotation
        current_pos = np.array([t.x, t.y, t.z])
        current_q   = np.array([r.x, r.y, r.z, r.w])

        # ── Proportional controller → twist ──────────────────────────────────
        pos_err = target_pos - current_pos
        pos_err_norm = float(np.linalg.norm(pos_err))

        linear = np.zeros(3)
        if pos_err_norm > self._db_pos:
            linear = self._kp_lin * pos_err
            # Clamp magnitude
            spd = float(np.linalg.norm(linear))
            if spd > self._max_lin:
                linear *= self._max_lin / spd

        angular = np.zeros(3)
        if self._ctrl_orient:
            # Map cube rotation to desired EE orientation:
            # apply the same cam→robot rotation to the cube orientation
            cube_R_cam = q_to_rot(cube_q_cam)
            target_R_robot = ws['R'] @ cube_R_cam @ ws['R'].T
            target_q_robot = rot_to_q(target_R_robot)

            rot_err = q_error_rotvec(target_q_robot, current_q)
            rot_err_norm = float(np.linalg.norm(rot_err))
            if rot_err_norm > self._db_rot:
                angular = self._kp_ang * rot_err
                spd = float(np.linalg.norm(angular))
                if spd > self._max_ang:
                    angular *= self._max_ang / spd

        # ── Publish TwistStamped ──────────────────────────────────────────────
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = self._base_frame
        twist_msg.twist.linear.x  = float(linear[0])
        twist_msg.twist.linear.y  = float(linear[1])
        twist_msg.twist.linear.z  = float(linear[2])
        twist_msg.twist.angular.x = float(angular[0])
        twist_msg.twist.angular.y = float(angular[1])
        twist_msg.twist.angular.z = float(angular[2])
        self._twist_pub.publish(twist_msg)

    def _publish_zero(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        self._twist_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AprilTagTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
