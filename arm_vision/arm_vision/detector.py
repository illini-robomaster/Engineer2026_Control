"""
AprilTag detection and cube pose estimation.

Given a video frame, camera calibration, and cube configuration, detects all
visible AprilTags and fuses the detections to estimate the cube centre's 6D
pose (position + quaternion) in the camera frame.

Multiple visible tags are averaged: positions are mean-averaged, orientations
are averaged via rotation-matrix SVD (nearest rotation to the sum).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


# ── Quaternion helpers (xyzw convention, matching scipy) ─────────────────────

def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _average_quaternions(quats: list[np.ndarray]) -> np.ndarray:
    """Average quaternions via rotation-matrix SVD."""
    R_sum = sum(Rotation.from_quat(q).as_matrix() for q in quats)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    return Rotation.from_matrix(R_avg).as_quat()


# ─────────────────────────────────────────────────────────────────────────────

class CubeDetector:
    """
    Detects AprilTags in a BGR frame and estimates the cube pose.

    Parameters
    ----------
    camera_matrix : 3×3 intrinsic matrix (from camera_cal.load())
    dist_coeffs   : distortion coefficients (from camera_cal.load())
    cube_config   : dict loaded by cube_cal.load()
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs:   np.ndarray,
        cube_config:   dict,
    ):
        try:
            from pupil_apriltags import Detector
        except ImportError:
            raise ImportError("Install pupil-apriltags:  pip install pupil-apriltags")

        self._K    = camera_matrix
        self._dist = dist_coeffs
        self._fx   = float(camera_matrix[0, 0])
        self._fy   = float(camera_matrix[1, 1])
        self._cx   = float(camera_matrix[0, 2])
        self._cy   = float(camera_matrix[1, 2])

        cfg   = cube_config
        self._side   = float(cfg['side_length'])
        self._family = cfg.get('tag_family', 'tag36h11')

        # face_data[tag_id] = {'pos': np.array [x,y,z], 'q': np.array [x,y,z,w]}
        # Both are the tag's pose in the cube frame.
        self._face: dict[int, dict] = {}
        for face in cfg['faces']:
            tid = int(face['id'])
            pos = np.array(face['position'], dtype=float)
            q   = np.array(face['orientation_xyzw'], dtype=float)
            q  /= np.linalg.norm(q)
            self._face[tid] = {'pos': pos, 'q': q}

        self._detector = Detector(
            families=self._family,
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
        )

    def detect(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, list]:
        """
        Detect AprilTags in a BGR frame and estimate cube centre pose.

        Returns
        -------
        cube_pos  : np.ndarray [x, y, z] in camera frame, or None if no tags found.
        cube_quat : np.ndarray [x, y, z, w] in camera frame, or None.
        raw       : list of pupil_apriltags Detection objects (for visualisation).
        """
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Undistort the image before detection for better accuracy
        gray = cv2.undistort(gray, self._K, self._dist)

        detections = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self._fx, self._fy, self._cx, self._cy),
            tag_size=self._side,
        )

        positions:   list[np.ndarray] = []
        orientations: list[np.ndarray] = []

        for d in detections:
            tid = int(d.tag_id)
            if tid not in self._face:
                continue    # unknown tag — not part of this cube

            face = self._face[tid]

            # pupil-apriltags: pose_R, pose_t map FROM tag frame TO camera frame
            #   p_cam = R * p_tag + t
            R_tag_to_cam = d.pose_R                 # 3×3
            t_tag_in_cam = d.pose_t.reshape(3)      # tag origin in camera frame

            # Tag orientation in camera frame as quaternion [xyzw]
            q_tag_cam = Rotation.from_matrix(R_tag_to_cam).as_quat()

            # ── Recover cube centre from tag pose ──────────────────────────────
            # T_cube_in_cam = T_tag_in_cam * inv(T_tag_in_cube)
            #
            # inv(T_tag_in_cube):
            #   rotation = conj(face.q)
            #   translation = -R_inv * face.pos = -R_conj * face.pos
            #
            q_face_conj   = _qconj(face['q'])
            R_face_inv    = Rotation.from_quat(q_face_conj).as_matrix()
            t_cube_in_tag = R_face_inv @ (-face['pos'])

            # cube centre position in camera frame
            cube_pos = t_tag_in_cam + R_tag_to_cam @ t_cube_in_tag

            # cube orientation in camera frame
            cube_q = _qmul(q_tag_cam, q_face_conj)
            cube_q /= np.linalg.norm(cube_q)

            positions.append(cube_pos)
            orientations.append(cube_q)

        if not positions:
            return None, None, detections

        cube_pos_avg  = np.mean(positions, axis=0)
        cube_quat_avg = (
            orientations[0] if len(orientations) == 1
            else _average_quaternions(orientations)
        )

        return cube_pos_avg, cube_quat_avg, detections

    def draw(self, frame_bgr: np.ndarray, raw_detections: list) -> np.ndarray:
        """Draw tag outlines and IDs onto a BGR frame.  Returns the annotated copy."""
        import cv2
        out = frame_bgr.copy()
        for d in raw_detections:
            corners = d.corners.astype(int)
            known = int(d.tag_id) in self._face
            colour = (0, 220, 0) if known else (100, 100, 100)
            cv2.polylines(out, [corners.reshape(-1, 1, 2)], True, colour, 2)
            cv2.putText(out, f'id={d.tag_id}',
                        tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, colour, 2)
        return out
