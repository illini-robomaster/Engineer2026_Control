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


# ── Rotation helpers ─────────────────────────────────────────────────────────

def _nearest_rotation(R: np.ndarray) -> np.ndarray:
    """Project an approximate rotation matrix onto SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean


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
        self._tag_size = float(cfg.get('tag_size', cfg.get('side_length', 0.032)))
        self._family   = cfg.get('tag_family', 'tag25h9')

        # tag_data[tag_id] = {'pos': np.array [x,y,z], 'q': np.array [x,y,z,w]}
        # Both are the tag's pose in the cube frame.
        # Supports both 'tags' (new format) and 'faces' (old format).
        self._face: dict[int, dict] = {}
        for entry in cfg.get('tags', cfg.get('faces', [])):
            tid = int(entry['id'])
            pos = np.array(entry['position'], dtype=float)
            q   = np.array(entry['orientation_xyzw'], dtype=float)
            q  /= np.linalg.norm(q)
            self._face[tid] = {'pos': pos, 'q': q}

        # Primary detector: fast path, must complete in <15ms on target hardware.
        # quad_decimate=2.0 at 1280×800 → internal 640×400 → ~5ms detection.
        self._detector = Detector(
            families=self._family,
            nthreads=4,
            quad_decimate=2.0,   # 2× downsample — fast primary path
            quad_sigma=0.4,
            refine_edges=True,
            decode_sharpening=0.5,
        )

        # Fallback detector: full resolution for steep angles / difficult frames.
        # Only runs when the primary finds <2 known tags (~10-15% of frames).
        # quad_decimate=1.0 catches small/angled tags that 2× misses.
        self._detector_fallback = Detector(
            families=self._family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.8,
            refine_edges=True,
            decode_sharpening=1.0,
        )

        # CLAHE for local contrast normalisation — steep-angle tags have lower
        # contrast due to perspective foreshortening.
        self._clahe = None  # lazily created (needs cv2)

        # Precomputed undistort maps (lazily initialized on first frame).
        # cv2.remap with precomputed maps is ~3–5× faster than undistort() per frame.
        # Uses getOptimalNewCameraMatrix(alpha=0) so the undistorted image has
        # no black borders — prevents false detection failures at frame edges.
        self._remap_map1: 'np.ndarray | None' = None
        self._remap_map2: 'np.ndarray | None' = None
        self._new_K: np.ndarray | None = None

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

        # Undistort using precomputed maps (lazy init on first frame).
        # alpha=0 → crop to valid pixels only (no black borders that confuse detection).
        # The new camera matrix may differ from the original, so we use its intrinsics
        # for pose estimation below.
        if self._remap_map1 is None:
            h, w = gray.shape[:2]
            self._new_K, _ = cv2.getOptimalNewCameraMatrix(
                self._K, self._dist, (w, h), alpha=0)
            self._remap_map1, self._remap_map2 = cv2.initUndistortRectifyMap(
                self._K, self._dist, None, self._new_K, (w, h), cv2.CV_16SC2)
        gray = cv2.remap(gray, self._remap_map1, self._remap_map2, cv2.INTER_LINEAR)

        # CLAHE: normalise local contrast so steep-angle tags aren't washed out
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = self._clahe.apply(gray)

        # Use the NEW camera matrix intrinsics for pose estimation on the undistorted image
        nfx = float(self._new_K[0, 0])
        nfy = float(self._new_K[1, 1])
        ncx = float(self._new_K[0, 2])
        ncy = float(self._new_K[1, 2])

        cam_params = (nfx, nfy, ncx, ncy)

        detections = self._detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=cam_params,
            tag_size=self._tag_size,
        )

        # Count known tags in primary result
        n_known = sum(1 for d in detections if int(d.tag_id) in self._face)

        # Adaptive retry: if <2 known tags, try the fallback detector with
        # heavier blur and sharpening.  Only costs extra time on difficult frames.
        if n_known < 2:
            fallback = self._detector_fallback.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=cam_params,
                tag_size=self._tag_size,
            )
            fb_known = sum(1 for d in fallback if int(d.tag_id) in self._face)
            if fb_known > n_known:
                detections = fallback

        positions:   list[np.ndarray] = []
        orientations: list[np.ndarray] = []

        for d in detections:
            tid = int(d.tag_id)
            if tid not in self._face:
                continue    # unknown tag — not part of this cube

            face = self._face[tid]

            # pupil-apriltags: pose_R, pose_t map FROM tag frame TO camera frame
            #   p_cam = R * p_tag + t
            # Project onto SO(3) — numerical noise can give det < 0
            R_tag_to_cam = _nearest_rotation(d.pose_R)  # 3×3
            t_tag_in_cam = d.pose_t.reshape(3)           # tag origin in camera frame

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
