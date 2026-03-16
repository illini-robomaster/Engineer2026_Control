"""
Workspace mapping: cube pose in camera frame → EE target in robot base frame.

Position mapping:
  ee_target = ee_home + R_cam2robot * gain * (cube_pos - cam_origin)

cam_origin is a fixed reference point in camera space (e.g. 30 cm in front
of the camera, at image centre).  When the cube is at cam_origin, the arm
is at ee_home.  Moving the cube away from cam_origin moves the arm by
gain × the displacement, rotated into the robot frame.

Press H to re-home (clutch): cam_origin resets to the current cube position
and ee_home resets to the current EE position, so you can reposition your
hand without moving the arm.

Orientation uses a delta approach:
  dR = R_cube_current * inv(R_cube_at_origin)   (in robot frame)
  ee_ori = dR * ee_neutral_ori

Config is loaded from config/workspace.yaml.
"""

from __future__ import annotations

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


class WorkspaceMapper:
    """Cube-delta teleop mapper with fixed camera reference point."""

    def __init__(self, config_path: str = 'config/workspace.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)['workspace']

        self._gain = np.array(cfg.get('gain', [2.0, 2.0, 2.0]), dtype=float)

        flat = cfg['cam_to_robot']
        if isinstance(flat, list):
            flat = [float(v) for v in flat]
        self._R = np.array(flat, dtype=float).reshape(3, 3)

        # Fixed reference: when cube is here, arm is at ee_home
        self._cam_origin = np.array(cfg['cam_origin'], dtype=float)
        self._ee_home = np.array(cfg.get('ee_home', [0.426, 0.0, 0.395]), dtype=float)

        # Safety limits
        self._pos_min = np.array(cfg.get('pos_min', [-1.0, -1.0, -1.0]), dtype=float)
        self._pos_max = np.array(cfg.get('pos_max', [ 1.0,  1.0,  1.0]), dtype=float)
        self._reach_max = float(cfg.get('reach_max', 0.82))
        self._reach_min = float(cfg.get('reach_min', 0.08))

        # Baseline distance for reach_frac normalisation
        self._dist_home = float(np.linalg.norm(self._ee_home))

        # Orientation
        self._ee_neutral_rot = Rotation.from_quat(
            np.array(cfg.get('ee_neutral_quat', [0.0, 0.0, 0.0, 1.0]), dtype=float))

        # Cube orientation at cam_origin (robot frame) — set on first detection
        # or manually via calibration.  Until set, orientation mapping returns neutral.
        self._cube_origin_rot_robot: Rotation | None = None

    # ── Re-home (clutch) ───────────────────────────────────────────────────

    def re_home(self, cube_pos: np.ndarray, cube_quat: np.ndarray | None,
                ee_pos: np.ndarray):
        """Clutch: reset reference so current cube pos → current EE pos."""
        self._cam_origin = cube_pos.copy()
        self._ee_home = ee_pos.copy()
        self._dist_home = float(np.linalg.norm(ee_pos))
        if cube_quat is not None:
            R_cam = Rotation.from_quat(cube_quat)
            self._cube_origin_rot_robot = Rotation.from_matrix(
                self._R @ R_cam.as_matrix() @ self._R.T)

    # ── Mapping ────────────────────────────────────────────────────────────

    def map_position(self, cube_pos: np.ndarray) -> tuple[np.ndarray, float]:
        """Map cube position to EE target.

        Returns (ee_pos, reach_frac).  reach_frac is 0 at ee_home, 1.0 at limit.
        """
        delta_cam = cube_pos - self._cam_origin
        delta_robot = self._R @ (self._gain * delta_cam)
        pos = self._ee_home + delta_robot

        # Spherical reach clamp
        dist = float(np.linalg.norm(pos))
        outward_budget = max(self._reach_max - self._dist_home, 0.01)
        reach_frac = max(0.0, dist - self._dist_home) / outward_budget

        if dist > self._reach_max:
            pos = pos * (self._reach_max / dist)
        elif dist < self._reach_min and dist > 1e-6:
            pos = pos * (self._reach_min / dist)

        pos = np.clip(pos, self._pos_min, self._pos_max)
        return pos, reach_frac

    def map_orientation(self, cube_quat: np.ndarray) -> np.ndarray:
        """Map cube orientation delta to EE target quaternion [x,y,z,w]."""
        # Record first orientation as the origin reference
        if self._cube_origin_rot_robot is None:
            R_cam = Rotation.from_quat(cube_quat)
            self._cube_origin_rot_robot = Rotation.from_matrix(
                self._R @ R_cam.as_matrix() @ self._R.T)
            return self._ee_neutral_rot.as_quat()

        # Current cube orientation in robot frame
        R_cam = Rotation.from_quat(cube_quat)
        R_cube_robot = Rotation.from_matrix(
            self._R @ R_cam.as_matrix() @ self._R.T)

        # Delta from origin
        dR = R_cube_robot * self._cube_origin_rot_robot.inv()
        return (dR * self._ee_neutral_rot).as_quat()

    def map(
        self,
        cube_pos: np.ndarray,
        cube_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns (ee_pos, ee_quat, reach_frac) in robot base frame."""
        pos, reach_frac = self.map_position(cube_pos)
        return pos, self.map_orientation(cube_quat), reach_frac

    # ── Safe zone (for camera overlay) ─────────────────────────────────────

    def safe_zone_cam(self, cube_pos: np.ndarray, cam_z: float
                      ) -> tuple[float, float, float, float]:
        """Compute how far the cube can move before hitting limits.

        Returns (cam_x_min, cam_x_max, cam_y_min, cam_y_max) in camera metres.
        """
        ee_pos, _ = self.map_position(cube_pos)

        # Room in robot frame before hitting box limits
        room_lo = ee_pos - self._pos_min
        room_hi = self._pos_max - ee_pos

        # Tighten by sphere
        dist = float(np.linalg.norm(ee_pos))
        if dist > 0.01:
            sphere_room = max(0.0, self._reach_max - dist)
            for i in range(3):
                room_lo[i] = min(room_lo[i], sphere_room + abs(ee_pos[i]))
                room_hi[i] = min(room_hi[i], sphere_room + abs(ee_pos[i]))

        # Inverse-map to camera coordinates.
        # R = [[0,0,-1],[1,0,0],[0,-1,0]] → R.T maps:
        #   cam_x = robot_y / gain[0]
        #   cam_y = -robot_z / gain[1]
        g = self._gain
        cam_x_lo = cube_pos[0] - room_lo[1] / g[0]
        cam_x_hi = cube_pos[0] + room_hi[1] / g[0]
        cam_y_lo = cube_pos[1] - room_hi[2] / g[1]  # robot_z up → cam_y up (negative)
        cam_y_hi = cube_pos[1] + room_lo[2] / g[1]

        return float(cam_x_lo), float(cam_x_hi), float(cam_y_lo), float(cam_y_hi)

    # ── Z-zone shift ─────────────────────────────────────────────────────

    def shift_z_zone(self, delta_z: float,
                     current_cube_pos: np.ndarray | None = None):
        """Shift Z workspace by *delta_z* (robot frame, positive = up).

        Adjusts cam_origin so the arm does not jump — the current cube
        position continues to map to the same EE target.
        """
        self._ee_home[2] += delta_z
        if current_cube_pos is not None:
            # Continuity: same cube_Y must produce the same robot_Z
            self._cam_origin[1] -= delta_z / self._gain[1]
        self._dist_home = float(np.linalg.norm(self._ee_home))

    @property
    def z_limits(self) -> tuple[float, float]:
        """Safety-clamp Z boundaries (pos_min[2], pos_max[2])."""
        return float(self._pos_min[2]), float(self._pos_max[2])

    @property
    def ee_home_z(self) -> float:
        """Current Z-axis home of the workspace mapping."""
        return float(self._ee_home[2])
