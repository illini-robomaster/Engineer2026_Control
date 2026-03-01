"""
Workspace mapping: cube pose in camera frame → EE target in robot base frame.

The affine mapping is:
  ee_target_pos = ee_origin + R_cam2robot * diag(scale) * (cube_pos - cam_origin)

For orientation (optional):
  ee_target_quat computed by rotating the cube's orientation through R_cam2robot.

Config is loaded from config/workspace.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


class WorkspaceMapper:
    """
    Maps a cube centre pose (camera frame) to a desired EE pose (robot frame).

    Parameters
    ----------
    config_path : path to workspace.yaml
    """

    def __init__(self, config_path: str = 'config/workspace.yaml'):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)['workspace']

        self._cam_origin = np.array(cfg['cam_origin'], dtype=float)
        self._ee_origin  = np.array(cfg['ee_origin'],  dtype=float)
        self._scale      = np.array(cfg['scale'],       dtype=float)

        flat = cfg['cam_to_robot']
        if isinstance(flat, list):
            flat = [float(v) for v in flat]
        self._R = np.array(flat, dtype=float).reshape(3, 3)

    def map_position(self, cube_pos_cam: np.ndarray) -> np.ndarray:
        """
        Map cube centre position in camera frame to desired EE position in robot frame.

        Parameters
        ----------
        cube_pos_cam : np.ndarray shape (3,)

        Returns
        -------
        ee_target : np.ndarray shape (3,)
        """
        delta_cam   = cube_pos_cam - self._cam_origin
        delta_robot = self._R @ (self._scale * delta_cam)
        return self._ee_origin + delta_robot

    def map_orientation(self, cube_quat_cam: np.ndarray) -> np.ndarray:
        """
        Map cube orientation (camera frame) to desired EE orientation (robot frame).

        Applies the cam→robot rotation: R_robot = R_cam2robot * R_cube_cam * R_cam2robot^T

        Parameters
        ----------
        cube_quat_cam : np.ndarray [x, y, z, w]

        Returns
        -------
        ee_quat : np.ndarray [x, y, z, w]
        """
        R_cube = Rotation.from_quat(cube_quat_cam).as_matrix()
        R_ee   = self._R @ R_cube @ self._R.T
        return Rotation.from_matrix(R_ee).as_quat()

    def map(
        self,
        cube_pos_cam:  np.ndarray,
        cube_quat_cam: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience wrapper.  Returns (ee_pos, ee_quat) both in robot frame.
        """
        return self.map_position(cube_pos_cam), self.map_orientation(cube_quat_cam)
