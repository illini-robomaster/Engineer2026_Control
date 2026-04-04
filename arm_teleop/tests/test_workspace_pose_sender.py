from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / 'scripts' / 'workspace_pose_sender.py'
)
SPEC = importlib.util.spec_from_file_location('workspace_pose_sender', MODULE_PATH)
workspace_pose_sender = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = workspace_pose_sender
SPEC.loader.exec_module(workspace_pose_sender)


class WorkspacePoseSenderTests(unittest.TestCase):
    def test_quat_slerp_hits_endpoints_and_midpoint(self) -> None:
        q0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        q1 = np.array([math.sin(math.radians(45.0)), 0.0, 0.0, math.cos(math.radians(45.0))], dtype=float)

        np.testing.assert_allclose(workspace_pose_sender._quat_slerp(q0, q1, 0.0), q0, atol=1e-6)
        np.testing.assert_allclose(workspace_pose_sender._quat_slerp(q0, q1, 1.0), q1, atol=1e-6)

        q_half = workspace_pose_sender._quat_slerp(q0, q1, 0.5)
        expected = np.array(
            [math.sin(math.radians(22.5)), 0.0, 0.0, math.cos(math.radians(22.5))],
            dtype=float,
        )
        np.testing.assert_allclose(q_half, expected, atol=1e-6)

    def test_outside_probe_extends_same_ray(self) -> None:
        home = workspace_pose_sender.WorkspacePose(
            label='home',
            q=np.zeros(6, dtype=float),
            pos=np.array([0.5, 0.0, 0.4], dtype=float),
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            radius=0.0,
            azimuth=0.0,
            elevation=0.0,
        )
        shell = workspace_pose_sender.WorkspacePose(
            label='shell',
            q=np.zeros(6, dtype=float),
            pos=np.array([0.6, 0.1, 0.45], dtype=float),
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            radius=float(np.linalg.norm(np.array([0.1, 0.1, 0.05]))),
            azimuth=0.0,
            elevation=0.0,
        )

        outside = workspace_pose_sender._outside_probe(home, shell, 0.015)
        base_vec = shell.pos - home.pos
        outside_vec = outside - home.pos

        np.testing.assert_allclose(
            outside_vec / np.linalg.norm(outside_vec),
            base_vec / np.linalg.norm(base_vec),
            atol=1e-6,
        )
        self.assertAlmostEqual(
            np.linalg.norm(outside_vec) - np.linalg.norm(base_vec),
            0.015,
            places=6,
        )


if __name__ == '__main__':
    unittest.main()
