from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figure8_waypoint_sender import build_horizontal_figure8_sequence


class Figure8WaypointSenderTests(unittest.TestCase):
    def test_sequence_is_closed_and_stays_in_yz_plane(self) -> None:
        sequence = build_horizontal_figure8_sequence(
            amp_y=0.05,
            amp_z=0.03,
            waypoint_count=32,
            loops=1,
            hold_s=0.0,
            timeout_s=2.0,
            pos_tolerance_mm=8.0,
            ori_tolerance_deg=5.0,
        )

        positions = np.array([wp.position for wp in sequence.waypoints], dtype=float)
        self.assertTrue(np.allclose(positions[:, 0], 0.0))
        np.testing.assert_allclose(positions[0], np.zeros(3), atol=1e-9)
        np.testing.assert_allclose(positions[-1], np.zeros(3), atol=1e-9)
        self.assertAlmostEqual(float(np.max(positions[:, 1])), 0.05, places=6)
        self.assertAlmostEqual(float(np.min(positions[:, 1])), -0.05, places=6)
        self.assertAlmostEqual(float(np.max(positions[:, 2])), 0.03, places=6)
        self.assertAlmostEqual(float(np.min(positions[:, 2])), -0.03, places=6)

    def test_multiple_loops_repeat_the_center_crossing(self) -> None:
        waypoint_count = 24
        loops = 2
        sequence = build_horizontal_figure8_sequence(
            amp_y=0.04,
            amp_z=0.02,
            waypoint_count=waypoint_count,
            loops=loops,
            hold_s=0.0,
            timeout_s=2.0,
            pos_tolerance_mm=8.0,
            ori_tolerance_deg=5.0,
        )

        self.assertEqual(len(sequence.waypoints), 1 + waypoint_count * loops)
        for step in (0, waypoint_count, waypoint_count * loops):
            np.testing.assert_allclose(sequence.waypoints[step].position, np.zeros(3), atol=1e-9)


if __name__ == '__main__':
    unittest.main()
