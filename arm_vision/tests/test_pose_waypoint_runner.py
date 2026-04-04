from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_waypoint_runner import PoseWaypointRunner, RunnerStatus, load_waypoint_sequences


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def _feedback_from_pose(pos: np.ndarray, rot: Rotation, **extra) -> dict:
    quat = rot.as_quat()
    feedback = {
        'fk_x': float(pos[0]),
        'fk_y': float(pos[1]),
        'fk_z': float(pos[2]),
        'fk_qx': float(quat[0]),
        'fk_qy': float(quat[1]),
        'fk_qz': float(quat[2]),
        'fk_qw': float(quat[3]),
        'ik_ok': True,
    }
    feedback.update(extra)
    return feedback


class PoseWaypointRunnerTests(unittest.TestCase):
    def test_relative_start_sequence_resolves_from_current_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / 'waypoints.yaml'
            _write_yaml(
                cfg_path,
                """
waypoints:
  defaults:
    frame: relative_start
    hold_s: 0.0
    timeout_s: 1.0
    pos_tolerance_mm: 1.0
    ori_tolerance_deg: 1.0
  sequences:
    probe:
      waypoints:
        - name: offset
          position: [0.020, -0.010, 0.015]
          rpy_deg: [5.0, 0.0, -10.0]
""",
            )
            sequences = load_waypoint_sequences(str(cfg_path))
            runner = PoseWaypointRunner(sequences)

            start_pos_a = np.array([0.30, 0.10, 0.20], dtype=float)
            start_rot_a = Rotation.from_euler('xyz', [0.0, 20.0, 5.0], degrees=True)
            self.assertTrue(runner.start(start_pos_a, start_rot_a, now=1.0))
            target_pos_a, target_rot_a = runner.current_target_pose
            np.testing.assert_allclose(target_pos_a, start_pos_a + np.array([0.02, -0.01, 0.015]))
            np.testing.assert_allclose(
                target_rot_a.as_quat(),
                (start_rot_a * Rotation.from_euler('xyz', [5.0, 0.0, -10.0], degrees=True)).as_quat(),
                atol=1e-6,
            )

            runner.cancel('reset', {}, start_pos_a, start_rot_a, now=1.1)

            start_pos_b = np.array([0.45, -0.05, 0.18], dtype=float)
            start_rot_b = Rotation.from_euler('xyz', [10.0, -15.0, 30.0], degrees=True)
            self.assertTrue(runner.start(start_pos_b, start_rot_b, now=2.0))
            target_pos_b, target_rot_b = runner.current_target_pose
            np.testing.assert_allclose(target_pos_b, start_pos_b + np.array([0.02, -0.01, 0.015]))
            np.testing.assert_allclose(
                target_rot_b.as_quat(),
                (start_rot_b * Rotation.from_euler('xyz', [5.0, 0.0, -10.0], degrees=True)).as_quat(),
                atol=1e-6,
            )

    def test_sequence_completion_writes_log_with_feedback_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / 'waypoints.yaml'
            _write_yaml(
                cfg_path,
                """
waypoints:
  defaults:
    frame: relative_start
    hold_s: 0.0
    timeout_s: 1.0
    pos_tolerance_mm: 2.0
    ori_tolerance_deg: 2.0
  sequences:
    sweep:
      waypoints:
        - name: first
          position: [0.010, 0.000, 0.000]
          rpy_deg: [0.0, 0.0, 0.0]
        - name: second
          position: [0.020, 0.005, 0.000]
          rpy_deg: [0.0, 5.0, 0.0]
""",
            )
            runner = PoseWaypointRunner(load_waypoint_sequences(str(cfg_path)), log_dir=tmpdir)
            start_pos = np.array([0.40, 0.00, 0.20], dtype=float)
            start_rot = Rotation.identity()
            runner.start(start_pos, start_rot, now=0.0)

            target_pos, target_rot = runner.current_target_pose
            runner.tick(
                _feedback_from_pose(
                    target_pos,
                    target_rot,
                    ik_result='6D',
                    ik_seed='last_solved',
                    ik_seed_kind='deterministic',
                ),
                start_pos,
                start_rot,
                now=0.1,
            )

            target_pos, target_rot = runner.current_target_pose
            runner.tick(
                _feedback_from_pose(
                    target_pos,
                    target_rot,
                    ik_result='POS_ONLY',
                    ik_fail_cause='orientation unreachable near current branch',
                    ik6d_best_failed_class='position_only',
                ),
                start_pos,
                start_rot,
                now=0.2,
            )

            self.assertEqual(runner.status, RunnerStatus.SUCCEEDED)
            self.assertTrue(runner.last_log_path)

            with open(runner.last_log_path, newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(any(row['event'] == 'sequence_start' for row in rows))
            self.assertTrue(any(row['event'] == 'waypoint_start' for row in rows))
            self.assertTrue(any(row['event'] == 'sequence_complete' for row in rows))
            self.assertTrue(any(row['ik_result'] == 'POS_ONLY' for row in rows))
            self.assertTrue(any(row['ik_fail_cause'] == 'orientation unreachable near current branch' for row in rows))

    def test_timeout_marks_sequence_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = Path(tmpdir) / 'waypoints.yaml'
            _write_yaml(
                cfg_path,
                """
waypoints:
  defaults:
    frame: relative_start
    hold_s: 0.2
    timeout_s: 0.5
    pos_tolerance_mm: 1.0
    ori_tolerance_deg: 1.0
  sequences:
    timeout_probe:
      waypoints:
        - name: far
          position: [0.050, 0.000, 0.000]
          rpy_deg: [0.0, 0.0, 0.0]
""",
            )
            runner = PoseWaypointRunner(load_waypoint_sequences(str(cfg_path)))
            start_pos = np.array([0.20, 0.00, 0.10], dtype=float)
            start_rot = Rotation.identity()
            runner.start(start_pos, start_rot, now=0.0)

            runner.tick(
                _feedback_from_pose(start_pos, start_rot, ik_result='FAIL'),
                start_pos,
                start_rot,
                now=0.6,
            )

            self.assertEqual(runner.status, RunnerStatus.FAILED)
            self.assertFalse(runner.is_active)


if __name__ == '__main__':
    unittest.main()
