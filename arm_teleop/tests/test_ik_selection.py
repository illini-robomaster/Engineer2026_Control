import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from arm_teleop.ik_selection import build_candidate, choose_best_candidate


class IkSelectionTests(unittest.TestCase):

    def test_continuity_beats_posture_bias(self):
        last_solved = [0.0] * 6
        joint_state = [0.0] * 6
        q_preferred = [5.0] * 6
        local = build_candidate(
            'last_solved',
            [0.05, 0.0, 0.0, 0.0, 0.0, 0.0],
            last_solved,
            joint_state,
            q_preferred,
            seed_index=0,
            max_joint_jump=0.8,
        )
        remote_random = build_candidate(
            'random',
            [4.9, 5.0, 5.0, 5.0, 5.0, 5.0],
            last_solved,
            joint_state,
            q_preferred,
            seed_index=1,
            max_joint_jump=0.8,
        )

        winner, summary = choose_best_candidate([remote_random, local])

        self.assertIsNotNone(winner)
        self.assertEqual(winner.label, 'last_solved')
        self.assertEqual(summary['winner_pool'], 'under_jump')
        self.assertEqual(summary['valid_random'], 1)

    def test_under_jump_candidate_beats_large_jump_candidate(self):
        last_solved = [0.0] * 6
        joint_state = [0.0] * 6
        q_preferred = [1.0] * 6
        under_jump = build_candidate(
            'joint_states',
            [0.2, 0.2, 0.1, 0.0, 0.0, 0.0],
            last_solved,
            joint_state,
            q_preferred,
            seed_index=0,
            max_joint_jump=0.8,
        )
        over_jump = build_candidate(
            'random',
            [1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
            last_solved,
            joint_state,
            q_preferred,
            seed_index=1,
            max_joint_jump=0.8,
        )

        winner, summary = choose_best_candidate([over_jump, under_jump])

        self.assertIsNotNone(winner)
        self.assertEqual(winner.label, 'joint_states')
        self.assertEqual(summary['valid_under_jump'], 1)
        self.assertEqual(summary['winner_pool'], 'under_jump')

    def test_large_jump_candidate_allowed_when_no_under_jump_exists(self):
        last_solved = [0.0] * 6
        joint_state = [0.0] * 6
        q_preferred = [0.0] * 6
        random_candidate = build_candidate(
            'random',
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            last_solved,
            joint_state,
            q_preferred,
            seed_index=1,
            max_joint_jump=0.3,
        )
        deterministic_candidate = build_candidate(
            'zero/home',
            [1.3, 1.3, 1.2, 1.2, 1.1, 1.1],
            last_solved,
            joint_state,
            q_preferred,
            seed_index=0,
            max_joint_jump=0.3,
        )

        winner, summary = choose_best_candidate([deterministic_candidate, random_candidate])

        self.assertIsNotNone(winner)
        self.assertEqual(winner.label, 'random')
        self.assertEqual(summary['valid_under_jump'], 0)
        self.assertEqual(summary['winner_pool'], 'all')

    def test_deterministic_seed_wins_exact_tie(self):
        last_solved = [0.0] * 6
        joint_state = [0.0] * 6
        q_preferred = [0.0] * 6
        deterministic_candidate = build_candidate(
            'joint_states',
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            last_solved,
            joint_state,
            q_preferred,
            seed_index=0,
            max_joint_jump=0.8,
        )
        random_candidate = build_candidate(
            'random',
            [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
            last_solved,
            joint_state,
            q_preferred,
            seed_index=1,
            max_joint_jump=0.8,
        )

        winner, _ = choose_best_candidate([random_candidate, deterministic_candidate])

        self.assertIsNotNone(winner)
        self.assertEqual(winner.label, 'joint_states')


if __name__ == '__main__':
    unittest.main()
