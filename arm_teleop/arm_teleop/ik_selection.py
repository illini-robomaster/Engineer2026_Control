from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class IkCandidate:
    label: str
    seed_kind: str
    joints: list[float]
    continuity_cost: float
    joint_state_cost: float
    posture_cost: float
    max_jump: Optional[float]
    within_jump_limit: bool
    seed_index: int


def seed_kind_from_label(label: str) -> str:
    return 'random' if label.startswith('random') else 'deterministic'


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed angular difference (a − b), result in (−π, π]."""
    d = float(a) - float(b)
    return (d + math.pi) % (2 * math.pi) - math.pi


def _l2_distance(ref: Optional[Sequence[float]], joints: Sequence[float]) -> float:
    if ref is None:
        return float('inf')
    return math.sqrt(sum(_angle_diff(float(a), float(b)) ** 2 for a, b in zip(ref, joints)))


def _max_abs_delta(ref: Optional[Sequence[float]], joints: Sequence[float]) -> Optional[float]:
    if ref is None:
        return None
    return max(abs(_angle_diff(float(a), float(b))) for a, b in zip(ref, joints))


def build_candidate(
    label: str,
    joints: Sequence[float],
    last_solved: Optional[Sequence[float]],
    joint_state: Optional[Sequence[float]],
    q_preferred: Optional[Sequence[float]],
    seed_index: int,
    max_joint_jump: float,
) -> IkCandidate:
    continuity_ref = last_solved if last_solved is not None else joint_state
    max_jump = _max_abs_delta(last_solved, joints)
    within_jump_limit = (
        max_joint_jump <= 0.0
        or max_jump is None
        or max_jump <= max_joint_jump
    )
    return IkCandidate(
        label=label,
        seed_kind=seed_kind_from_label(label),
        joints=[float(v) for v in joints],
        continuity_cost=_l2_distance(continuity_ref, joints),
        joint_state_cost=_l2_distance(joint_state, joints),
        posture_cost=_l2_distance(q_preferred, joints),
        max_jump=max_jump,
        within_jump_limit=within_jump_limit,
        seed_index=seed_index,
    )


def candidate_sort_key(candidate: IkCandidate) -> tuple[float, float, int, float, int]:
    return (
        candidate.continuity_cost,
        candidate.joint_state_cost,
        0 if candidate.seed_kind == 'deterministic' else 1,
        candidate.posture_cost,
        candidate.seed_index,
    )


def choose_best_candidate(candidates: Sequence[IkCandidate]) -> tuple[Optional[IkCandidate], dict]:
    if not candidates:
        return None, {
            'valid_total': 0,
            'valid_deterministic': 0,
            'valid_random': 0,
            'valid_under_jump': 0,
            'winner_pool': '',
        }

    valid_under_jump = [candidate for candidate in candidates if candidate.within_jump_limit]
    pool = valid_under_jump if valid_under_jump else list(candidates)
    winner = min(pool, key=candidate_sort_key)
    return winner, {
        'valid_total': len(candidates),
        'valid_deterministic': sum(
            1 for candidate in candidates if candidate.seed_kind == 'deterministic'
        ),
        'valid_random': sum(1 for candidate in candidates if candidate.seed_kind == 'random'),
        'valid_under_jump': len(valid_under_jump),
        'winner_pool': 'under_jump' if valid_under_jump else 'all',
    }
