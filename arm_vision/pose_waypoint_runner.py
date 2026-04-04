from __future__ import annotations

import csv
import datetime
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def _safe_name(value: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9._-]+', '_', value.strip())
    return text or 'sequence'


def _quat_array(rotation: Rotation) -> np.ndarray:
    quat = np.array(rotation.as_quat(), dtype=float)
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0], dtype=float)


def _rotation_error_deg(target: Rotation, actual: Rotation) -> float:
    return float(np.degrees((target * actual.inv()).magnitude()))


def pose_from_feedback(
    feedback: dict,
    fallback_pos: np.ndarray,
    fallback_rot: Rotation,
) -> tuple[np.ndarray, Rotation, str]:
    pos = np.array(fallback_pos, dtype=float)
    rot = fallback_rot
    source = 'command'

    if all(k in feedback for k in ('fk_x', 'fk_y', 'fk_z')):
        pos = np.array(
            [float(feedback['fk_x']), float(feedback['fk_y']), float(feedback['fk_z'])],
            dtype=float,
        )
        source = 'fk_pos'

    if all(k in feedback for k in ('fk_qx', 'fk_qy', 'fk_qz', 'fk_qw')):
        quat = np.array(
            [
                float(feedback['fk_qx']),
                float(feedback['fk_qy']),
                float(feedback['fk_qz']),
                float(feedback['fk_qw']),
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(quat))
        if norm > 1e-9:
            rot = Rotation.from_quat(quat / norm)
            source = 'fk_pose' if source == 'fk_pos' else 'fk_quat'

    return pos, rot, source


@dataclass(frozen=True)
class WaypointDef:
    name: str
    position: np.ndarray
    rotation: Rotation
    hold_s: float
    timeout_s: float
    pos_tolerance_m: float
    ori_tolerance_deg: float


@dataclass(frozen=True)
class WaypointSequence:
    name: str
    frame: str
    description: str
    waypoints: list[WaypointDef]


@dataclass(frozen=True)
class ResolvedWaypoint:
    name: str
    position: np.ndarray
    rotation: Rotation
    hold_s: float
    timeout_s: float
    pos_tolerance_m: float
    ori_tolerance_deg: float


class RunnerStatus(str, Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    SUCCEEDED = 'succeeded'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


def _load_rotation(raw: dict) -> Rotation:
    quat = raw.get('quat')
    if quat is not None:
        if len(quat) != 4:
            raise ValueError(f'Expected quat with 4 values, got {quat!r}')
        return Rotation.from_quat(np.array(quat, dtype=float))
    rpy_deg = raw.get('rpy_deg', [0.0, 0.0, 0.0])
    if len(rpy_deg) != 3:
        raise ValueError(f'Expected rpy_deg with 3 values, got {rpy_deg!r}')
    return Rotation.from_euler('xyz', np.array(rpy_deg, dtype=float), degrees=True)


def load_waypoint_sequences(path: str) -> dict[str, WaypointSequence]:
    with open(path, 'r') as f:
        raw_root = yaml.safe_load(f) or {}

    raw = raw_root.get('waypoints', raw_root)
    defaults = raw.get('defaults', {})
    sequences_raw = raw.get('sequences', {})
    if not sequences_raw:
        raise ValueError(f'No waypoint sequences found in {path}')

    default_frame = str(defaults.get('frame', 'relative_start'))
    default_hold_s = float(defaults.get('hold_s', 0.25))
    default_timeout_s = float(defaults.get('timeout_s', 5.0))
    default_pos_tol_m = float(defaults.get('pos_tolerance_mm', 5.0)) / 1000.0
    default_ori_tol_deg = float(defaults.get('ori_tolerance_deg', 5.0))

    sequences: dict[str, WaypointSequence] = {}
    for name, seq_raw in sequences_raw.items():
        frame = str(seq_raw.get('frame', default_frame))
        if frame not in ('relative_start', 'absolute'):
            raise ValueError(f'Sequence {name!r} has unsupported frame {frame!r}')

        description = str(seq_raw.get('description', ''))
        waypoints_raw = seq_raw.get('waypoints', [])
        if not waypoints_raw:
            raise ValueError(f'Sequence {name!r} has no waypoints')

        waypoints: list[WaypointDef] = []
        for i, wp_raw in enumerate(waypoints_raw, start=1):
            position = wp_raw.get('position')
            if position is None or len(position) != 3:
                raise ValueError(
                    f'Sequence {name!r} waypoint {i} must define position: [x, y, z]'
                )
            hold_s = float(wp_raw.get('hold_s', seq_raw.get('hold_s', default_hold_s)))
            timeout_s = float(wp_raw.get('timeout_s', seq_raw.get('timeout_s', default_timeout_s)))
            pos_tolerance_m = (
                float(wp_raw.get('pos_tolerance_mm', seq_raw.get('pos_tolerance_mm', defaults.get('pos_tolerance_mm', 5.0))))
                / 1000.0
            )
            ori_tolerance_deg = float(
                wp_raw.get('ori_tolerance_deg', seq_raw.get('ori_tolerance_deg', default_ori_tol_deg))
            )
            waypoints.append(
                WaypointDef(
                    name=str(wp_raw.get('name', f'wp{i}')),
                    position=np.array(position, dtype=float),
                    rotation=_load_rotation(wp_raw),
                    hold_s=hold_s,
                    timeout_s=timeout_s,
                    pos_tolerance_m=pos_tolerance_m,
                    ori_tolerance_deg=ori_tolerance_deg,
                )
            )

        sequences[str(name)] = WaypointSequence(
            name=str(name),
            frame=frame,
            description=description,
            waypoints=waypoints,
        )

    return sequences


class PoseWaypointRunner:
    def __init__(self, sequences: dict[str, WaypointSequence], log_dir: str = '') -> None:
        if not sequences:
            raise ValueError('PoseWaypointRunner requires at least one sequence')

        self._sequences = dict(sequences)
        self._sequence_names = list(sequences.keys())
        self._selected_idx = 0
        self._log_dir = log_dir

        self._status = RunnerStatus.IDLE
        self._resolved_waypoints: list[ResolvedWaypoint] = []
        self._current_idx = 0
        self._waypoint_started_at = 0.0
        self._hold_started_at: Optional[float] = None
        self._last_command_pos: Optional[np.ndarray] = None
        self._last_command_rot: Optional[Rotation] = None
        self._last_actual_pos: Optional[np.ndarray] = None
        self._last_actual_rot: Optional[Rotation] = None
        self._last_feedback_source = 'command'
        self._last_pos_err_mm: Optional[float] = None
        self._last_ori_err_deg: Optional[float] = None
        self._last_ik_result = ''
        self._last_fail_cause = ''
        self._last_status_detail = ''
        self._last_log_path = ''

        self._log_file = None
        self._log_writer = None
        self._start_pos: Optional[np.ndarray] = None
        self._start_rot: Optional[Rotation] = None

    @property
    def sequence_names(self) -> list[str]:
        return list(self._sequence_names)

    @property
    def selected_name(self) -> str:
        return self._sequence_names[self._selected_idx]

    @property
    def selected_sequence(self) -> WaypointSequence:
        return self._sequences[self.selected_name]

    @property
    def is_active(self) -> bool:
        return self._status == RunnerStatus.RUNNING and bool(self._resolved_waypoints)

    @property
    def status(self) -> RunnerStatus:
        return self._status

    @property
    def last_log_path(self) -> str:
        return self._last_log_path

    @property
    def current_target_pose(self) -> tuple[Optional[np.ndarray], Optional[Rotation]]:
        if self._last_command_pos is None or self._last_command_rot is None:
            return None, None
        return self._last_command_pos.copy(), self._last_command_rot

    @property
    def status_text(self) -> str:
        seq_name = self.selected_name
        if self.is_active and self._resolved_waypoints:
            wp = self._resolved_waypoints[self._current_idx]
            detail = ''
            if self._last_pos_err_mm is not None and self._last_ori_err_deg is not None:
                detail = (
                    f' pos={self._last_pos_err_mm:.1f}mm'
                    f' ori={self._last_ori_err_deg:.1f}deg'
                    f' src={self._last_feedback_source}'
                )
            if self._last_ik_result:
                detail += f' ik={self._last_ik_result}'
            return (
                f'{seq_name} {self._current_idx + 1}/{len(self._resolved_waypoints)}'
                f' {wp.name}{detail}'
            )

        if self._last_status_detail:
            return f'{seq_name} {self._status.value}: {self._last_status_detail}'
        return f'{seq_name} {self._status.value}'

    def select_next_sequence(self) -> str:
        if self.is_active:
            return self.selected_name
        self._selected_idx = (self._selected_idx + 1) % len(self._sequence_names)
        self._status = RunnerStatus.IDLE
        self._last_status_detail = ''
        return self.selected_name

    def start(self, start_pos: np.ndarray, start_rot: Rotation, now: Optional[float] = None) -> bool:
        if self.is_active:
            return False

        now = time.monotonic() if now is None else float(now)
        sequence = self.selected_sequence
        self._resolved_waypoints = self._resolve_waypoints(sequence, start_pos, start_rot)
        self._current_idx = 0
        self._status = RunnerStatus.RUNNING
        self._waypoint_started_at = now
        self._hold_started_at = None
        self._start_pos = np.array(start_pos, dtype=float)
        self._start_rot = start_rot
        self._last_command_pos = self._resolved_waypoints[0].position.copy()
        self._last_command_rot = self._resolved_waypoints[0].rotation
        self._last_actual_pos = None
        self._last_actual_rot = None
        self._last_feedback_source = 'command'
        self._last_pos_err_mm = None
        self._last_ori_err_deg = None
        self._last_ik_result = ''
        self._last_fail_cause = ''
        self._last_status_detail = self._resolved_waypoints[0].name
        self._open_log(now)
        self._log_event('sequence_start', now, {})
        self._log_event('waypoint_start', now, {})
        return True

    def cancel(
        self,
        reason: str,
        feedback: dict,
        fallback_pos: np.ndarray,
        fallback_rot: Rotation,
        now: Optional[float] = None,
    ) -> None:
        now = time.monotonic() if now is None else float(now)
        actual_pos, actual_rot, source = pose_from_feedback(feedback, fallback_pos, fallback_rot)
        self._last_command_pos = np.array(actual_pos, dtype=float)
        self._last_command_rot = actual_rot
        self._last_actual_pos = np.array(actual_pos, dtype=float)
        self._last_actual_rot = actual_rot
        self._last_feedback_source = source
        self._last_status_detail = reason
        self._last_ik_result = str(feedback.get('ik_result', ''))
        self._last_fail_cause = str(feedback.get('ik_fail_cause', ''))
        if self.is_active:
            self._status = RunnerStatus.CANCELLED
            self._log_event('sequence_cancelled', now, feedback)
        self._resolved_waypoints = []
        self._hold_started_at = None
        self._close_log()

    def tick(
        self,
        feedback: dict,
        fallback_pos: np.ndarray,
        fallback_rot: Rotation,
        now: Optional[float] = None,
    ) -> tuple[np.ndarray, Rotation]:
        if not self.is_active:
            if self._last_command_pos is not None and self._last_command_rot is not None:
                return self._last_command_pos.copy(), self._last_command_rot
            return np.array(fallback_pos, dtype=float), fallback_rot

        now = time.monotonic() if now is None else float(now)
        waypoint = self._resolved_waypoints[self._current_idx]
        actual_pos, actual_rot, source = pose_from_feedback(feedback, fallback_pos, fallback_rot)
        self._last_actual_pos = np.array(actual_pos, dtype=float)
        self._last_actual_rot = actual_rot
        self._last_feedback_source = source
        self._last_command_pos = waypoint.position.copy()
        self._last_command_rot = waypoint.rotation
        self._last_ik_result = str(feedback.get('ik_result', ''))
        self._last_fail_cause = str(feedback.get('ik_fail_cause', ''))

        pos_err_m = float(np.linalg.norm(waypoint.position - actual_pos))
        ori_err_deg = _rotation_error_deg(waypoint.rotation, actual_rot)
        self._last_pos_err_mm = pos_err_m * 1000.0
        self._last_ori_err_deg = ori_err_deg

        within_pos = pos_err_m <= waypoint.pos_tolerance_m
        within_ori = ori_err_deg <= waypoint.ori_tolerance_deg
        hold_elapsed = 0.0
        event = 'tick'

        if within_pos and within_ori:
            if self._hold_started_at is None:
                self._hold_started_at = now
                event = 'hold_start'
            hold_elapsed = now - self._hold_started_at
            if hold_elapsed >= waypoint.hold_s:
                event = 'waypoint_reached'
                self._last_status_detail = waypoint.name
                self._log_event(event, now, feedback)
                if self._current_idx + 1 >= len(self._resolved_waypoints):
                    self._status = RunnerStatus.SUCCEEDED
                    self._last_status_detail = waypoint.name
                    self._resolved_waypoints = []
                    self._hold_started_at = None
                    self._log_event('sequence_complete', now, feedback)
                    self._close_log()
                    return self._last_command_pos.copy(), self._last_command_rot

                self._current_idx += 1
                next_wp = self._resolved_waypoints[self._current_idx]
                self._waypoint_started_at = now
                self._hold_started_at = None
                self._last_command_pos = next_wp.position.copy()
                self._last_command_rot = next_wp.rotation
                self._last_status_detail = next_wp.name
                self._log_event('waypoint_start', now, feedback)
                return self._last_command_pos.copy(), self._last_command_rot
        else:
            if self._hold_started_at is not None:
                event = 'hold_reset'
            self._hold_started_at = None

        if (now - self._waypoint_started_at) > waypoint.timeout_s:
            self._status = RunnerStatus.FAILED
            self._last_status_detail = f'{waypoint.name} timeout'
            self._log_event('waypoint_timeout', now, feedback)
            self._resolved_waypoints = []
            self._hold_started_at = None
            self._close_log()
            return self._last_command_pos.copy(), self._last_command_rot

        self._log_event(event, now, feedback)
        return self._last_command_pos.copy(), self._last_command_rot

    def _resolve_waypoints(
        self,
        sequence: WaypointSequence,
        start_pos: np.ndarray,
        start_rot: Rotation,
    ) -> list[ResolvedWaypoint]:
        resolved: list[ResolvedWaypoint] = []
        start_pos = np.array(start_pos, dtype=float)
        for wp in sequence.waypoints:
            if sequence.frame == 'relative_start':
                position = start_pos + wp.position
                rotation = start_rot * wp.rotation
            else:
                position = np.array(wp.position, dtype=float)
                rotation = wp.rotation
            resolved.append(
                ResolvedWaypoint(
                    name=wp.name,
                    position=np.array(position, dtype=float),
                    rotation=rotation,
                    hold_s=wp.hold_s,
                    timeout_s=wp.timeout_s,
                    pos_tolerance_m=wp.pos_tolerance_m,
                    ori_tolerance_deg=wp.ori_tolerance_deg,
                )
            )
        return resolved

    def _open_log(self, now: float) -> None:
        self._close_log()
        self._last_log_path = ''
        if not self._log_dir:
            return

        os.makedirs(self._log_dir, exist_ok=True)
        seq_name = _safe_name(self.selected_name)
        wall_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._last_log_path = os.path.join(self._log_dir, f'waypoints_{seq_name}_{wall_time}.csv')
        self._log_file = open(self._last_log_path, 'w', newline='', buffering=1)
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow([
            'wall_time',
            'mono_s',
            'event',
            'status',
            'sequence',
            'frame',
            'waypoint_index',
            'waypoint_name',
            'feedback_source',
            'start_x',
            'start_y',
            'start_z',
            'start_qx',
            'start_qy',
            'start_qz',
            'start_qw',
            'cmd_x',
            'cmd_y',
            'cmd_z',
            'cmd_qx',
            'cmd_qy',
            'cmd_qz',
            'cmd_qw',
            'actual_x',
            'actual_y',
            'actual_z',
            'actual_qx',
            'actual_qy',
            'actual_qz',
            'actual_qw',
            'pos_err_mm',
            'ori_err_deg',
            'hold_elapsed_s',
            'hold_target_s',
            'waypoint_elapsed_s',
            'waypoint_timeout_s',
            'pos_tolerance_mm',
            'ori_tolerance_deg',
            'ik_ok',
            'ik_result',
            'ik_seed',
            'ik_seed_kind',
            'ik_pool',
            'ik_fail_cause',
            'ik_continuity_cost',
            'ik_joint_state_cost',
            'ik_posture_cost',
            'ik_max_jump_rad',
            'ik6d_time_budget_hit',
            'ik6d_best_failed_seed',
            'ik6d_best_failed_seed_kind',
            'ik6d_best_failed_reason',
            'ik6d_best_failed_class',
            'ik6d_best_failed_pos_err_mm',
            'ik6d_best_failed_ori_err_deg',
            'ik6d_best_failed_sigma_min',
            'ik6d_best_failed_singular',
            'log_path',
        ])
        self._log_file.flush()

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
        self._log_file = None
        self._log_writer = None

    def _log_event(self, event: str, now: float, feedback: dict) -> None:
        if self._log_writer is None:
            return

        cmd_pos = self._last_command_pos if self._last_command_pos is not None else np.zeros(3, dtype=float)
        cmd_rot = self._last_command_rot if self._last_command_rot is not None else Rotation.identity()
        act_pos = self._last_actual_pos if self._last_actual_pos is not None else cmd_pos
        act_rot = self._last_actual_rot if self._last_actual_rot is not None else cmd_rot
        start_pos = self._start_pos if self._start_pos is not None else cmd_pos
        start_rot = self._start_rot if self._start_rot is not None else cmd_rot
        cmd_quat = _quat_array(cmd_rot)
        act_quat = _quat_array(act_rot)
        start_quat = _quat_array(start_rot)

        waypoint_name = ''
        waypoint_index = ''
        hold_elapsed_s = ''
        hold_target_s = ''
        waypoint_elapsed_s = ''
        waypoint_timeout_s = ''
        pos_tolerance_mm = ''
        ori_tolerance_deg = ''
        if self.is_active and self._resolved_waypoints:
            waypoint = self._resolved_waypoints[self._current_idx]
            waypoint_name = waypoint.name
            waypoint_index = self._current_idx + 1
            hold_elapsed_s = (
                f'{max(0.0, now - self._hold_started_at):.3f}'
                if self._hold_started_at is not None else '0.000'
            )
            hold_target_s = f'{waypoint.hold_s:.3f}'
            waypoint_elapsed_s = f'{max(0.0, now - self._waypoint_started_at):.3f}'
            waypoint_timeout_s = f'{waypoint.timeout_s:.3f}'
            pos_tolerance_mm = f'{waypoint.pos_tolerance_m * 1000.0:.3f}'
            ori_tolerance_deg = f'{waypoint.ori_tolerance_deg:.3f}'

        self._log_writer.writerow([
            datetime.datetime.now().isoformat(timespec='milliseconds'),
            f'{now:.6f}',
            event,
            self._status.value,
            self.selected_name,
            self.selected_sequence.frame,
            waypoint_index,
            waypoint_name,
            self._last_feedback_source,
            f'{start_pos[0]:.6f}',
            f'{start_pos[1]:.6f}',
            f'{start_pos[2]:.6f}',
            f'{start_quat[0]:.6f}',
            f'{start_quat[1]:.6f}',
            f'{start_quat[2]:.6f}',
            f'{start_quat[3]:.6f}',
            f'{cmd_pos[0]:.6f}',
            f'{cmd_pos[1]:.6f}',
            f'{cmd_pos[2]:.6f}',
            f'{cmd_quat[0]:.6f}',
            f'{cmd_quat[1]:.6f}',
            f'{cmd_quat[2]:.6f}',
            f'{cmd_quat[3]:.6f}',
            f'{act_pos[0]:.6f}',
            f'{act_pos[1]:.6f}',
            f'{act_pos[2]:.6f}',
            f'{act_quat[0]:.6f}',
            f'{act_quat[1]:.6f}',
            f'{act_quat[2]:.6f}',
            f'{act_quat[3]:.6f}',
            '' if self._last_pos_err_mm is None else f'{self._last_pos_err_mm:.3f}',
            '' if self._last_ori_err_deg is None else f'{self._last_ori_err_deg:.3f}',
            hold_elapsed_s,
            hold_target_s,
            waypoint_elapsed_s,
            waypoint_timeout_s,
            pos_tolerance_mm,
            ori_tolerance_deg,
            feedback.get('ik_ok', ''),
            feedback.get('ik_result', ''),
            feedback.get('ik_seed', ''),
            feedback.get('ik_seed_kind', ''),
            feedback.get('ik_pool', ''),
            feedback.get('ik_fail_cause', ''),
            feedback.get('ik_continuity_cost', ''),
            feedback.get('ik_joint_state_cost', ''),
            feedback.get('ik_posture_cost', ''),
            feedback.get('ik_max_jump_rad', ''),
            feedback.get('ik6d_time_budget_hit', ''),
            feedback.get('ik6d_best_failed_seed', ''),
            feedback.get('ik6d_best_failed_seed_kind', ''),
            feedback.get('ik6d_best_failed_reason', ''),
            feedback.get('ik6d_best_failed_class', ''),
            feedback.get('ik6d_best_failed_pos_err_mm', ''),
            feedback.get('ik6d_best_failed_ori_err_deg', ''),
            feedback.get('ik6d_best_failed_sigma_min', ''),
            feedback.get('ik6d_best_failed_singular', ''),
            self._last_log_path,
        ])
        self._log_file.flush()
