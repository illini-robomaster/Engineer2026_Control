"""
Assembly FSM — shared-autonomy state machine for Energy Unit insertion.

The operator controls translation via SpaceMouse. This module manages stage
sequencing, automatic arc/lift execution, and confirmation gating.

Arc motions (P and Q) simultaneously interpolate both position and orientation
along a circular path so the EE tracks the correct tangent direction.

Usage:
    from assembly_fsm import AssemblyFSM, TaskConfig, load_assembly_config
    cfg = load_assembly_config('config/assembly_params.yaml')
    fsm = AssemblyFSM(cfg)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ── States ──────────────────────────────────────────────────────────────────

class AssemblyState(Enum):
    IDLE            = auto()  # no task active, manual teleop
    APPROACH        = auto()  # human: translate toward core
    INSERT          = auto()  # human: push into core
    LIFT            = auto()  # auto: move along EE x-axis
    AUTO_ROTATE_P   = auto()  # auto: 90° arc (169mm radius)
    AUTO_ROTATE_Q   = auto()  # auto: arc to target angle (60mm radius)
    RELEASE         = auto()  # human: withdraw EE backward along EE x-axis
    READY_CONFIRM   = auto()  # all stages done; hold pose
    CONFIRMED       = auto()  # confirmation issued, freeze everything
    ABORTED         = auto()  # user aborted


# States where SpaceMouse translation is applied
_TRANSLATION_ALLOWED = frozenset({
    AssemblyState.IDLE,
    AssemblyState.APPROACH,
    AssemblyState.INSERT,
})

# Stages where the user can press LEFT to advance
_ADVANCEABLE = frozenset({
    AssemblyState.APPROACH,
    AssemblyState.INSERT,
    AssemblyState.AUTO_ROTATE_Q,   # manual arc: LEFT when satisfied
    AssemblyState.RELEASE,         # LEFT when withdrawn far enough
})

# Difficulty → ordered stage sequence
DIFFICULTY_STAGES: dict[int, list[AssemblyState]] = {
    1: [AssemblyState.APPROACH, AssemblyState.INSERT,
        AssemblyState.READY_CONFIRM],
    2: [AssemblyState.APPROACH, AssemblyState.INSERT,
        AssemblyState.LIFT, AssemblyState.READY_CONFIRM],
    3: [AssemblyState.APPROACH, AssemblyState.INSERT,
        AssemblyState.LIFT, AssemblyState.AUTO_ROTATE_P,
        AssemblyState.AUTO_ROTATE_Q, AssemblyState.RELEASE,
        AssemblyState.READY_CONFIRM],
    4: [AssemblyState.APPROACH, AssemblyState.INSERT,
        AssemblyState.LIFT, AssemblyState.AUTO_ROTATE_P,
        AssemblyState.AUTO_ROTATE_Q, AssemblyState.RELEASE,
        AssemblyState.READY_CONFIRM],
}


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskConfig:
    difficulty: int
    q_target_deg: float           # Q-arc angle (difficulty 3/4)
    init_pitch_deg: float         # initial EE pitch (-90° = EE x → world +Z)
    init_yaw_deg: float           # approach yaw (0 = +X, 90 = +Y, etc.)
    rotation_speed_deg_s: float   # °/s for arc motions
    approach_orient_speed_deg_s: float  # °/s for IDLE→APPROACH orientation transition

    # Lift — axis in EE frame at lift start (transformed to world at runtime)
    lift_distance_mm: float
    lift_speed_m_s: float
    lift_axis_ee: np.ndarray         # EE-frame unit vec: direction to lift ([1,0,0] = EE x)

    # P arc: 90° pitch — EE x stays tangent (vectors in EE frame at arc start)
    p_arc_radius_mm: float
    p_arc_angle_deg: float
    p_arc_center_dir_ee: np.ndarray  # EE-frame unit vec: EE → pivot
    p_arc_axis_ee: np.ndarray        # EE-frame rotation axis

    # Q arc: pitch forward — EE x stays tangent (vectors in EE frame at arc start)
    q_arc_radius_mm: float
    q_arc_center_dir_ee: np.ndarray  # EE-frame unit vec: EE → pivot
    q_arc_axis_ee: np.ndarray        # EE-frame rotation axis

    # Confirmation
    stability_hold_s: float
    stability_threshold_m: float
    confirm_cooldown_s: float

    # Optional — default-valued fields must be last
    approach_speed_scale: float = 1.0   # SpaceMouse speed multiplier during APPROACH

    def __post_init__(self):
        if self.difficulty not in DIFFICULTY_STAGES:
            raise ValueError(f'Invalid difficulty: {self.difficulty}')


def load_assembly_config(
    path: str,
    difficulty_override: int | None = None,
    q_angle_override: float | None = None,
    yaw_override: float | None = None,
) -> TaskConfig:
    """Load assembly parameters from YAML, with optional CLI overrides."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)['assembly']

    difficulty = difficulty_override if difficulty_override is not None else raw['difficulty']
    q_target = q_angle_override if q_angle_override is not None else raw['q_target_deg']
    yaw = yaw_override if yaw_override is not None else raw.get('init_yaw_deg', 0.0)

    return TaskConfig(
        difficulty=difficulty,
        q_target_deg=q_target,
        init_pitch_deg=raw.get('init_pitch_deg', -90.0),
        init_yaw_deg=yaw,
        rotation_speed_deg_s=raw['rotation_speed_deg_s'],
        approach_orient_speed_deg_s=raw.get('approach_orient_speed_deg_s', 45.0),
        approach_speed_scale=raw.get('approach_speed_scale', 1.0),
        lift_distance_mm=raw['lift_distance_mm'],
        lift_speed_m_s=raw.get('lift_speed_m_s', 0.05),
        lift_axis_ee=np.array(raw.get('lift_axis_ee', [1.0, 0.0, 0.0]), dtype=float),
        p_arc_radius_mm=raw['p_arc_radius_mm'],
        p_arc_angle_deg=raw.get('p_arc_angle_deg', 90.0),
        p_arc_center_dir_ee=np.array(raw['p_arc_center_dir_ee'], dtype=float),
        p_arc_axis_ee=np.array(raw['p_arc_axis_ee'], dtype=float),
        q_arc_radius_mm=raw['q_arc_radius_mm'],
        q_arc_center_dir_ee=np.array(raw['q_arc_center_dir_ee'], dtype=float),
        q_arc_axis_ee=np.array(raw['q_arc_axis_ee'], dtype=float),
        stability_hold_s=raw['stability_hold_s'],
        stability_threshold_m=raw['stability_threshold_m'],
        confirm_cooldown_s=raw['confirm_cooldown_s'],
    )


# ── Interpolators ───────────────────────────────────────────────────────────

class RotationSlerp:
    """Smooth SLERP between two orientations at a given angular speed."""

    def __init__(self, start: Rotation, target: Rotation,
                 speed_deg_s: float) -> None:
        self._start = start
        self._target = target
        self._delta = start.inv() * target
        angle_rad = self._delta.magnitude()
        self._duration = max(np.degrees(angle_rad) / speed_deg_s, 0.1)
        self._elapsed = 0.0

    @property
    def is_complete(self) -> bool:
        return self._elapsed >= self._duration

    def tick(self, dt: float) -> tuple[Rotation, bool]:
        self._elapsed = min(self._elapsed + dt, self._duration)
        alpha = self._elapsed / self._duration
        interp = self._start * Rotation.from_rotvec(
            self._delta.as_rotvec() * alpha)
        return interp, self.is_complete


class LinearInterpolator:
    """Straight-line position interpolation at a given speed."""

    def __init__(self, start: np.ndarray, target: np.ndarray,
                 speed_m_s: float) -> None:
        self._start = start.copy()
        self._target = target.copy()
        distance = float(np.linalg.norm(target - start))
        self._duration = max(distance / speed_m_s, 0.1)
        self._elapsed = 0.0

    @property
    def is_complete(self) -> bool:
        return self._elapsed >= self._duration

    def tick(self, dt: float) -> tuple[np.ndarray, bool]:
        self._elapsed = min(self._elapsed + dt, self._duration)
        alpha = self._elapsed / self._duration
        pos = self._start + alpha * (self._target - self._start)
        return pos, self.is_complete


class ArcInterpolator:
    """Circular arc — position and orientation change together.

    The EE traces a circular path around a pivot point.  Both position and
    orientation are rotated by the same incremental rotation so the
    specified EE axis remains tangent to the arc at all times.
    """

    def __init__(self, start_pos: np.ndarray, start_rot: Rotation,
                 pivot: np.ndarray, axis: np.ndarray,
                 angle_deg: float, speed_deg_s: float) -> None:
        self._pivot = pivot.copy()
        self._offset_0 = start_pos - pivot         # radius vector at t=0
        self._start_rot = start_rot
        self._axis = axis / np.linalg.norm(axis)    # ensure unit
        self._total_rad = np.radians(angle_deg)
        self._duration = max(abs(angle_deg) / speed_deg_s, 0.3)
        self._elapsed = 0.0

    @property
    def is_complete(self) -> bool:
        return self._elapsed >= self._duration

    def tick(self, dt: float) -> tuple[np.ndarray, Rotation, bool]:
        """Returns (position, orientation, is_complete)."""
        self._elapsed = min(self._elapsed + dt, self._duration)
        alpha = self._elapsed / self._duration
        theta = alpha * self._total_rad

        arc_rot = Rotation.from_rotvec(self._axis * theta)
        new_pos = self._pivot + arc_rot.apply(self._offset_0)
        new_rot = arc_rot * self._start_rot

        return new_pos, new_rot, self.is_complete


# ── Stage completion checkers (modular stubs) ───────────────────────────────

class StageCheckers:
    """Modular completion-check functions.

    Each method is a pure function that can be monkey-patched with a
    better implementation without touching the FSM.
    """

    @staticmethod
    def check_stability(recent_positions: deque, threshold_m: float) -> bool:
        if len(recent_positions) < 2:
            return False
        positions = np.array(recent_positions)
        spread = np.max(positions, axis=0) - np.min(positions, axis=0)
        return bool(np.all(spread < threshold_m))

    @staticmethod
    def check_sm_inactive(sm_lin_magnitude: float, deadband: float) -> bool:
        return sm_lin_magnitude < deadband


# ── FSM ─────────────────────────────────────────────────────────────────────

class AssemblyFSM:
    """Shared-autonomy assembly state machine.

    The main loop calls:
      - query properties (state, translation_allowed, position_override, …)
      - action methods (start, advance, abort, confirm) on button presses
      - tick() every iteration to advance auto stages
    """

    def __init__(self, config: TaskConfig) -> None:
        self._cfg = config
        self._stages = DIFFICULTY_STAGES[config.difficulty]
        self._stage_idx = 0
        self._state = AssemblyState.IDLE

        # Interpolator state (only one active at a time)
        self._lift_interp: Optional[LinearInterpolator] = None
        self._arc_interp: Optional[ArcInterpolator] = None

        # Current interpolated values (read by overrides)
        self._current_pos: Optional[np.ndarray] = None
        self._current_rot: Optional[Rotation] = None

        # APPROACH orientation: pitch ramps automatically, yaw from SpaceMouse
        self._approach_pitch_deg: float = 0.0
        self._approach_yaw_deg: float = 0.0

        # Manual Q arc state (initialized on first tick of AUTO_ROTATE_Q)
        self._q_manual_initialized: bool = False
        self._q_pivot: Optional[np.ndarray] = None
        self._q_axis_world: Optional[np.ndarray] = None
        self._q_offset_0: Optional[np.ndarray] = None
        self._q_start_rot: Optional[Rotation] = None
        self._q_arc_angle: float = 0.0

        # RELEASE state: withdraw along EE x captured at RELEASE start
        self._release_initialized: bool = False
        self._release_ee_x: Optional[np.ndarray] = None

        # Stability tracking for READY_CONFIRM
        self._stability_timer = 0.0
        window_size = max(int(self._cfg.stability_hold_s * 100), 10)
        self._recent_positions: deque[np.ndarray] = deque(maxlen=window_size)

        # Stage-start position snapshot
        self._stage_start_pos: Optional[np.ndarray] = None

    # ── Query properties ────────────────────────────────────────────────────

    @property
    def state(self) -> AssemblyState:
        return self._state

    @property
    def init_orientation(self) -> Rotation:
        """EE orientation at task start.

        Composition: Rz(yaw) * Ry(-pitch).
          - Yaw selects the approach direction (0 = +X, 90 = +Y).
          - Pitch tilts EE x toward +Z (positive = nose up).
        Scipy Ry(+θ) maps x toward -Z, so we negate pitch.
        """
        R_yaw = Rotation.from_euler('z', self._cfg.init_yaw_deg, degrees=True)
        R_pitch = Rotation.from_euler('y', -self._cfg.init_pitch_deg, degrees=True)
        return R_yaw * R_pitch

    @property
    def approach_yaw_deg(self) -> float:
        """Current approach yaw angle (degrees)."""
        return self._approach_yaw_deg

    @property
    def approach_speed_scale(self) -> float:
        """SpaceMouse speed multiplier — reduced during APPROACH for fine alignment."""
        if self._state == AssemblyState.APPROACH:
            return self._cfg.approach_speed_scale
        return 1.0

    @property
    def translation_allowed(self) -> bool:
        return self._state in _TRANSLATION_ALLOWED

    @property
    def position_override(self) -> Optional[np.ndarray]:
        """Position from auto-lift, arc interpolation, or RELEASE withdrawal."""
        if self._state == AssemblyState.LIFT and self._lift_interp is not None:
            return self._current_pos
        if self._state == AssemblyState.AUTO_ROTATE_P and self._arc_interp is not None:
            return self._current_pos
        if self._state == AssemblyState.AUTO_ROTATE_Q and self._q_manual_initialized:
            return self._current_pos
        if self._state == AssemblyState.RELEASE and self._release_initialized:
            return self._current_pos
        return None

    @property
    def rotation_override(self) -> Optional[Rotation]:
        """Orientation from approach control, arc, or manual Q arc."""
        if self._state == AssemblyState.APPROACH:
            return self._current_rot
        if self._state == AssemblyState.AUTO_ROTATE_P and self._arc_interp is not None:
            return self._current_rot
        if self._state == AssemblyState.AUTO_ROTATE_Q and self._q_manual_initialized:
            return self._current_rot
        return None

    @property
    def stage_label(self) -> str:
        total = len(self._stages)
        idx = min(self._stage_idx + 1, total)
        return f'{self._state.name} [{idx}/{total}]'

    @property
    def can_confirm(self) -> bool:
        if self._state != AssemblyState.READY_CONFIRM:
            return False
        return self._stability_timer >= self._cfg.stability_hold_s

    @property
    def stage_hint(self) -> str:
        hints = {
            AssemblyState.IDLE: 'Press LEFT to begin assembly task.',
            AssemblyState.APPROACH: 'Translate + rotate yaw to align. LEFT when ready.',
            AssemblyState.INSERT: 'Push to insert. Press LEFT when seated.',
            AssemblyState.LIFT: 'Auto-lifting +Z...',
            AssemblyState.AUTO_ROTATE_P: f'Arc P (85mm, 90deg)...',
            AssemblyState.AUTO_ROTATE_Q: f'Roll arc Q (60mm, target {self._cfg.q_target_deg}deg) — push L/R',
            AssemblyState.RELEASE: 'Withdraw EE backward (EE -X). LEFT when clear.',
            AssemblyState.READY_CONFIRM: 'Hold steady. Press LEFT to confirm.',
            AssemblyState.CONFIRMED: 'Confirmed! Press LEFT to reset.',
            AssemblyState.ABORTED: 'Aborted. Press LEFT to reset.',
        }
        return hints.get(self._state, '')

    # ── Action methods ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._stage_idx = 0
        self._state = self._stages[0]
        self._stage_start_pos = None
        self._stability_timer = 0.0
        self._recent_positions.clear()
        self._lift_interp = None
        self._arc_interp = None
        self._q_manual_initialized = False
        self._q_arc_angle = 0.0

        # APPROACH orientation: start at init_yaw; pitch ramps from 0 → init_pitch
        self._approach_pitch_deg = 0.0
        self._approach_yaw_deg = self._cfg.init_yaw_deg
        self._current_rot = Rotation.from_euler('z', self._cfg.init_yaw_deg, degrees=True)

    def advance(self) -> None:
        if self._state not in _ADVANCEABLE:
            return
        self._move_to_next_stage()

    def abort(self) -> None:
        self._state = AssemblyState.ABORTED
        self._lift_interp = None
        self._arc_interp = None
        self._q_manual_initialized = False

    def confirm(self) -> None:
        if self.can_confirm:
            self._state = AssemblyState.CONFIRMED

    def reset(self) -> None:
        if self._state in (AssemblyState.CONFIRMED, AssemblyState.ABORTED):
            self._state = AssemblyState.IDLE
            self._lift_interp = None
            self._arc_interp = None
            self._stage_start_pos = None
            self._q_manual_initialized = False
            self._q_arc_angle = 0.0
            self._release_initialized = False
            self._release_ee_x = None

    def emergency_reset(self) -> None:
        """Immediately reset to IDLE from any state (e.g. right-button homing)."""
        self._state = AssemblyState.IDLE
        self._stage_idx = 0
        self._lift_interp = None
        self._arc_interp = None
        self._q_manual_initialized = False
        self._q_arc_angle = 0.0
        self._release_initialized = False
        self._release_ee_x = None
        self._stage_start_pos = None
        self._stability_timer = 0.0
        self._recent_positions.clear()

    # ── Tick ────────────────────────────────────────────────────────────────

    def tick(self, ee_pos: np.ndarray, ee_rot: Rotation,
             sm_active: bool, sm_lin_delta: np.ndarray,
             sm_rot_delta: np.ndarray, dt: float) -> None:
        if self._stage_start_pos is None and self._state in _ADVANCEABLE:
            self._stage_start_pos = ee_pos.copy()

        if self._state == AssemblyState.APPROACH:
            self._tick_approach(sm_rot_delta, dt)

        elif self._state == AssemblyState.LIFT:
            self._tick_auto_lift(ee_pos, ee_rot, dt)

        elif self._state == AssemblyState.AUTO_ROTATE_P:
            self._tick_arc_auto(ee_pos, ee_rot, dt)

        elif self._state == AssemblyState.AUTO_ROTATE_Q:
            self._tick_manual_arc_q(ee_pos, ee_rot, sm_lin_delta)

        elif self._state == AssemblyState.RELEASE:
            self._tick_release(ee_pos, ee_rot, sm_lin_delta)

        elif self._state == AssemblyState.READY_CONFIRM:
            self._recent_positions.append(ee_pos.copy())
            if (StageCheckers.check_stability(
                    self._recent_positions, self._cfg.stability_threshold_m)
                    and not sm_active):
                self._stability_timer += dt
            else:
                self._stability_timer = 0.0

    # ── Internal helpers ────────────────────────────────────────────────────

    def _move_to_next_stage(self) -> None:
        self._stage_idx += 1
        if self._stage_idx >= len(self._stages):
            self._state = AssemblyState.CONFIRMED
            return
        self._state = self._stages[self._stage_idx]
        self._stage_start_pos = None
        self._stability_timer = 0.0
        self._recent_positions.clear()

    def _tick_approach(self, sm_rot_delta: np.ndarray, dt: float) -> None:
        """Ramp pitch toward init_pitch; all three SpaceMouse axes modify orientation.

        Pitch auto-ramp is applied as an incremental world-Y rotation so it
        composes naturally with any roll/yaw the operator is applying simultaneously.
        SpaceMouse angular deltas (roll, pitch, yaw) are then applied on top as
        small incremental world-frame rotations.
        """
        target = self._cfg.init_pitch_deg
        speed = self._cfg.approach_orient_speed_deg_s
        if self._approach_pitch_deg < target:
            delta_pitch = min(speed * dt, target - self._approach_pitch_deg)
            self._approach_pitch_deg += delta_pitch
            ramp = Rotation.from_euler('y', -delta_pitch, degrees=True)
            self._current_rot = ramp * self._current_rot

        # World-frame left-multiply — identical to manual mode.
        # Because the EE is pitched 90° (EE x → world Z), the same physical
        # SpaceMouse gestures naturally produce the right rotations:
        #   sm.roll  (tilt L/R)  → world X  (tips the insertion axis sideways)
        #   sm.pitch (tilt F/B)  → world Y  (tips the insertion axis fwd/back)
        #   sm.yaw   (twist)     → world -Z (spins around the vertical/insertion axis)
        self._current_rot = Rotation.from_rotvec(sm_rot_delta) * self._current_rot

        # Track yaw: world-Z rotation is driven by sm_rot_delta[2] (= -sm.yaw via ang_map).
        self._approach_yaw_deg += np.degrees(sm_rot_delta[2])

    def _tick_auto_lift(self, ee_pos: np.ndarray, ee_rot: Rotation,
                        dt: float) -> None:
        if self._lift_interp is None:
            # Compute lift direction in world frame from EE frame at lift start.
            # lift_axis_ee = [1,0,0] (EE x) → world +Z when pitch=90°, so
            # default behavior is preserved; any custom approach orientation
            # is followed automatically.
            lift_dir = ee_rot.apply(self._cfg.lift_axis_ee)
            lift_dir /= np.linalg.norm(lift_dir)
            target = ee_pos + lift_dir * (self._cfg.lift_distance_mm / 1000.0)
            self._lift_interp = LinearInterpolator(
                ee_pos, target, self._cfg.lift_speed_m_s)
            self._current_pos = ee_pos.copy()
            return

        pos, done = self._lift_interp.tick(dt)
        self._current_pos = pos
        if done:
            self._lift_interp = None
            self._move_to_next_stage()

    def _tick_arc_auto(self, ee_pos: np.ndarray, ee_rot: Rotation,
                       dt: float) -> None:
        """Automatic arc interpolation — used for AUTO_ROTATE_P."""
        if self._arc_interp is None:
            self._arc_interp = self._make_arc(ee_pos, ee_rot)
            self._current_pos = ee_pos.copy()
            self._current_rot = ee_rot
            return

        pos, rot, done = self._arc_interp.tick(dt)
        self._current_pos = pos
        self._current_rot = rot
        if done:
            self._arc_interp = None
            self._move_to_next_stage()

    def _tick_manual_arc_q(self, ee_pos: np.ndarray, ee_rot: Rotation,
                            sm_lin_delta: np.ndarray) -> None:
        """Manual arc — SpaceMouse left/right drives arc angle.

        EE y stays tangent to the 60mm circle.  The input is projected onto
        the current tangent direction so control is consistent regardless of
        how far the arc has swept.
        """
        if not self._q_manual_initialized:
            cfg = self._cfg
            radius = cfg.q_arc_radius_mm / 1000.0
            center_world = ee_rot.apply(cfg.q_arc_center_dir_ee)
            self._q_axis_world = ee_rot.apply(cfg.q_arc_axis_ee)
            self._q_pivot      = ee_pos + center_world * radius
            self._q_offset_0   = ee_pos - self._q_pivot
            self._q_start_rot  = ee_rot
            self._q_arc_angle  = 0.0
            self._current_pos  = ee_pos.copy()
            self._current_rot  = ee_rot
            self._q_manual_initialized = True
            return

        radius = self._cfg.q_arc_radius_mm / 1000.0

        # Tangent at current angle: axis × radius_vector (normalized)
        arc_rot_curr = Rotation.from_rotvec(self._q_axis_world * self._q_arc_angle)
        r_vec = arc_rot_curr.apply(self._q_offset_0)
        tangent = np.cross(self._q_axis_world, r_vec)
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-10:
            return
        tangent /= t_norm

        # Project SpaceMouse delta onto tangent → arc length → angle
        d_arc = float(np.dot(sm_lin_delta, tangent))
        self._q_arc_angle += d_arc / radius

        arc_rot = Rotation.from_rotvec(self._q_axis_world * self._q_arc_angle)
        self._current_pos = self._q_pivot + arc_rot.apply(self._q_offset_0)
        self._current_rot = arc_rot * self._q_start_rot

    def _tick_release(self, ee_pos: np.ndarray, ee_rot: Rotation,
                      sm_lin_delta: np.ndarray) -> None:
        """Manual withdrawal along EE x-axis.

        On first tick: captures EE x direction in world frame from current ee_rot.
        Each tick: projects SpaceMouse delta onto that axis and updates position.
        Rotation is unchanged (no rotation_override returned for RELEASE).
        LEFT button advances to READY_CONFIRM.
        """
        if not self._release_initialized:
            ee_x = ee_rot.apply(np.array([1.0, 0.0, 0.0]))
            self._release_ee_x = ee_x / np.linalg.norm(ee_x)
            self._current_pos = ee_pos.copy()
            self._release_initialized = True
            return

        # Project SpaceMouse delta onto EE x (forward = positive, backward = negative)
        d_x = float(np.dot(sm_lin_delta, self._release_ee_x))
        self._current_pos = self._current_pos + d_x * self._release_ee_x

    def _make_arc(self, ee_pos: np.ndarray,
                  ee_rot: Rotation) -> ArcInterpolator:
        """Build the ArcInterpolator for the current P or Q stage.

        Pivot and axis are defined in EE frame and transformed to world frame
        at arc-start time, so a different approach yaw is followed automatically.
        """
        cfg = self._cfg
        if self._state == AssemblyState.AUTO_ROTATE_P:
            radius = cfg.p_arc_radius_mm / 1000.0
            center_world = ee_rot.apply(cfg.p_arc_center_dir_ee)
            axis_world   = ee_rot.apply(cfg.p_arc_axis_ee)
            pivot = ee_pos + center_world * radius
            return ArcInterpolator(
                ee_pos, ee_rot, pivot,
                axis=axis_world,
                angle_deg=cfg.p_arc_angle_deg,
                speed_deg_s=cfg.rotation_speed_deg_s,
            )
        else:  # AUTO_ROTATE_Q
            radius = cfg.q_arc_radius_mm / 1000.0
            center_world = ee_rot.apply(cfg.q_arc_center_dir_ee)
            axis_world   = ee_rot.apply(cfg.q_arc_axis_ee)
            pivot = ee_pos + center_world * radius
            return ArcInterpolator(
                ee_pos, ee_rot, pivot,
                axis=axis_world,
                angle_deg=cfg.q_target_deg,
                speed_deg_s=cfg.rotation_speed_deg_s,
            )
