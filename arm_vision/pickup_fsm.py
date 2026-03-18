"""
Pickup FSM — shared-autonomy state machine for grabbing energy units from a
passive radial storage fixture (R≈130mm, center axis parallel to ground).

States:
    IDLE → APPROACH → ARC_ALIGN → LIFT_EE → CONFIRM → CONFIRMED
                                                      → ABORTED

- APPROACH:   operator translates EE toward fixture; pitch auto-ramps; yaw fixed.
- ARC_ALIGN:  SpaceMouse X-axis (push forward/back) drives arc angle around
              the fixture; EE x stays horizontal throughout.
- LIFT_EE:    auto: move EE 80mm along its own z-axis (lifts off the slot).
- CONFIRM:    stability gate; LEFT to confirm.

Usage:
    from pickup_fsm import PickupFSM, PickupConfig, load_pickup_config
    cfg = load_pickup_config('config/pickup_params.yaml')
    fsm = PickupFSM(cfg)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from assembly_fsm import LinearInterpolator, StageCheckers


# ── States ──────────────────────────────────────────────────────────────────

class PickupState(Enum):
    IDLE      = auto()   # waiting for LEFT to start
    APPROACH  = auto()   # operator translates toward fixture; pitch auto-ramps
    ARC_ALIGN = auto()   # SpaceMouse X drives arc along fixture circumference
    LIFT_EE   = auto()   # auto: move 80mm along EE z-axis
    CONFIRM   = auto()   # hold steady; LEFT to confirm after stability gate
    CONFIRMED = auto()   # confirmed, freeze everything
    ABORTED   = auto()   # user aborted


_TRANSLATION_ALLOWED = frozenset({
    PickupState.IDLE,
    PickupState.APPROACH,
})

_ACTIVE_STAGES = [
    PickupState.APPROACH,
    PickupState.ARC_ALIGN,
    PickupState.LIFT_EE,
    PickupState.CONFIRM,
]


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PickupConfig:
    init_pitch_deg: float                # 0.0 — EE x stays horizontal (device axis parallel to ground)
    init_yaw_deg: float                  # fixed approach direction; no yaw control
    approach_orient_speed_deg_s: float   # pitch ramp speed (°/s)
    arc_radius_mm: float                 # fixture radius (130mm)
    arc_center_dir_ee: np.ndarray        # EE-frame unit vec: EE → arc pivot
    arc_axis_ee: np.ndarray              # EE-frame rotation axis for arc
    lift_distance_mm: float              # 80.0 — EE z-axis lift after arc align
    lift_speed_m_s: float                # m/s for LIFT_EE
    stability_hold_s: float
    stability_threshold_m: float


def load_pickup_config(path: str) -> PickupConfig:
    """Load pickup parameters from YAML."""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)['pickup']
    return PickupConfig(
        init_pitch_deg=raw.get('init_pitch_deg', 0.0),
        init_yaw_deg=raw.get('init_yaw_deg', 0.0),
        approach_orient_speed_deg_s=raw.get('approach_orient_speed_deg_s', 45.0),
        arc_radius_mm=raw['arc_radius_mm'],
        arc_center_dir_ee=np.array(raw['arc_center_dir_ee'], dtype=float),
        arc_axis_ee=np.array(raw['arc_axis_ee'], dtype=float),
        lift_distance_mm=raw.get('lift_distance_mm', 80.0),
        lift_speed_m_s=raw.get('lift_speed_m_s', 0.05),
        stability_hold_s=raw['stability_hold_s'],
        stability_threshold_m=raw['stability_threshold_m'],
    )


# ── FSM ─────────────────────────────────────────────────────────────────────

class PickupFSM:
    """Shared-autonomy pickup state machine.

    Interface mirrors AssemblyFSM for drop-in use in the teleop main loop:
      - Query: state, translation_allowed, position_override, rotation_override,
               stage_label, stage_hint, can_confirm, approach_yaw_deg
      - Actions: start(), advance(), confirm(), reset(), emergency_reset()
      - Tick: tick(ee_pos, ee_rot, sm_active, sm_lin_delta, sm_yaw_delta, dt)
    """

    def __init__(self, config: PickupConfig) -> None:
        self._cfg = config
        self._state = PickupState.IDLE

        # APPROACH orientation
        self._approach_pitch_deg: float = 0.0
        self._current_rot: Optional[Rotation] = None
        self._current_pos: Optional[np.ndarray] = None

        # ARC_ALIGN state
        self._arc_initialized: bool = False
        self._arc_pivot: Optional[np.ndarray] = None
        self._arc_axis_world: Optional[np.ndarray] = None
        self._arc_offset_0: Optional[np.ndarray] = None
        self._arc_start_rot: Optional[Rotation] = None
        self._arc_angle: float = 0.0

        # LIFT_EE state
        self._lift_interp: Optional[LinearInterpolator] = None
        self._lift_rot: Optional[Rotation] = None  # frozen EE orientation during lift

        # CONFIRM stability tracking
        self._stability_timer: float = 0.0
        window_size = max(int(config.stability_hold_s * 100), 10)
        self._recent_positions: deque[np.ndarray] = deque(maxlen=window_size)

    # ── Query properties ────────────────────────────────────────────────────

    @property
    def state(self) -> PickupState:
        return self._state

    @property
    def approach_yaw_deg(self) -> float:
        return self._cfg.init_yaw_deg

    @property
    def translation_allowed(self) -> bool:
        return self._state in _TRANSLATION_ALLOWED

    @property
    def position_override(self) -> Optional[np.ndarray]:
        if self._state == PickupState.ARC_ALIGN and self._arc_initialized:
            return self._current_pos
        if self._state == PickupState.LIFT_EE and self._lift_interp is not None:
            return self._current_pos
        return None

    @property
    def rotation_override(self) -> Optional[Rotation]:
        if self._state == PickupState.APPROACH:
            return self._current_rot
        if self._state == PickupState.ARC_ALIGN and self._arc_initialized:
            return self._current_rot
        if self._state == PickupState.LIFT_EE and self._lift_rot is not None:
            return self._lift_rot
        return None

    @property
    def stage_label(self) -> str:
        total = len(_ACTIVE_STAGES)
        try:
            idx = _ACTIVE_STAGES.index(self._state) + 1
        except ValueError:
            idx = total if self._state in (PickupState.CONFIRMED, PickupState.ABORTED) else 0
        return f'{self._state.name} [{idx}/{total}]'

    @property
    def stage_hint(self) -> str:
        hints = {
            PickupState.IDLE:      'Press LEFT to begin pickup.',
            PickupState.APPROACH:  'Translate toward fixture. LEFT when aligned.',
            PickupState.ARC_ALIGN: f'Arc along fixture (R={self._cfg.arc_radius_mm:.0f}mm) — push fwd/back (X). LEFT when ready.',
            PickupState.LIFT_EE:   f'Auto-lifting {self._cfg.lift_distance_mm:.0f}mm along EE z...',
            PickupState.CONFIRM:   'Hold steady. Press LEFT to confirm.',
            PickupState.CONFIRMED: 'Confirmed! Press LEFT to reset.',
            PickupState.ABORTED:   'Aborted. Press LEFT to reset.',
        }
        return hints.get(self._state, '')

    @property
    def can_confirm(self) -> bool:
        if self._state != PickupState.CONFIRM:
            return False
        return self._stability_timer >= self._cfg.stability_hold_s

    # ── Action methods ──────────────────────────────────────────────────────

    def start(self) -> None:
        """IDLE → APPROACH."""
        if self._state != PickupState.IDLE:
            return
        self._state = PickupState.APPROACH
        self._approach_pitch_deg = 0.0
        self._current_rot = Rotation.from_euler('z', self._cfg.init_yaw_deg, degrees=True)
        self._stability_timer = 0.0
        self._recent_positions.clear()

    def advance(self) -> None:
        """LEFT-button advance for advanceable states."""
        if self._state == PickupState.APPROACH:
            self._state = PickupState.ARC_ALIGN
            self._arc_initialized = False
            self._arc_angle = 0.0
        elif self._state == PickupState.ARC_ALIGN:
            # Auto-lift starts on next tick; orientation frozen from arc end
            self._state = PickupState.LIFT_EE
            self._lift_interp = None

    def confirm(self) -> None:
        if self.can_confirm:
            self._state = PickupState.CONFIRMED

    def reset(self) -> None:
        if self._state in (PickupState.CONFIRMED, PickupState.ABORTED):
            self._state = PickupState.IDLE
            self._arc_initialized = False
            self._arc_angle = 0.0
            self._lift_interp = None
            self._lift_rot = None
            self._stability_timer = 0.0
            self._recent_positions.clear()

    def emergency_reset(self) -> None:
        """Immediately reset to IDLE from any state."""
        self._state = PickupState.IDLE
        self._arc_initialized = False
        self._arc_angle = 0.0
        self._lift_interp = None
        self._lift_rot = None
        self._stability_timer = 0.0
        self._recent_positions.clear()
        self._current_pos = None
        self._current_rot = None

    # ── Tick ────────────────────────────────────────────────────────────────

    def tick(self, ee_pos: np.ndarray, ee_rot: Rotation,
             sm_active: bool, sm_lin_delta: np.ndarray,
             sm_yaw_delta: float, dt: float) -> None:
        if self._state == PickupState.APPROACH:
            self._tick_approach(dt)
        elif self._state == PickupState.ARC_ALIGN:
            self._tick_arc_align(ee_pos, ee_rot, sm_lin_delta)
        elif self._state == PickupState.LIFT_EE:
            self._tick_lift_ee(ee_pos, ee_rot, dt)
        elif self._state == PickupState.CONFIRM:
            self._recent_positions.append(ee_pos.copy())
            if (StageCheckers.check_stability(
                    self._recent_positions, self._cfg.stability_threshold_m)
                    and not sm_active):
                self._stability_timer += dt
            else:
                self._stability_timer = 0.0

    # ── Internal helpers ────────────────────────────────────────────────────

    def _tick_approach(self, dt: float) -> None:
        """Ramp pitch toward init_pitch_deg; yaw stays fixed at init_yaw_deg."""
        target = self._cfg.init_pitch_deg
        speed = self._cfg.approach_orient_speed_deg_s
        if self._approach_pitch_deg < target:
            self._approach_pitch_deg = min(
                self._approach_pitch_deg + speed * dt, target)

        R_yaw = Rotation.from_euler('z', self._cfg.init_yaw_deg, degrees=True)
        R_pitch = Rotation.from_euler('y', -self._approach_pitch_deg, degrees=True)
        self._current_rot = R_yaw * R_pitch

    def _tick_arc_align(self, ee_pos: np.ndarray, ee_rot: Rotation,
                        sm_lin_delta: np.ndarray) -> None:
        """Manual circumferential arc — SpaceMouse X-axis drives arc angle.

        On first tick: initializes pivot and axis in world frame from EE frame
        config (arc_center_dir_ee, arc_axis_ee) so any approach yaw is handled.

        Arc control: sm_lin_delta[0] (world X = SpaceMouse push forward/back)
        is used directly as arc travel in meters.  Positive = forward = increasing
        arc angle.
        """
        if not self._arc_initialized:
            cfg = self._cfg
            radius = cfg.arc_radius_mm / 1000.0
            center_world = ee_rot.apply(cfg.arc_center_dir_ee)
            self._arc_axis_world = ee_rot.apply(cfg.arc_axis_ee)
            self._arc_pivot      = ee_pos + center_world * radius
            self._arc_offset_0   = ee_pos - self._arc_pivot
            self._arc_start_rot  = ee_rot
            self._arc_angle      = 0.0
            self._current_pos    = ee_pos.copy()
            self._current_rot    = ee_rot
            self._arc_initialized = True
            return

        radius = self._cfg.arc_radius_mm / 1000.0

        # SpaceMouse X (world X = push forward) directly drives arc travel
        d_arc = float(sm_lin_delta[0])
        self._arc_angle += d_arc / radius

        arc_rot = Rotation.from_rotvec(self._arc_axis_world * self._arc_angle)
        self._current_pos = self._arc_pivot + arc_rot.apply(self._arc_offset_0)
        self._current_rot = arc_rot * self._arc_start_rot

    def _tick_lift_ee(self, ee_pos: np.ndarray, ee_rot: Rotation,
                      dt: float) -> None:
        """Auto-lift along EE z-axis; auto-advances to CONFIRM when done."""
        if self._lift_interp is None:
            # First tick: compute target along current EE z-axis
            ee_z_world = ee_rot.apply(np.array([0.0, 0.0, 1.0]))
            target = ee_pos + ee_z_world * (self._cfg.lift_distance_mm / 1000.0)
            self._lift_interp = LinearInterpolator(ee_pos, target, self._cfg.lift_speed_m_s)
            self._lift_rot = ee_rot          # freeze orientation during lift
            self._current_pos = ee_pos.copy()
            return

        pos, done = self._lift_interp.tick(dt)
        self._current_pos = pos
        if done:
            self._lift_interp = None
            self._state = PickupState.CONFIRM
            self._stability_timer = 0.0
            self._recent_positions.clear()
