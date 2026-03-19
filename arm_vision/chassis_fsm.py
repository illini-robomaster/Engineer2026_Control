"""
Chassis FSM — autonomous STORE / FETCH waypoint sequencer.

STORE: homing → WP1 → WP2 → [pause] → [claw open] → WP3 → homing → MANUAL
FETCH: homing → WP3 → WP2 → [pause] → [claw close] → WP1 → homing → MANUAL

Usage::

    from chassis_fsm import ChassisFSM, ChassisOperation, load_chassis_config

    cfg = load_chassis_config('arm_vision/config/chassis_params.yaml')
    fsm = ChassisFSM(client, cfg)

    # In spacemouse_teleop main loop:
    fsm.start('left', ChassisOperation.STORE)
    ...
    fsm.tick(feedback, dt)
    if fsm.done:
        current_mode = TeleopMode.MANUAL
        fsm.emergency_reset()
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arm_vision.socket_client import PoseSocketClient


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class SideConfig:
    wp1: list[float]
    wp2: list[float]
    wp3: list[float]


@dataclass
class ChassisConfig:
    pause_s: float
    left: SideConfig
    right: SideConfig


def load_chassis_config(path: str) -> ChassisConfig:
    """Load chassis_params.yaml and return a ChassisConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f)['chassis']

    def load_side(d: dict) -> SideConfig:
        return SideConfig(wp1=d['wp1'], wp2=d['wp2'], wp3=d['wp3'])

    return ChassisConfig(
        pause_s=raw['pause_s'],
        left=load_side(raw['left']),
        right=load_side(raw['right']),
    )


# ── State / operation enums ───────────────────────────────────────────────────

class ChassisState(Enum):
    IDLE         = auto()   # waiting for start()
    HOMING_PRE   = auto()   # bookend homing before WP1
    STEP_1       = auto()   # moving to waypoints[0]
    STEP_2       = auto()   # moving to waypoints[1]
    PAUSING      = auto()   # fixed pause before claw op
    CLAW_OP      = auto()   # claw open/close (stub, instantaneous)
    STEP_3       = auto()   # moving to waypoints[2]
    HOMING_POST  = auto()   # bookend homing after WP3
    DONE         = auto()   # signal caller to switch to MANUAL


class ChassisOperation(Enum):
    STORE = auto()
    FETCH = auto()


# ── FSM ───────────────────────────────────────────────────────────────────────

class ChassisFSM:
    """Joint-space waypoint sequencer for STORE and FETCH operations.

    All state is immutable between ticks; tick() returns a new state by
    advancing through the sequence based on elapsed timers and feedback flags.
    """

    def __init__(self, client: 'PoseSocketClient', cfg: ChassisConfig) -> None:
        self._client   = client
        self._cfg      = cfg
        self._state    = ChassisState.IDLE
        self._operation: ChassisOperation | None = None
        self._side: str | None = None
        self._waypoints: list[list[float]] = []
        self._homing_seen:   bool  = False
        self._planning_seen: bool  = False
        self._entered:       bool  = False
        self._timer:         float = 0.0
        self.done: bool = False

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self, side: str, operation: ChassisOperation) -> None:
        """Begin a STORE or FETCH sequence for the given side ('left' or 'right')."""
        side_cfg = self._cfg.left if side == 'left' else self._cfg.right
        wps = [side_cfg.wp1, side_cfg.wp2, side_cfg.wp3]
        if operation == ChassisOperation.FETCH:
            wps = list(reversed(wps))

        self._waypoints     = wps
        self._operation     = operation
        self._side          = side
        self._homing_seen   = False
        self._planning_seen = False
        self._entered       = False
        self._timer         = 0.0
        self.done           = False
        self._state         = ChassisState.HOMING_PRE
        self._client.send_home()

    def tick(self, feedback: dict, dt: float) -> None:
        """Advance the FSM one time step.  Call every control loop iteration."""
        s = self._state

        if s == ChassisState.IDLE:
            return

        elif s == ChassisState.HOMING_PRE:
            homing = bool(feedback.get('homing', False))
            if homing:
                self._homing_seen = True
            if self._homing_seen and not homing:
                self._advance_to(ChassisState.STEP_1)

        elif s == ChassisState.STEP_1:
            if not self._entered:
                self._client.send_plan_joints(self._waypoints[0])
                self._planning_seen = False
                self._entered       = True
            planning = bool(feedback.get('planning_active', False))
            if planning:
                self._planning_seen = True
            if self._planning_seen and not planning:
                if feedback.get('planning_ok', True):
                    self._advance_to(ChassisState.STEP_2)
                else:
                    self._advance_to(ChassisState.DONE)
                    self.done = True

        elif s == ChassisState.STEP_2:
            if not self._entered:
                self._client.send_plan_joints(self._waypoints[1])
                self._planning_seen = False
                self._entered       = True
            planning = bool(feedback.get('planning_active', False))
            if planning:
                self._planning_seen = True
            if self._planning_seen and not planning:
                if feedback.get('planning_ok', True):
                    self._advance_to(ChassisState.PAUSING)
                else:
                    self._advance_to(ChassisState.DONE)
                    self.done = True

        elif s == ChassisState.PAUSING:
            if not self._entered:
                self._timer   = self._cfg.pause_s
                self._entered = True
            self._timer -= dt
            if self._timer <= 0.0:
                self._advance_to(ChassisState.CLAW_OP)

        elif s == ChassisState.CLAW_OP:
            if not self._entered:
                self._client.send_claw(open=(self._operation == ChassisOperation.STORE))
                self._entered = True
            # Claw op is instantaneous — advance immediately
            self._advance_to(ChassisState.STEP_3)

        elif s == ChassisState.STEP_3:
            if not self._entered:
                self._client.send_plan_joints(self._waypoints[2])
                self._planning_seen = False
                self._entered       = True
            planning = bool(feedback.get('planning_active', False))
            if planning:
                self._planning_seen = True
            if self._planning_seen and not planning:
                if feedback.get('planning_ok', True):
                    self._advance_to(ChassisState.HOMING_POST)
                else:
                    self._advance_to(ChassisState.DONE)
                    self.done = True

        elif s == ChassisState.HOMING_POST:
            if not self._entered:
                self._client.send_home()
                self._homing_seen = False
                self._entered     = True
            homing = bool(feedback.get('homing', False))
            if homing:
                self._homing_seen = True
            if self._homing_seen and not homing:
                self._advance_to(ChassisState.DONE)

        elif s == ChassisState.DONE:
            self.done = True

    def emergency_reset(self) -> None:
        """Immediately return to IDLE without any further commands."""
        self._state         = ChassisState.IDLE
        self._entered       = False
        self._homing_seen   = False
        self._planning_seen = False
        self.done           = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> ChassisState:
        return self._state

    @property
    def operation(self) -> ChassisOperation | None:
        return self._operation

    @property
    def side(self) -> str | None:
        return self._side

    @property
    def is_active(self) -> bool:
        return self._state not in (ChassisState.IDLE, ChassisState.DONE)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _advance_to(self, next_state: ChassisState) -> None:
        self._state   = next_state
        self._entered = False
