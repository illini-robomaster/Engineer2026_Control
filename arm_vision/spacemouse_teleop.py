#!/usr/bin/env python3
"""
SpaceMouse teleoperation client.

Reads 6-DOF SpaceMouse input, integrates position/orientation deltas, and
streams absolute EE poses to the ROS ik_teleop_node via TCP.

Install prerequisites (Mac):
    brew install hidapi
    pip install pyspacemouse

Usage:
    python spacemouse_teleop.py --host 172.16.51.47 --port 9999
    python spacemouse_teleop.py --no-socket            # dry run, no arm needed
    python spacemouse_teleop.py --no-socket --assembly config/assembly_params.yaml --pickup config/pickup_params.yaml
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from enum import Enum, auto

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ── Helpers ──────────────────────────────────────────────────────────────────

def _joints_line(feedback: dict) -> str:
    """Format joint angles (rad → deg) from feedback into a single display line."""
    if not any(f'j{i}' in feedback for i in range(1, 7)):
        return '  J  (no joint feedback yet)'
    vals = [float(np.degrees(feedback.get(f'j{i}', 0.0))) for i in range(1, 7)]
    return ('  J ' + '  '.join(f'J{i}{v:+7.1f}°' for i, v in enumerate(vals, 1)))


def _bar(value: float, lo: float, hi: float, width: int = 10) -> str:
    """Return a filled bar representing *value* within [lo, hi]."""
    span  = hi - lo if hi != lo else 1.0
    frac  = max(0.0, min(1.0, (value - lo) / span))
    filled = round(frac * width)
    return '█' * filled + '░' * (width - filled)


_VIZ_LINES        = 5   # number of lines render_pose prints
_ASM_VIZ_LINES    = 7   # number of lines render_assembly / render_pickup print
_CHASSIS_VIZ_LINES = 5  # number of lines render_chassis prints


def render_pose(
    pos: 'np.ndarray',
    sm_ang: 'np.ndarray',
    pos_min: 'np.ndarray',
    pos_max: 'np.ndarray',
    init_pos: 'np.ndarray',
    feedback: dict,
    header: str = '',
    first: bool = False,
) -> None:
    """Overwrite the last _VIZ_LINES terminal lines with a live 6-DOF display."""
    roll, pitch, yaw = sm_ang   # raw SpaceMouse values in [-1, 1]

    w = 12  # bar width

    half = (pos_max - pos_min) / 2.0
    line1 = (
        f"  X {pos[0]:+7.3f}m |{_bar(pos[0], init_pos[0] - half[0], init_pos[0] + half[0], w)}|"
        f"   Y {pos[1]:+7.3f}m |{_bar(pos[1], init_pos[1] - half[1], init_pos[1] + half[1], w)}|"
        f"   Z {pos[2]:+7.3f}m |{_bar(pos[2], init_pos[2] - half[2], init_pos[2] + half[2], w)}|"
    )
    line2 = (
        f"  R {roll:+6.3f}   |{_bar(roll,  -1, 1, w)}|"
        f"   P {pitch:+6.3f}   |{_bar(pitch, -1, 1, w)}|"
        f"   Y {yaw:+6.3f}   |{_bar(yaw,   -1, 1, w)}|"
    )
    sep = '─' * len(line1)
    if 'fk_x' in feedback:
        ik_str = 'OK' if feedback.get('ik_ok', True) else 'SNAP/FAIL'
        line4 = (f"  FK  X {feedback['fk_x']:+7.3f}m"
                 f"   Y {feedback['fk_y']:+7.3f}m"
                 f"   Z {feedback['fk_z']:+7.3f}m   IK:{ik_str}")
    else:
        line4 = "  FK  (no feedback yet)"

    if not first:
        print(f'\033[{_VIZ_LINES}A', end='')

    if header:
        print(header.ljust(len(sep)))
    else:
        print(sep)
    print(line1)
    print(line2)
    print(line4)
    print(_joints_line(feedback).ljust(len(sep)))


def render_assembly(
    fsm,
    pos: 'np.ndarray',
    difficulty: int,
    q_target: float,
    feedback: dict,
    mode_header: str = '',
    first: bool = False,
) -> None:
    """Overwrite terminal lines with assembly-mode status display."""
    from assembly_fsm import AssemblyState

    width = 60
    sep = '─' * width

    hint = fsm.stage_hint
    if fsm.state == AssemblyState.READY_CONFIRM:
        hint = '  OK — press LEFT to confirm' if fsm.can_confirm else '  Waiting for stability...'

    if 'fk_x' in feedback:
        ik_str = 'OK' if feedback.get('ik_ok', True) else 'SNAP/FAIL'
        fb_line = (f'  FK  X {feedback["fk_x"]:+7.3f}'
                   f'  Y {feedback["fk_y"]:+7.3f}'
                   f'  Z {feedback["fk_z"]:+7.3f}  IK:{ik_str}')
    else:
        fb_line = '  FK  (no feedback — is IK node running?)'

    header = mode_header if mode_header else f'  [ASSEMBLY] Difficulty {difficulty} | Q-target: {q_target}°'
    lines = [
        sep,
        header,
        f'  State: {fsm.stage_label}  Yaw: {fsm.approach_yaw_deg:+.1f}°',
        f'  Cmd: X {pos[0]:+7.3f}  Y {pos[1]:+7.3f}  Z {pos[2]:+7.3f}',
        hint,
        fb_line,
        _joints_line(feedback),
    ]

    if not first:
        print(f'\033[{_ASM_VIZ_LINES}A', end='')

    for line in lines:
        print(line.ljust(width))


def render_pickup(
    fsm,
    pos: 'np.ndarray',
    feedback: dict,
    mode_header: str = '',
    first: bool = False,
) -> None:
    """Overwrite terminal lines with pickup-mode status display."""
    from pickup_fsm import PickupState

    width = 60
    sep = '─' * width

    hint = fsm.stage_hint
    if fsm.state == PickupState.CONFIRM:
        hint = '  OK — press LEFT to confirm' if fsm.can_confirm else '  Waiting for stability...'

    if 'fk_x' in feedback:
        ik_str = 'OK' if feedback.get('ik_ok', True) else 'SNAP/FAIL'
        fb_line = (f'  FK  X {feedback["fk_x"]:+7.3f}'
                   f'  Y {feedback["fk_y"]:+7.3f}'
                   f'  Z {feedback["fk_z"]:+7.3f}  IK:{ik_str}')
    else:
        fb_line = '  FK  (no feedback — is IK node running?)'

    header = mode_header if mode_header else f'  [PICKUP] R={fsm._cfg.arc_radius_mm:.0f}mm | Yaw: {fsm.approach_yaw_deg:+.1f}°'
    lines = [
        sep,
        header,
        f'  State: {fsm.stage_label}',
        f'  Cmd: X {pos[0]:+7.3f}  Y {pos[1]:+7.3f}  Z {pos[2]:+7.3f}',
        hint,
        fb_line,
        _joints_line(feedback),
    ]

    if not first:
        print(f'\033[{_ASM_VIZ_LINES}A', end='')

    for line in lines:
        print(line.ljust(width))


def render_chassis(
    fsm,
    feedback: dict,
    mode_header: str = '',
    first: bool = False,
) -> None:
    """Overwrite terminal lines with STORE/FETCH mode status display."""
    from chassis_fsm import ChassisState, ChassisOperation

    width = 60
    sep = '─' * width

    step_names = {
        ChassisState.IDLE:        'IDLE — press L/R to start',
        ChassisState.HOMING_PRE:  'HOMING (pre) …',
        ChassisState.STEP_1:      'MOVING → WP1',
        ChassisState.STEP_2:      'MOVING → WP2',
        ChassisState.PAUSING:     'PAUSING …',
        ChassisState.CLAW_OP:     'CLAW OP',
        ChassisState.STEP_3:      'MOVING → WP3',
        ChassisState.HOMING_POST: 'HOMING (post) …',
        ChassisState.DONE:        'DONE',
    }

    op_str   = fsm.operation.name if fsm.operation else '—'
    side_str = fsm.side.upper() if fsm.side else '—'
    state_str = step_names.get(fsm.state, str(fsm.state))

    if 'fk_x' in feedback:
        ik_str = 'OK' if feedback.get('ik_ok', True) else 'SNAP/FAIL'
        fb_line = (f'  FK  X {feedback["fk_x"]:+7.3f}'
                   f'  Y {feedback["fk_y"]:+7.3f}'
                   f'  Z {feedback["fk_z"]:+7.3f}  IK:{ik_str}')
    else:
        fb_line = '  FK  (no feedback — is IK node running?)'

    header = mode_header if mode_header else f'  [{op_str}] Side: {side_str}'
    lines = [
        sep,
        header,
        f'  State: {state_str}',
        fb_line,
        _joints_line(feedback),
    ]

    if not first:
        print(f'\033[{_CHASSIS_VIZ_LINES}A', end='')

    for line in lines:
        print(line.ljust(width))


# ── P-arc recording analysis ─────────────────────────────────────────────────

def analyze_p_arc(
    start_pos: 'np.ndarray',
    start_rot: 'Rotation',
    end_pos: 'np.ndarray',
    end_rot: 'Rotation',
    handle_offset_ee_mm: 'np.ndarray | None' = None,
) -> None:
    """Derive P-arc parameters from the initial and final EE 6D poses.

    Because the recording constrains motion to EE x-z translation and EE y
    rotation, the two poses define a pure rotation about a fixed world axis.
    The pivot is solved analytically from the rigid-body constraint plus the
    axis-plane constraint (pivot lies in the plane ⊥ axis through start_pos).
    """
    if handle_offset_ee_mm is None:
        handle_offset_ee_mm = np.array([-108.0, 0.0, 0.0])

    arc_rot   = end_rot * start_rot.inv()
    rv        = arc_rot.as_rotvec()
    angle_rad = float(np.linalg.norm(rv))
    if angle_rad < 1e-4:
        print('[P-ARC] No rotation detected — move more during recording.')
        return

    angle_deg = float(np.degrees(angle_rad))

    # Rotation axis in world frame — should equal start EE y (constrained during recording)
    axis_world = rv / angle_rad

    # Pivot: solve (I − R) @ pivot = end_pos − R @ start_pos
    # The null space of (I − R) is along axis, so add one axis-plane constraint:
    #   axis · pivot = axis · start_pos  (pivot lies in plane ⊥ axis through start)
    R   = arc_rot.as_matrix()
    A   = np.vstack([np.eye(3) - R, axis_world[np.newaxis, :]])
    b   = np.append(end_pos - R @ start_pos, float(np.dot(axis_world, start_pos)))
    pivot, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    center_world  = pivot - start_pos
    ee_radius_m   = float(np.linalg.norm(center_world))
    if ee_radius_m < 1e-4:
        print('[P-ARC] Pivot too close to EE start — check recording.')
        return

    center_dir_ee = start_rot.inv().apply(center_world / ee_radius_m)

    # Handle arc radius (constant throughout arc by construction)
    h_off        = handle_offset_ee_mm / 1000.0
    handle_start = start_pos + start_rot.apply(h_off)
    r_handle_mm  = float(np.linalg.norm(handle_start - pivot) * 1000.0)

    # Verification: re-predict end_pos from derived pivot + arc_rot
    predicted_end = pivot + arc_rot.apply(start_pos - pivot)
    pos_err_mm    = float(np.linalg.norm(predicted_end - end_pos) * 1000.0)

    sep = '=' * 62
    print(f'\n{sep}')
    print('[P-ARC RECORD] Analysis (start + end pose)')
    print(f'  Arc angle:             {angle_deg:.1f}°')
    print(f'  Axis (world):          [{axis_world[0]:+.4f}, {axis_world[1]:+.4f}, {axis_world[2]:+.4f}]')
    print(f'  Pivot (world):         [{pivot[0]:+.4f}, {pivot[1]:+.4f}, {pivot[2]:+.4f}]')
    print(f'  Handle arc radius:     {r_handle_mm:.1f} mm')
    print(f'  EE arc radius:         {ee_radius_m * 1000:.1f} mm')
    print(f'  center_dir_ee:         [{center_dir_ee[0]:.4f}, {center_dir_ee[1]:.4f}, {center_dir_ee[2]:.4f}]')
    print(f'  Verification error:    {pos_err_mm:.2f} mm')
    print()
    print('  ── Paste into assembly_params.yaml ──────────────────────')
    print(f'  p_arc_radius_mm:       {ee_radius_m * 1000:.1f}')
    print(f'  p_arc_angle_deg:       {angle_deg:.1f}')
    print(f'  p_arc_center_dir_ee:   [{center_dir_ee[0]:.4f}, {center_dir_ee[1]:.4f}, {center_dir_ee[2]:.4f}]')
    print(sep)


# ── Mode + Button combo ───────────────────────────────────────────────────────

class TeleopMode(Enum):
    MANUAL   = auto()
    ASSEMBLY = auto()
    PICKUP   = auto()
    STORE    = auto()
    FETCH    = auto()


class ButtonComboDetector:
    """Detect simultaneous LEFT+RIGHT press (within COMBO_WINDOW seconds).

    Call update() every tick.  Returns one of: 'NONE', 'LEFT', 'RIGHT', 'COMBO'.

    A COMBO fires when both edges arrive within COMBO_WINDOW of each other.
    Single-button events are delayed by up to COMBO_WINDOW to check for a combo
    partner — imperceptible in practice for 150ms.
    """
    COMBO_WINDOW = 0.15  # seconds

    def __init__(self) -> None:
        self._left_pending: bool = False
        self._right_pending: bool = False
        self._left_t: float = 0.0
        self._right_t: float = 0.0

    def update(self, left_edge: bool, right_edge: bool, now: float) -> str:
        if left_edge:
            self._left_pending = True
            self._left_t = now
        if right_edge:
            self._right_pending = True
            self._right_t = now

        # If both pending, check if within combo window → COMBO
        if self._left_pending and self._right_pending:
            if abs(self._left_t - self._right_t) <= self.COMBO_WINDOW:
                self._left_pending = False
                self._right_pending = False
                return 'COMBO'

        # Expire single-button pending events
        result = 'NONE'
        if self._left_pending and (now - self._left_t) > self.COMBO_WINDOW:
            self._left_pending = False
            result = 'LEFT'
        if self._right_pending and (now - self._right_t) > self.COMBO_WINDOW:
            self._right_pending = False
            # If LEFT also fired this tick, promote to COMBO — but that's caught above.
            # Here it's a lone RIGHT.
            result = 'RIGHT' if result == 'NONE' else result  # shouldn't both fire alone

        return result


# ── CLI + config ─────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='spacemouse_teleop',
        description='Stream SpaceMouse EE poses to ROS ik_teleop_node via TCP.',
    )
    p.add_argument('--host',      default='172.16.51.47',
                   help='ROS host IP (default: 172.16.51.47)')
    p.add_argument('--port',      type=int, default=9999,
                   help='ROS socket port (default: 9999)')
    p.add_argument('--config',    default='config/spacemouse_params.yaml',
                   help='Path to spacemouse_params.yaml')
    p.add_argument('--speed',     type=float, default=1.0,
                   help='Global speed multiplier (default: 1.0)')
    p.add_argument('--no-socket', action='store_true', dest='no_socket',
                   help='Dry-run: print poses instead of sending over TCP')
    p.add_argument('--assembly', default=None,
                   help='Path to assembly_params.yaml (enables assembly mode)')
    p.add_argument('--pickup',   default=None,
                   help='Path to pickup_params.yaml (enables pickup mode)')
    p.add_argument('--chassis',  default=None,
                   help='Path to chassis_params.yaml (enables STORE/FETCH modes)')
    p.add_argument('--difficulty', type=int, default=None,
                   help='Override difficulty level (1-4)')
    p.add_argument('--q-angle', type=float, default=None,
                   help='Override Q-axis target angle (degrees)')
    p.add_argument('--yaw', type=float, default=None,
                   help='Override approach yaw angle (degrees, 0=+X)')
    p.add_argument('--record-p-arc', action='store_true', dest='record_p_arc',
                   help='Manually draw the P arc with SpaceMouse to derive arc params')
    p.add_argument('--handle-offset', type=float, nargs=3,
                   default=[-108.0, 0.0, 0.0], metavar=('X', 'Y', 'Z'),
                   help='Handle offset in EE frame mm (default: -108 0 0)')
    p.add_argument('--log-dir', default='.', metavar='DIR',
                   help='Directory for P-arc session .log files (default: current dir, '
                        'pass empty string to disable)')
    return p


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    cfg = raw['spacemouse']
    cfg['lin_map']   = np.array(cfg['lin_map'],   dtype=float).reshape(3, 3)
    cfg['ang_map']   = np.array(cfg['ang_map'],   dtype=float).reshape(3, 3)
    cfg['pos_min']   = np.array(cfg['pos_min'],   dtype=float)
    cfg['pos_max']   = np.array(cfg['pos_max'],   dtype=float)
    cfg['init_pos']  = np.array(cfg['init_pos'],  dtype=float)
    cfg['init_quat'] = np.array(cfg['init_quat'], dtype=float)
    return cfg


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    cfg  = load_config(args.config)

    try:
        import pyspacemouse
    except ImportError:
        sys.exit('[ERROR] pyspacemouse not installed.\n'
                 '  brew install hidapi && pip install pyspacemouse')

    try:
        device = pyspacemouse.open()
    except Exception as exc:
        sys.exit(f'[ERROR] Could not open SpaceMouse: {exc}\n'
                 'Is it plugged in? Check USB/HID permissions.')

    if args.no_socket:
        print('[spacemouse] --no-socket: printing poses only, no TCP connection.')
        client = None
    else:
        from arm_vision.socket_client import PoseSocketClient
        print(f'[spacemouse] Connecting to {args.host}:{args.port}...')
        client = PoseSocketClient(host=args.host, port=args.port)
        client.start()

    # ── Load optional mode configs ─────────────────────────────────────────
    asm_fsm = None
    asm_cfg = None
    if args.assembly:
        from assembly_fsm import AssemblyFSM, load_assembly_config
        asm_cfg = load_assembly_config(args.assembly,
                                       difficulty_override=args.difficulty,
                                       q_angle_override=args.q_angle,
                                       yaw_override=args.yaw)
        asm_fsm = AssemblyFSM(asm_cfg)
        print(f'[assembly] Difficulty {asm_cfg.difficulty}, Q-angle={asm_cfg.q_target_deg}°')

    pickup_fsm = None
    pickup_cfg = None
    if args.pickup:
        from pickup_fsm import PickupFSM, load_pickup_config
        pickup_cfg = load_pickup_config(args.pickup)
        pickup_fsm = PickupFSM(pickup_cfg)
        print(f'[pickup] R={pickup_cfg.arc_radius_mm}mm, yaw={pickup_cfg.init_yaw_deg}°')

    chassis_fsm = None
    if args.chassis:
        from chassis_fsm import ChassisFSM, ChassisOperation, load_chassis_config
        _chassis_cfg = load_chassis_config(args.chassis)
        chassis_fsm  = ChassisFSM(client, _chassis_cfg)
        print(f'[chassis] Loaded config from {args.chassis}')

    # ── Build available mode cycle ─────────────────────────────────────────
    available_modes = [TeleopMode.MANUAL]
    if asm_fsm is not None:
        available_modes.append(TeleopMode.ASSEMBLY)
    if pickup_fsm is not None:
        available_modes.append(TeleopMode.PICKUP)
    if chassis_fsm is not None:
        available_modes.append(TeleopMode.STORE)
        available_modes.append(TeleopMode.FETCH)

    current_mode = TeleopMode.MANUAL
    mode_idx = 0

    def cycle_mode():
        nonlocal current_mode, mode_idx
        mode_idx = (mode_idx + 1) % len(available_modes)
        current_mode = available_modes[mode_idx]

        # Emergency-reset all FSMs
        if asm_fsm is not None:
            asm_fsm.emergency_reset()
        if pickup_fsm is not None:
            pickup_fsm.emergency_reset()
        if chassis_fsm is not None:
            chassis_fsm.emergency_reset()

        # Resync EE pose
        if client is not None:
            fb = client.feedback
            if 'fk_x' in fb:
                new_pos = np.array([fb['fk_x'], fb['fk_y'], fb['fk_z']])
                new_pos = np.clip(new_pos, cfg['pos_min'], cfg['pos_max'])
            else:
                new_pos = cfg['init_pos'].copy()
        else:
            new_pos = cfg['init_pos'].copy()

        return new_pos, Rotation.from_quat(cfg['init_quat'])

    # ── State ──────────────────────────────────────────────────────────────
    ee_pos = cfg['init_pos'].copy()
    ee_rot = Rotation.from_quat(cfg['init_quat'])

    dt          = 1.0 / cfg['poll_hz']
    print_every = int(cfg['poll_hz'])
    tick        = 0
    prev_buttons = [0, 0]
    _arm_ready           = args.no_socket
    _homing_active       = False
    _homing_seen_running = False
    _homing_t            = 0.0
    _ik_was_failing      = False
    # Homing requires RIGHT held for this long to prevent accidental triggers
    HOMING_HOLD_S        = 2.0
    ASSEMBLY_EXIT_HOLD_S = 1.5   # shorter hold exits assembly FSM → MANUAL
    _right_hold_t         = None   # monotonic time when RIGHT was first pressed
    _right_hold_fired     = False  # True once the hold-trigger fires (until release)
    _right_asm_exit_fired = False  # True once the 1.5s assembly-exit fires (until release)

    combo_detector = ButtonComboDetector()

    # ── Approach discrete-step rotation ───────────────────────────────────
    # Twist and Y-linear axes fire a single 5° step when pushed past the
    # threshold.  The axis must return to neutral before firing again.
    APPROACH_STEP_DEG       = 5.0
    APPROACH_STEP_THRESHOLD = 0.8   # fraction of SM max (0–1)
    _approach_roll_state    = 0     # last fired direction: -1, 0, or +1
    _approach_pitch_state   = 0
    _approach_yaw_state     = 0

    # ── P-arc recording state ──────────────────────────────────────────────
    _p_arc_recording  = False
    _p_arc_start_pos  = None
    _p_arc_start_rot  = None
    _p_arc_prev_state = None
    _handle_offset_ee = np.array(args.handle_offset, dtype=float)

    # ── P-arc session log ──────────────────────────────────────────────────
    _parc_log_file: object = None   # open file handle, or None when not logging
    _parc_log_path: str    = ''
    _parc_log_t0:  float   = 0.0

    mode_names = ' / '.join(m.name for m in available_modes)
    print(f'[spacemouse] Modes available: {mode_names}')
    print(f'[spacemouse] BOTH buttons to cycle modes | current: {current_mode.name}')
    print(f'[spacemouse] Ready — press RIGHT to home and start streaming.')
    print(f'[spacemouse] init_pos={ee_pos}  speed={args.speed}')

    try:
        while True:
            t_start = time.monotonic()
            now     = t_start

            state = device.read()
            if state is None:
                time.sleep(dt)
                continue

            # ── Raw inputs ────────────────────────────────────────────────
            sm_lin = np.array([state.x, state.y, state.z], dtype=float)
            sm_lin = np.where(np.abs(sm_lin) > cfg['deadband'], sm_lin, 0.0)

            sm_ang = np.array([state.roll, state.pitch, state.yaw], dtype=float)
            sm_ang = np.where(np.abs(sm_ang) > cfg['deadband'], sm_ang, 0.0)

            # ── Button edge detection ─────────────────────────────────────
            buttons = list(state.buttons) if state.buttons else [0, 0]
            left_edge  = bool(buttons[0] and not prev_buttons[0])
            right_held = bool(len(buttons) > 1 and buttons[1])
            right_edge = bool(right_held and not prev_buttons[1])
            prev_buttons = buttons[:2] if len(buttons) >= 2 else buttons + [0]

            # ── Combo detection ───────────────────────────────────────────
            event = combo_detector.update(left_edge, right_edge, now)

            # Always compute delta_p — used for arc control even when gated
            delta_p = (cfg['lin_map'] @ sm_lin) * cfg['linear_speed'] * args.speed * dt
            delta_r = (cfg['ang_map'] @ sm_ang) * cfg['angular_speed'] * args.speed * dt

            # ── COMBO: cycle mode ─────────────────────────────────────────
            if event == 'COMBO':
                ee_pos, ee_rot = cycle_mode()
                _right_hold_t    = None   # cancel any pending homing hold
                _right_hold_fired = False
                print(f'\n[spacemouse] Switched to mode: {current_mode.name}')
                # Consume event — no further button processing this tick
                event = 'NONE'

            # ── Right-button hold tracking ────────────────────────────────
            # Homing requires RIGHT held for HOMING_HOLD_S seconds to prevent
            # accidental triggers during normal spacemouse operation.
            if right_edge:
                _right_hold_t         = now
                _right_hold_fired     = False
                _right_asm_exit_fired = False
            elif not right_held:
                _right_hold_t         = None
                _right_hold_fired     = False
                _right_asm_exit_fired = False

            _right_homing_trigger = (
                _right_hold_t is not None
                and not _right_hold_fired
                and (now - _right_hold_t) >= HOMING_HOLD_S
            )
            if _right_homing_trigger:
                _right_hold_fired = True  # fire exactly once per hold

            # Assembly exit: 1.5 s hold → emergency-reset FSM and return to MANUAL.
            # Also marks _right_hold_fired so the 2 s homing doesn't chain.
            _right_asm_exit = (
                current_mode == TeleopMode.ASSEMBLY
                and _right_hold_t is not None
                and not _right_asm_exit_fired
                and (now - _right_hold_t) >= ASSEMBLY_EXIT_HOLD_S
            )
            if _right_asm_exit:
                _right_asm_exit_fired = True
                _right_hold_fired     = True   # suppress the 2 s homing

            # ── Homing (RIGHT held 2 s in MANUAL or PICKUP) ───────────────
            _right_for_homing = _right_homing_trigger and current_mode == TeleopMode.MANUAL
            _right_for_pickup = _right_homing_trigger and current_mode == TeleopMode.PICKUP

            # ── Mode dispatch ─────────────────────────────────────────────

            if current_mode == TeleopMode.ASSEMBLY and asm_fsm is not None:
                from assembly_fsm import AssemblyState

                if _right_asm_exit:
                    asm_fsm.emergency_reset()
                    _p_arc_recording = False
                    current_mode = TeleopMode.MANUAL
                    mode_idx     = available_modes.index(TeleopMode.MANUAL)
                    print('\n[spacemouse] Assembly exited — back to MANUAL')

                # ── P-arc recording: detect entry into AUTO_ROTATE_P ──────
                if (args.record_p_arc
                        and asm_fsm.state == AssemblyState.AUTO_ROTATE_P
                        and _p_arc_prev_state != AssemblyState.AUTO_ROTATE_P
                        and not _p_arc_recording):
                    _p_arc_recording  = True
                    _p_arc_log        = []
                    _p_arc_start_pos  = ee_pos.copy()
                    _p_arc_start_rot  = ee_rot
                    print('\n[P-ARC RECORD] Draw the arc (EE x-z plane, EE y rotation only).')
                    print('[P-ARC RECORD] Press LEFT at the end pose to derive params.')
                    if args.log_dir:
                        os.makedirs(args.log_dir, exist_ok=True)
                        _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        _parc_log_path = os.path.join(args.log_dir, f'parc_{_ts}.log')
                        _parc_log_file = open(_parc_log_path, 'w', buffering=1)
                        _parc_log_t0   = time.monotonic()
                        _parc_log_file.write(f'# P-arc session: {datetime.datetime.now().isoformat()}\n')
                        _parc_log_file.write(f'# assembly: {args.assembly}\n')
                        _parc_log_file.write(
                            f'# init_pos: {ee_pos[0]:.4f} {ee_pos[1]:.4f} {ee_pos[2]:.4f}\n')
                        _parc_log_file.write(
                            '# Columns: t_s'
                            ' | sent_x sent_y sent_z'
                            ' | qw qx qy qz'
                            ' | fk_x fk_y fk_z'
                            ' | ik_ok'
                            ' | j1 j2 j3 j4 j5 j6\n')
                        print(f'[P-ARC LOG] Writing to {_parc_log_path}')
                _p_arc_prev_state = asm_fsm.state

                # ── Button handling ───────────────────────────────────────
                if event == 'LEFT':
                    if _p_arc_recording and asm_fsm.state == AssemblyState.AUTO_ROTATE_P:
                        _p_arc_recording = False
                        if _parc_log_file is not None:
                            _parc_log_file.write(f'# session ended at t={time.monotonic()-_parc_log_t0:.3f}s\n')
                            _parc_log_file.close()
                            _parc_log_file = None
                            print(f'[P-ARC LOG] Saved: {_parc_log_path}')
                        analyze_p_arc(_p_arc_start_pos, _p_arc_start_rot,
                                      ee_pos, ee_rot, _handle_offset_ee)
                        asm_fsm.advance()   # AUTO_ROTATE_P is now in _ADVANCEABLE
                    elif asm_fsm.state == AssemblyState.IDLE:
                        asm_fsm.start()
                    elif asm_fsm.state == AssemblyState.READY_CONFIRM:
                        asm_fsm.confirm()
                    elif asm_fsm.state in (AssemblyState.CONFIRMED, AssemblyState.ABORTED):
                        asm_fsm.reset()
                    else:
                        asm_fsm.advance()

                spd = asm_fsm.approach_speed_scale

                if _p_arc_recording and asm_fsm.state == AssemblyState.AUTO_ROTATE_P:
                    # Constrained SpaceMouse control for P-arc recording:
                    #   Translation  → EE x-z plane only (no EE-y drift)
                    #   Rotation     → EE y-axis only (pitch-like arc rotation)
                    R_mat  = ee_rot.as_matrix()   # columns: ee_x, ee_y, ee_z in world
                    ee_x   = R_mat[:, 0]
                    ee_y   = R_mat[:, 1]
                    ee_z   = R_mat[:, 2]
                    dp = delta_p * spd
                    dr = delta_r * spd
                    # Project translation onto EE x and EE z (drop EE y component)
                    dp_constrained = np.dot(dp, ee_x) * ee_x + np.dot(dp, ee_z) * ee_z
                    # Project rotation onto EE y only
                    dr_constrained = np.dot(dr, ee_y) * ee_y
                    ee_pos = np.clip(ee_pos + dp_constrained, cfg['pos_min'], cfg['pos_max'])
                    ee_rot = Rotation.from_rotvec(dr_constrained) * ee_rot
                    # Do NOT tick the FSM — keeps the arc from auto-completing
                    if _parc_log_file is not None:
                        _fb  = client.feedback if client is not None else {}
                        _q   = ee_rot.as_quat()   # [x, y, z, w]
                        _j   = [_fb.get(f'j{_i}', 0.0) for _i in range(1, 7)]
                        _parc_log_file.write(
                            f'{time.monotonic() - _parc_log_t0:.3f}  '
                            f'{ee_pos[0]:.4f} {ee_pos[1]:.4f} {ee_pos[2]:.4f}  '
                            f'{_q[3]:.4f} {_q[0]:.4f} {_q[1]:.4f} {_q[2]:.4f}  '
                            f'{_fb.get("fk_x", 0.0):.4f} {_fb.get("fk_y", 0.0):.4f} {_fb.get("fk_z", 0.0):.4f}  '
                            f'{int(_fb.get("ik_ok", 1))}  '
                            f'{_j[0]:.4f} {_j[1]:.4f} {_j[2]:.4f} {_j[3]:.4f} {_j[4]:.4f} {_j[5]:.4f}\n'
                        )
                else:
                    rot_override = asm_fsm.rotation_override

                    if asm_fsm.translation_allowed:
                        if asm_fsm.state == AssemblyState.APPROACH and rot_override is not None:
                            # EE-frame navigation: project SpaceMouse axes onto the
                            # current EE frame so that:
                            #   SM +X (push forward) → EE −Z (approach axis)
                            #   SM +Y (push right)   → EE +Y (lateral)
                            #   SM +Z (push down)    → EE +X (vertical)
                            R = rot_override.as_matrix()
                            dp = (delta_p[0] * (-R[:, 2])   # SM X → EE −Z
                                + delta_p[1] * R[:, 1]       # SM Y → EE +Y
                                + delta_p[2] * R[:, 0])      # SM Z → EE +X
                            ee_pos = np.clip(ee_pos + dp * spd, cfg['pos_min'], cfg['pos_max'])
                        else:
                            ee_pos = np.clip(ee_pos + delta_p * spd, cfg['pos_min'], cfg['pos_max'])

                    if rot_override is not None:
                        ee_rot = rot_override

                    sm_active = float(np.linalg.norm(sm_lin)) > cfg['deadband']

                    if asm_fsm.state == AssemblyState.APPROACH:
                        # Discrete-step rotation for APPROACH (EE frame, right-multiply):
                        #   SM twist     (ang[2])  → EE roll  — 5° per edge trigger
                        #   SM pitch     (ang[1])  → EE pitch — 5° per edge trigger
                        #   SM roll tilt (ang[0])  → EE yaw   — 5° per edge trigger
                        # Note: lin[1] (forward push) was previously used for yaw but
                        # conflicts with approach translation and rarely reaches the
                        # 0.8 threshold during normal use — replaced with ang[0] roll tilt.
                        #
                        # Edge rule: fires once when the axis crosses APPROACH_STEP_THRESHOLD.
                        # Re-arms only when the axis returns to 0.0 (below deadband) —
                        # NOT when it merely dips below the threshold.  Without this
                        # hysteresis the state resets whenever the analog value briefly
                        # oscillates around 0.8, causing continuous unintended stepping.
                        _step_rad = np.radians(APPROACH_STEP_DEG)

                        _twist = sm_ang[2]   # raw [-1, 1] after deadband
                        if abs(_twist) >= APPROACH_STEP_THRESHOLD:
                            _d = 1 if _twist > 0 else -1
                            if _approach_roll_state != _d:
                                _approach_roll_state = _d
                                _roll_step = -_d * _step_rad  # inverted: CW twist → positive EE roll
                            else:
                                _roll_step = 0.0
                        else:
                            if _twist == 0.0:   # re-arm only at true neutral (below deadband)
                                _approach_roll_state = 0
                            _roll_step = 0.0

                        _pitch = sm_ang[1]   # raw [-1, 1] after deadband
                        if abs(_pitch) >= APPROACH_STEP_THRESHOLD:
                            _d = 1 if _pitch > 0 else -1
                            if _approach_pitch_state != _d:
                                _approach_pitch_state = _d
                                _pitch_step = _d * _step_rad
                            else:
                                _pitch_step = 0.0
                        else:
                            if _pitch == 0.0:   # re-arm only at true neutral (below deadband)
                                _approach_pitch_state = 0
                            _pitch_step = 0.0

                        _roll_tilt = sm_ang[0]   # state.roll — tilt left/right, raw [-1, 1] after deadband
                        if abs(_roll_tilt) >= APPROACH_STEP_THRESHOLD:
                            _d = 1 if _roll_tilt > 0 else -1
                            if _approach_yaw_state != _d:
                                _approach_yaw_state = _d
                                _yaw_step = _d * _step_rad
                            else:
                                _yaw_step = 0.0
                        else:
                            if _roll_tilt == 0.0:   # re-arm only at true neutral (below deadband)
                                _approach_yaw_state = 0
                            _yaw_step = 0.0

                        _sm_rot_approach = np.array([
                            _roll_step,    # twist      → EE roll  (discrete 5°)
                            _pitch_step,   # pitch      → EE pitch (discrete 5°)
                            _yaw_step,     # roll tilt  → EE yaw   (discrete 5°)
                        ])
                        asm_fsm.tick(ee_pos, ee_rot, sm_active=sm_active,
                                     sm_lin_delta=delta_p * spd,
                                     sm_rot_delta=_sm_rot_approach, dt=dt)
                    else:
                        asm_fsm.tick(ee_pos, ee_rot, sm_active=sm_active,
                                     sm_lin_delta=delta_p * spd, sm_rot_delta=delta_r * spd, dt=dt)

                    pos_override = asm_fsm.position_override
                    if pos_override is not None:
                        ee_pos = pos_override

            elif current_mode == TeleopMode.PICKUP and pickup_fsm is not None:
                from pickup_fsm import PickupState

                if _right_for_pickup and not _homing_active:
                    pickup_fsm.emergency_reset()
                    ee_rot = Rotation.from_quat(cfg['init_quat'])
                    if client is not None:
                        client.send_home()
                        _homing_active       = True
                        _homing_seen_running = False
                        _homing_t            = time.monotonic()
                        print('\n[spacemouse] Homing command sent — streaming paused')
                    else:
                        print('\n[spacemouse] No socket — skipping homing')

                if event == 'LEFT':
                    if pickup_fsm.state == PickupState.IDLE:
                        pickup_fsm.start()
                    elif pickup_fsm.state == PickupState.CONFIRM:
                        pickup_fsm.confirm()
                    elif pickup_fsm.state in (PickupState.CONFIRMED, PickupState.ABORTED):
                        pickup_fsm.reset()
                    else:
                        pickup_fsm.advance()

                if pickup_fsm.translation_allowed:
                    ee_pos = np.clip(ee_pos + delta_p, cfg['pos_min'], cfg['pos_max'])

                rot_override = pickup_fsm.rotation_override
                if rot_override is not None:
                    ee_rot = rot_override

                sm_active = float(np.linalg.norm(sm_lin)) > cfg['deadband']
                pickup_fsm.tick(ee_pos, ee_rot, sm_active=sm_active,
                                sm_lin_delta=delta_p, sm_yaw_delta=delta_r[2], dt=dt)

                pos_override = pickup_fsm.position_override
                if pos_override is not None:
                    ee_pos = pos_override

            elif current_mode in (TeleopMode.STORE, TeleopMode.FETCH) and chassis_fsm is not None:
                from chassis_fsm import ChassisOperation
                _chassis_op = (ChassisOperation.STORE
                               if current_mode == TeleopMode.STORE
                               else ChassisOperation.FETCH)
                if not chassis_fsm.is_active:
                    left_pressed  = (event == 'LEFT')
                    right_pressed = (event == 'RIGHT')
                    if left_pressed:
                        chassis_fsm.start('left', _chassis_op)
                    elif right_pressed:
                        chassis_fsm.start('right', _chassis_op)
                _fb_chassis = client.feedback if client is not None else {}
                chassis_fsm.tick(_fb_chassis, dt)
                if chassis_fsm.done:
                    current_mode = TeleopMode.MANUAL
                    mode_idx = available_modes.index(TeleopMode.MANUAL)
                    chassis_fsm.emergency_reset()

            else:
                # ── MANUAL mode ───────────────────────────────────────────
                ee_pos = np.clip(ee_pos + delta_p, cfg['pos_min'], cfg['pos_max'])

                if cfg['control_orientation']:
                    ee_rot = Rotation.from_rotvec(delta_r) * ee_rot

                # IK recovery resync
                if client is not None and not _homing_active:
                    fb = client.feedback
                    ik_ok = fb.get('ik_ok', True)
                    if _ik_was_failing and ik_ok and 'fk_x' in fb:
                        ee_pos = np.clip(
                            np.array([fb['fk_x'], fb['fk_y'], fb['fk_z']]),
                            cfg['pos_min'], cfg['pos_max'])
                        _ik_was_failing = False
                        print('[spacemouse] IK recovered — resynced to FK')
                    elif not ik_ok:
                        _ik_was_failing = True

                if _right_for_homing and not _homing_active:
                    if client is not None:
                        client.send_home()
                        _homing_active       = True
                        _homing_seen_running = False
                        _homing_t            = time.monotonic()
                        print('\n[spacemouse] Homing command sent — streaming paused')
                    else:
                        print('[spacemouse] No socket — skipping homing')

            # ── Homing monitor ────────────────────────────────────────────
            if _homing_active and client is not None:
                fb      = client.feedback
                homing  = fb.get('homing', False)
                elapsed = time.monotonic() - _homing_t
                if homing:
                    _homing_seen_running = True
                if _homing_seen_running and not homing:
                    if 'fk_x' in fb:
                        # ik_teleop_node atomically clears homing + sets fk_x,
                        # so this is the authoritative post-home position.
                        ee_pos = np.array([fb['fk_x'], fb['fk_y'], fb['fk_z']])
                        ee_rot = Rotation.from_quat(cfg['init_quat'])
                        print(f'\n[spacemouse] Homing done — resynced to FK: {ee_pos}')
                        _homing_active  = False
                        _arm_ready      = True
                        _ik_was_failing = False
                    # else: fk_x not yet in feedback — keep waiting this tick;
                    # the node will send it within the next 100 ms feedback interval.
                elif elapsed > 15.0:
                    # Hard timeout: use FK if available, init_pos as last resort.
                    ee_rot = Rotation.from_quat(cfg['init_quat'])
                    if 'fk_x' in fb:
                        ee_pos = np.array([fb['fk_x'], fb['fk_y'], fb['fk_z']])
                        print(f'\n[spacemouse] Homing timeout — resynced to FK: {ee_pos}')
                    else:
                        ee_pos = cfg['init_pos'].copy()
                        print('\n[spacemouse] Homing timeout — no FK, reset to init_pos')
                    _homing_active  = False
                    _arm_ready      = True
                    _ik_was_failing = False

            # ── Send pose ─────────────────────────────────────────────────
            if _arm_ready and not _homing_active:
                quat = ee_rot.as_quat()
                if client is not None:
                    client.send_pose(
                        x=float(ee_pos[0]),  y=float(ee_pos[1]),  z=float(ee_pos[2]),
                        qx=float(quat[0]),   qy=float(quat[1]),
                        qz=float(quat[2]),   qw=float(quat[3]),
                    )

            # ── Display ───────────────────────────────────────────────────
            if tick % print_every == 0:
                _fb = client.feedback if client is not None else {}
                hold_hint = ''
                if _right_hold_t is not None and not _right_hold_fired:
                    remaining = max(0.0, HOMING_HOLD_S - (now - _right_hold_t))
                    hold_hint = f'  [HOME in {remaining:.1f}s]'
                if _p_arc_recording:
                    if _p_arc_start_rot is not None:
                        _arc_rv  = (ee_rot * _p_arc_start_rot.inv()).as_rotvec()
                        _arc_deg = float(np.degrees(np.linalg.norm(_arc_rv)))
                    else:
                        _arc_deg = 0.0
                    mode_hdr = f'  [P-ARC RECORD] swept {_arc_deg:.1f}° — LEFT to finish'
                else:
                    mode_hdr = f'  [{current_mode.name}]  (BOTH to switch){hold_hint}'

                if not _arm_ready and not _homing_active:
                    print('\r[spacemouse] Waiting — press RIGHT to home and start   ',
                          end='', flush=True)
                elif _homing_active:
                    print(f'\r[spacemouse] HOMING...   ', end='', flush=True)
                elif current_mode == TeleopMode.ASSEMBLY and asm_fsm is not None:
                    render_assembly(asm_fsm, ee_pos, asm_cfg.difficulty,
                                    asm_cfg.q_target_deg, _fb,
                                    mode_header=mode_hdr, first=(tick == 0))
                elif current_mode == TeleopMode.PICKUP and pickup_fsm is not None:
                    render_pickup(pickup_fsm, ee_pos, _fb,
                                  mode_header=mode_hdr, first=(tick == 0))
                elif current_mode in (TeleopMode.STORE, TeleopMode.FETCH) and chassis_fsm is not None:
                    render_chassis(chassis_fsm, _fb,
                                   mode_header=mode_hdr, first=(tick == 0))
                else:
                    render_pose(ee_pos, sm_ang, cfg['pos_min'], cfg['pos_max'],
                                cfg['init_pos'], _fb,
                                header=mode_hdr, first=(tick == 0))

            tick += 1

            # ── Rate limit ────────────────────────────────────────────────
            elapsed = time.monotonic() - t_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        pass
    finally:
        if client is not None:
            client.stop()
        print('\n[spacemouse] Stopped.')


if __name__ == '__main__':
    main()
