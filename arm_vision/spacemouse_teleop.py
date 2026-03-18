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
import sys
import time
from enum import Enum, auto

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bar(value: float, lo: float, hi: float, width: int = 10) -> str:
    """Return a filled bar representing *value* within [lo, hi]."""
    span  = hi - lo if hi != lo else 1.0
    frac  = max(0.0, min(1.0, (value - lo) / span))
    filled = round(frac * width)
    return '█' * filled + '░' * (width - filled)


_VIZ_LINES     = 4   # number of lines render_pose prints
_ASM_VIZ_LINES = 6   # number of lines render_assembly / render_pickup print


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
    ]

    if not first:
        print(f'\033[{_ASM_VIZ_LINES}A', end='')

    for line in lines:
        print(line.ljust(width))


# ── Mode + Button combo ───────────────────────────────────────────────────────

class TeleopMode(Enum):
    MANUAL   = auto()
    ASSEMBLY = auto()
    PICKUP   = auto()


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
    p.add_argument('--difficulty', type=int, default=None,
                   help='Override difficulty level (1-4)')
    p.add_argument('--q-angle', type=float, default=None,
                   help='Override Q-axis target angle (degrees)')
    p.add_argument('--yaw', type=float, default=None,
                   help='Override approach yaw angle (degrees, 0=+X)')
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

    # ── Build available mode cycle ─────────────────────────────────────────
    available_modes = [TeleopMode.MANUAL]
    if asm_fsm is not None:
        available_modes.append(TeleopMode.ASSEMBLY)
    if pickup_fsm is not None:
        available_modes.append(TeleopMode.PICKUP)

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

    combo_detector = ButtonComboDetector()

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
            right_edge = bool(len(buttons) > 1 and buttons[1] and not prev_buttons[1])
            prev_buttons = buttons[:2] if len(buttons) >= 2 else buttons + [0]

            # ── Combo detection ───────────────────────────────────────────
            event = combo_detector.update(left_edge, right_edge, now)

            # Always compute delta_p — used for arc control even when gated
            delta_p = (cfg['lin_map'] @ sm_lin) * cfg['linear_speed'] * args.speed * dt
            delta_r = (cfg['ang_map'] @ sm_ang) * cfg['angular_speed'] * args.speed * dt

            # ── COMBO: cycle mode ─────────────────────────────────────────
            if event == 'COMBO':
                ee_pos, ee_rot = cycle_mode()
                print(f'\n[spacemouse] Switched to mode: {current_mode.name}')
                # Consume event — no further button processing this tick
                event = 'NONE'

            # ── Homing (RIGHT in non-FSM modes, or from any mode) ────────
            # RIGHT alone triggers homing in MANUAL
            # Assembly/Pickup use RIGHT for FSM home (handled per-mode below)
            _right_for_homing = (event == 'RIGHT'
                                  and current_mode == TeleopMode.MANUAL)
            _right_for_asm    = (event == 'RIGHT'
                                  and current_mode == TeleopMode.ASSEMBLY)
            _right_for_pickup = (event == 'RIGHT'
                                  and current_mode == TeleopMode.PICKUP)

            # ── Mode dispatch ─────────────────────────────────────────────

            if current_mode == TeleopMode.ASSEMBLY and asm_fsm is not None:
                from assembly_fsm import AssemblyState

                if _right_for_asm and not _homing_active:
                    asm_fsm.emergency_reset()
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
                    if asm_fsm.state == AssemblyState.IDLE:
                        asm_fsm.start()
                    elif asm_fsm.state == AssemblyState.READY_CONFIRM:
                        asm_fsm.confirm()
                    elif asm_fsm.state in (AssemblyState.CONFIRMED, AssemblyState.ABORTED):
                        asm_fsm.reset()
                    else:
                        asm_fsm.advance()

                if asm_fsm.translation_allowed:
                    ee_pos = np.clip(ee_pos + delta_p, cfg['pos_min'], cfg['pos_max'])

                rot_override = asm_fsm.rotation_override
                if rot_override is not None:
                    ee_rot = rot_override

                sm_active = float(np.linalg.norm(sm_lin)) > cfg['deadband']
                asm_fsm.tick(ee_pos, ee_rot, sm_active=sm_active,
                             sm_lin_delta=delta_p, sm_rot_delta=delta_r, dt=dt)

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
                    ee_rot = Rotation.from_quat(cfg['init_quat'])
                    if 'fk_x' in fb:
                        ee_pos = np.array([fb['fk_x'], fb['fk_y'], fb['fk_z']])
                        print(f'\n[spacemouse] Homing done — resynced to FK: {ee_pos}')
                    else:
                        ee_pos = cfg['init_pos'].copy()
                        print('\n[spacemouse] Homing done — no FK, reset to init_pos')
                    _homing_active  = False
                    _arm_ready      = True
                    _ik_was_failing = False
                elif elapsed > 15.0:
                    ee_pos = cfg['init_pos'].copy()
                    ee_rot = Rotation.from_quat(cfg['init_quat'])
                    _homing_active  = False
                    _arm_ready      = True
                    _ik_was_failing = False
                    print('\n[spacemouse] Homing timeout — reset to init_pos')

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
                mode_hdr = f'  [{current_mode.name}]  (BOTH buttons to switch modes)'

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
