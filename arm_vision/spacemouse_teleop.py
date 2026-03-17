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
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def _bar(value: float, lo: float, hi: float, width: int = 10) -> str:
    """Return a filled bar representing *value* within [lo, hi]."""
    span  = hi - lo if hi != lo else 1.0
    frac  = max(0.0, min(1.0, (value - lo) / span))
    filled = round(frac * width)
    return '█' * filled + '░' * (width - filled)


_VIZ_LINES     = 4   # number of lines render_pose prints
_ASM_VIZ_LINES = 6   # number of lines render_assembly prints

def render_pose(
    pos: 'np.ndarray',
    sm_ang: 'np.ndarray',
    pos_min: 'np.ndarray',
    pos_max: 'np.ndarray',
    init_pos: 'np.ndarray',
    feedback: dict,
    first: bool = False,
) -> None:
    """Overwrite the last _VIZ_LINES terminal lines with a live 6-DOF display."""
    roll, pitch, yaw = sm_ang   # raw SpaceMouse values in [-1, 1]

    w = 12  # bar width

    # Center bars on init_pos; half-range = half the workspace span per axis
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
    line3 = "─" * len(line1)
    if 'fk_x' in feedback:
        ik_str = 'OK' if feedback.get('ik_ok', True) else 'SNAP/FAIL'
        line4 = (f"  FK  X {feedback['fk_x']:+7.3f}m"
                 f"   Y {feedback['fk_y']:+7.3f}m"
                 f"   Z {feedback['fk_z']:+7.3f}m   IK:{ik_str}")
    else:
        line4 = "  FK  (no feedback yet)"

    if not first:
        # Move cursor up _VIZ_LINES lines to overwrite previous display
        print(f'\033[{_VIZ_LINES}A', end='')

    print(line3)
    print(line1)
    print(line2)
    print(line4)


def render_assembly(
    fsm: 'AssemblyFSM',
    pos: 'np.ndarray',
    difficulty: int,
    q_target: float,
    feedback: dict,
    first: bool = False,
) -> None:
    """Overwrite terminal lines with assembly-mode status display."""
    from assembly_fsm import AssemblyState

    width = 60
    sep = '─' * width

    # Status line for READY_CONFIRM
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

    lines = [
        sep,
        f'  [ASSEMBLY] Difficulty {difficulty} | Q-target: {q_target}°',
        f'  State: {fsm.stage_label}',
        f'  Cmd: X {pos[0]:+7.3f}  Y {pos[1]:+7.3f}  Z {pos[2]:+7.3f}',
        hint,
        fb_line,
    ]

    if not first:
        print(f'\033[{_ASM_VIZ_LINES}A', end='')

    for line in lines:
        print(line.ljust(width))


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
    p.add_argument('--difficulty', type=int, default=None,
                   help='Override difficulty level (1-4)')
    p.add_argument('--q-angle', type=float, default=None,
                   help='Override Q-axis target angle (degrees)')
    return p


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    cfg = raw['spacemouse']
    # Convert flat lists → 3×3 numpy matrices
    cfg['lin_map']   = np.array(cfg['lin_map'],   dtype=float).reshape(3, 3)
    cfg['ang_map']   = np.array(cfg['ang_map'],   dtype=float).reshape(3, 3)
    cfg['pos_min']   = np.array(cfg['pos_min'],   dtype=float)
    cfg['pos_max']   = np.array(cfg['pos_max'],   dtype=float)
    cfg['init_pos']  = np.array(cfg['init_pos'],  dtype=float)
    cfg['init_quat'] = np.array(cfg['init_quat'], dtype=float)
    return cfg


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

    # ── Assembly mode (optional) ───────────────────────────────────────────
    if args.assembly:
        from assembly_fsm import AssemblyFSM, AssemblyState, TaskConfig, load_assembly_config
        asm_cfg = load_assembly_config(args.assembly,
                                        difficulty_override=args.difficulty,
                                        q_angle_override=args.q_angle)
        fsm = AssemblyFSM(asm_cfg)
        print(f'[assembly] Difficulty {asm_cfg.difficulty}, '
              f'Q-angle={asm_cfg.q_target_deg}°')
    else:
        fsm = None

    # ── State ─────────────────────────────────────────────────────────────────
    ee_pos = cfg['init_pos'].copy()
    ee_rot = Rotation.from_quat(cfg['init_quat'])

    dt          = 1.0 / cfg['poll_hz']
    print_every = int(cfg['poll_hz'])   # status line once per second
    tick        = 0
    prev_buttons = [0, 0]   # for edge detection
    _arm_ready           = args.no_socket  # False until first homing completes (True in dry-run)
    _homing_active       = False  # True from send_home() until homing + resync done
    _homing_seen_running = False  # True once feedback showed homing=True
    _homing_t            = 0.0   # monotonic time when homing was triggered
    _ik_was_failing      = False  # True when last feedback showed ik_ok=False (manual mode)

    if fsm is not None:
        print('[spacemouse] Assembly mode — RIGHT: home and start, LEFT: advance/confirm.')
    else:
        print('[spacemouse] Ready — press RIGHT to home and start streaming.')
    print(f'[spacemouse] init_pos={ee_pos}  speed={args.speed}')

    try:
        while True:
            t_start = time.monotonic()

            state = device.read()
            if state is None:
                time.sleep(dt)
                continue

            # ── Raw inputs ─────────────────────────────────────────────────
            sm_lin = np.array([state.x, state.y, state.z], dtype=float)
            sm_lin = np.where(np.abs(sm_lin) > cfg['deadband'], sm_lin, 0.0)

            sm_ang = np.array([state.roll, state.pitch, state.yaw], dtype=float)
            sm_ang = np.where(np.abs(sm_ang) > cfg['deadband'], sm_ang, 0.0)

            # ── Button edge detection ──────────────────────────────────────
            buttons = list(state.buttons) if state.buttons else [0, 0]
            left_edge  = buttons[0] and not prev_buttons[0]
            right_edge = len(buttons) > 1 and buttons[1] and not prev_buttons[1]
            prev_buttons = buttons[:2] if len(buttons) >= 2 else buttons + [0]

            # Always compute delta_p — used for arc control even when translation gated
            delta_p = (cfg['lin_map'] @ sm_lin) * cfg['linear_speed'] * args.speed * dt

            if fsm is not None:
                # ── Assembly mode ──────────────────────────────────────────

                # RIGHT → home + reset FSM (always, from any state)
                if right_edge and not _homing_active:
                    fsm.emergency_reset()
                    ee_rot = Rotation.from_quat(cfg['init_quat'])
                    if client is not None:
                        client.send_home()
                        _homing_active       = True
                        _homing_seen_running = False
                        _homing_t            = time.monotonic()
                        print('\n[spacemouse] Homing command sent — streaming paused')
                    else:
                        print('\n[spacemouse] No socket — skipping homing')

                # LEFT button — per-state advancement (skip if right was pressed)
                if not right_edge:
                    if fsm.state == AssemblyState.IDLE:
                        if left_edge:
                            fsm.start()
                            ee_rot = fsm.init_orientation
                    elif fsm.state == AssemblyState.READY_CONFIRM:
                        if left_edge:
                            fsm.confirm()
                    elif fsm.state in (AssemblyState.CONFIRMED, AssemblyState.ABORTED):
                        if left_edge:
                            fsm.reset()
                    else:
                        if left_edge:
                            fsm.advance()

                # Input gating: translation only during allowed stages
                if fsm.translation_allowed:
                    ee_pos = np.clip(ee_pos + delta_p, cfg['pos_min'], cfg['pos_max'])

                # Orientation: never from SpaceMouse during assembly
                rot_override = fsm.rotation_override
                if rot_override is not None:
                    ee_rot = rot_override

                # FSM tick — pass sm_lin_delta for manual Q arc control
                sm_active = float(np.linalg.norm(sm_lin)) > cfg['deadband']
                fsm.tick(ee_pos, ee_rot, sm_active=sm_active,
                         sm_lin_delta=delta_p, dt=dt)

                # Position override (auto-lift / arc motion) — no clip, arc is trusted
                pos_override = fsm.position_override
                if pos_override is not None:
                    ee_pos = pos_override

            else:
                # ── Pure manual mode ───────────────────────────────────────
                ee_pos = np.clip(ee_pos + delta_p, cfg['pos_min'], cfg['pos_max'])

                if cfg['control_orientation']:
                    delta_r = (cfg['ang_map'] @ sm_ang) * cfg['angular_speed'] * args.speed * dt
                    ee_rot = Rotation.from_rotvec(delta_r) * ee_rot

                # IK recovery resync: when arm was stuck at workspace boundary,
                # snap ee_pos to actual FK on recovery so motion resumes cleanly.
                if client is not None and not _homing_active:
                    fb = client.feedback
                    ik_ok = fb.get('ik_ok', True)   # default True = no freeze
                    if _ik_was_failing and ik_ok and 'fk_x' in fb:
                        ee_pos = np.clip(
                            np.array([fb['fk_x'], fb['fk_y'], fb['fk_z']]),
                            cfg['pos_min'], cfg['pos_max'])
                        _ik_was_failing = False
                        print('[spacemouse] IK recovered — resynced to FK')
                    elif not ik_ok:
                        _ik_was_failing = True

                # LEFT → reset to home
                if left_edge:
                    ee_pos = cfg['init_pos'].copy()
                    ee_rot = Rotation.from_quat(cfg['init_quat'])
                    print('[spacemouse] Reset to home')

                # RIGHT → trigger homing node via TCP
                if right_edge and not _homing_active:
                    if client is not None:
                        client.send_home()
                        _homing_active       = True
                        _homing_seen_running = False
                        _homing_t            = time.monotonic()
                        print('\n[spacemouse] Homing command sent — streaming paused')
                    else:
                        print('[spacemouse] No socket — skipping homing')

            # ── Homing monitor — driven by FK feedback homing flag ───────────
            if _homing_active and client is not None:
                fb      = client.feedback
                homing  = fb.get('homing', False)
                elapsed = time.monotonic() - _homing_t
                if homing:
                    _homing_seen_running = True
                # Transition: seen running → no longer running → resync
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

            # ── Send pose (skipped until arm is ready and while homing) ─────────
            if _arm_ready and not _homing_active:
                quat = ee_rot.as_quat()
                if client is not None:
                    client.send_pose(
                        x=float(ee_pos[0]),  y=float(ee_pos[1]),  z=float(ee_pos[2]),
                        qx=float(quat[0]),   qy=float(quat[1]),
                        qz=float(quat[2]),   qw=float(quat[3]),
                    )

            # ── Display ────────────────────────────────────────────────────
            if tick % print_every == 0:
                _fb = client.feedback if client is not None else {}
                if not _arm_ready and not _homing_active:
                    print('\r[spacemouse] Waiting — press RIGHT to home and start   ',
                          end='', flush=True)
                elif _homing_active:
                    print(f'\r[spacemouse] HOMING...   ', end='', flush=True)
                elif fsm is not None:
                    render_assembly(fsm, ee_pos, asm_cfg.difficulty,
                                    asm_cfg.q_target_deg, _fb, first=(tick == 0))
                else:
                    render_pose(ee_pos, sm_ang, cfg['pos_min'], cfg['pos_max'],
                                cfg['init_pos'], _fb, first=(tick == 0))

            tick += 1

            # ── Rate limit ─────────────────────────────────────────────────
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
