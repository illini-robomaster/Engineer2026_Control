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


_VIZ_LINES = 3   # number of lines render_pose prints

def render_pose(
    pos: 'np.ndarray',
    sm_ang: 'np.ndarray',
    pos_min: 'np.ndarray',
    pos_max: 'np.ndarray',
    init_pos: 'np.ndarray',
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

    if not first:
        # Move cursor up _VIZ_LINES lines to overwrite previous display
        print(f'\033[{_VIZ_LINES}A', end='')

    print(line3)
    print(line1)
    print(line2)


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

    # ── State ─────────────────────────────────────────────────────────────────
    ee_pos = cfg['init_pos'].copy()
    ee_rot = Rotation.from_quat(cfg['init_quat'])

    dt          = 1.0 / cfg['poll_hz']
    print_every = int(cfg['poll_hz'])   # status line once per second
    tick        = 0

    print('[spacemouse] Running — Ctrl-C to stop.  Button 0 resets to home.')
    print(f'[spacemouse] init_pos={ee_pos}  speed={args.speed}')

    try:
        while True:
            t_start = time.monotonic()

            state = device.read()
            if state is None:
                time.sleep(dt)
                continue

            # ── Position delta ─────────────────────────────────────────────
            sm_lin  = np.array([state.x, state.y, state.z], dtype=float)
            sm_lin  = np.where(np.abs(sm_lin) > cfg['deadband'], sm_lin, 0.0)
            delta_p = (cfg['lin_map'] @ sm_lin) * cfg['linear_speed'] * args.speed * dt
            ee_pos  = np.clip(ee_pos + delta_p, cfg['pos_min'], cfg['pos_max'])

            # ── Raw angular input (always captured for visualization) ──────
            sm_ang = np.array([state.roll, state.pitch, state.yaw], dtype=float)
            sm_ang = np.where(np.abs(sm_ang) > cfg['deadband'], sm_ang, 0.0)

            # ── Orientation delta (optional) ───────────────────────────────
            if cfg['control_orientation']:
                delta_r = (cfg['ang_map'] @ sm_ang) * cfg['angular_speed'] * args.speed * dt
                ee_rot  = Rotation.from_rotvec(delta_r) * ee_rot

            quat = ee_rot.as_quat()   # [x, y, z, w]

            # ── Button 0 → reset to home ───────────────────────────────────
            if state.buttons and state.buttons[0]:
                ee_pos = cfg['init_pos'].copy()
                ee_rot = Rotation.from_quat(cfg['init_quat'])
                print('[spacemouse] Reset to home')

            # ── Send / print ───────────────────────────────────────────────
            if client is not None:
                client.send_pose(
                    x=float(ee_pos[0]),  y=float(ee_pos[1]),  z=float(ee_pos[2]),
                    qx=float(quat[0]),   qy=float(quat[1]),
                    qz=float(quat[2]),   qw=float(quat[3]),
                )

            if tick % print_every == 0:
                render_pose(ee_pos, sm_ang, cfg['pos_min'], cfg['pos_max'],
                            cfg['init_pos'], first=(tick == 0))

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
