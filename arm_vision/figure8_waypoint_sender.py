#!/usr/bin/env python3
"""
Send a fixed waypoint figure-eight trajectory over the arm teleop socket.

This mirrors the `spacemouse_teleop.py` transport path, but replaces manual
SpaceMouse input with a discrete horizontal figure-eight in the Y-Z plane.

Examples:
    python arm_vision/figure8_waypoint_sender.py
    python arm_vision/figure8_waypoint_sender.py --host 172.16.51.47 --port 9999
    python arm_vision/figure8_waypoint_sender.py --amp-y 0.05 --amp-z 0.03 --waypoints 48
    python arm_vision/figure8_waypoint_sender.py --no-socket
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pose_waypoint_runner import (
    PoseWaypointRunner,
    RunnerStatus,
    WaypointDef,
    WaypointSequence,
    pose_from_feedback,
)


def load_config(path: str) -> dict:
    import yaml

    cfg_path = Path(path)
    if not cfg_path.is_file():
        cfg_path = Path(__file__).resolve().parent / path

    with open(cfg_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    cfg = raw['spacemouse']
    cfg['init_pos'] = np.array(cfg['init_pos'], dtype=float)
    cfg['init_quat'] = np.array(cfg['init_quat'], dtype=float)
    cfg['pos_min'] = np.array(cfg['pos_min'], dtype=float)
    cfg['pos_max'] = np.array(cfg['pos_max'], dtype=float)
    cfg['poll_hz'] = float(cfg['poll_hz'])
    return cfg


def build_horizontal_figure8_sequence(
    amp_y: float,
    amp_z: float,
    waypoint_count: int,
    loops: int,
    hold_s: float,
    timeout_s: float,
    pos_tolerance_mm: float,
    ori_tolerance_deg: float,
) -> WaypointSequence:
    if amp_y <= 0.0 or amp_z <= 0.0:
        raise ValueError('Figure-eight amplitudes must be positive')
    if waypoint_count < 8:
        raise ValueError('waypoint_count must be at least 8')
    if loops < 1:
        raise ValueError('loops must be at least 1')

    base_wp = WaypointDef(
        name='start_hold',
        position=np.zeros(3, dtype=float),
        rotation=Rotation.identity(),
        hold_s=float(hold_s),
        timeout_s=float(timeout_s),
        pos_tolerance_m=float(pos_tolerance_mm) / 1000.0,
        ori_tolerance_deg=float(ori_tolerance_deg),
    )

    waypoints = [base_wp]
    total_steps = waypoint_count * loops
    for step in range(1, total_steps + 1):
        theta = (2.0 * np.pi * step) / waypoint_count
        y = float(amp_y * np.sin(theta))
        z = float(amp_z * np.sin(2.0 * theta))
        waypoints.append(
            WaypointDef(
                name=f'fig8_{step:03d}',
                position=np.array([0.0, y, z], dtype=float),
                rotation=Rotation.identity(),
                hold_s=float(hold_s),
                timeout_s=float(timeout_s),
                pos_tolerance_m=float(pos_tolerance_mm) / 1000.0,
                ori_tolerance_deg=float(ori_tolerance_deg),
            )
        )

    return WaypointSequence(
        name='horizontal_figure8_yz',
        frame='relative_start',
        description='Horizontal figure-eight in the y-z plane around the start pose.',
        waypoints=waypoints,
    )


def validate_sequence_workspace(
    sequence: WaypointSequence,
    start_pos: np.ndarray,
    pos_min: np.ndarray,
    pos_max: np.ndarray,
) -> None:
    violations: list[str] = []
    for wp in sequence.waypoints:
        absolute = start_pos + wp.position
        below = absolute < pos_min
        above = absolute > pos_max
        if np.any(below) or np.any(above):
            violations.append(
                f'{wp.name}: abs={absolute.tolist()} outside '
                f'[{pos_min.tolist()} .. {pos_max.tolist()}]'
            )

    if violations:
        raise ValueError(
            'Figure-eight waypoints exceed workspace bounds:\n  '
            + '\n  '.join(violations[:8])
            + ('\n  ...' if len(violations) > 8 else '')
        )


def _feedback_from_pose(pos: np.ndarray, rot: Rotation, **extra) -> dict:
    quat = rot.as_quat()
    feedback = {
        'fk_x': float(pos[0]),
        'fk_y': float(pos[1]),
        'fk_z': float(pos[2]),
        'fk_qx': float(quat[0]),
        'fk_qy': float(quat[1]),
        'fk_qz': float(quat[2]),
        'fk_qw': float(quat[3]),
        'ik_ok': True,
    }
    feedback.update(extra)
    return feedback


def _wait_for_start_pose(
    client,
    fallback_pos: np.ndarray,
    fallback_rot: Rotation,
    timeout_s: float,
) -> tuple[np.ndarray, Rotation, str]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        fb = client.feedback
        if 'fk_x' in fb:
            return pose_from_feedback(fb, fallback_pos, fallback_rot)
        time.sleep(0.05)
    return fallback_pos.copy(), fallback_rot, 'config'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='figure8_waypoint_sender',
        description='Send a waypoint-based horizontal figure-eight over the teleop socket.',
    )
    parser.add_argument('--host', default='172.16.51.47', help='ROS host IP')
    parser.add_argument('--port', type=int, default=9999, help='ROS socket port')
    parser.add_argument(
        '--config',
        default='config/spacemouse_params.yaml',
        help='Path to spacemouse_params.yaml',
    )
    parser.add_argument(
        '--amp-y',
        type=float,
        default=0.05,
        help='Figure-eight amplitude along +/−Y in metres',
    )
    parser.add_argument(
        '--amp-z',
        type=float,
        default=0.03,
        help='Figure-eight amplitude along +/−Z in metres',
    )
    parser.add_argument(
        '--waypoints',
        type=int,
        default=32,
        help='Number of discrete waypoints per figure-eight loop',
    )
    parser.add_argument('--loops', type=int, default=1, help='How many figure-eight loops to run')
    parser.add_argument('--hold-s', type=float, default=0.0, help='Hold time at each waypoint')
    parser.add_argument(
        '--timeout-s',
        type=float,
        default=2.0,
        help='Timeout before marking a waypoint failed',
    )
    parser.add_argument(
        '--pos-tolerance-mm',
        type=float,
        default=8.0,
        help='Position tolerance for waypoint completion',
    )
    parser.add_argument(
        '--ori-tolerance-deg',
        type=float,
        default=5.0,
        help='Orientation tolerance for waypoint completion',
    )
    parser.add_argument(
        '--send-hz',
        type=float,
        default=None,
        help='Pose streaming rate; defaults to poll_hz from config',
    )
    parser.add_argument(
        '--fk-wait-s',
        type=float,
        default=2.0,
        help='Time to wait for FK feedback before falling back to config pose',
    )
    parser.add_argument(
        '--quat',
        type=float,
        nargs=4,
        default=None,
        metavar=('QX', 'QY', 'QZ', 'QW'),
        help='Override the locked EE orientation; default is current FK/config orientation',
    )
    parser.add_argument(
        '--log-dir',
        default='.',
        help='Directory for waypoint CSV logs; pass empty string to disable',
    )
    parser.add_argument(
        '--no-socket',
        action='store_true',
        help='Dry run: do not connect, simulate perfect feedback locally',
    )
    return parser


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    fallback_pos = cfg['init_pos'].copy()
    fallback_rot = Rotation.from_quat(cfg['init_quat'])
    send_hz = float(args.send_hz) if args.send_hz else float(cfg['poll_hz'])
    send_dt = 1.0 / max(send_hz, 1.0)

    client = None
    if args.no_socket:
        print('[figure8] --no-socket: simulating trajectory locally.')
        start_pos = fallback_pos.copy()
        start_rot = fallback_rot
        start_source = 'config'
    else:
        from arm_vision.socket_client import PoseSocketClient

        print(f'[figure8] Connecting to {args.host}:{args.port}...')
        client = PoseSocketClient(host=args.host, port=args.port)
        client.start()
        start_pos, start_rot, start_source = _wait_for_start_pose(
            client,
            fallback_pos,
            fallback_rot,
            timeout_s=float(args.fk_wait_s),
        )

    if args.quat is not None:
        start_rot = Rotation.from_quat(np.array(args.quat, dtype=float))
        start_source += '+quat_override'

    sequence = build_horizontal_figure8_sequence(
        amp_y=float(args.amp_y),
        amp_z=float(args.amp_z),
        waypoint_count=int(args.waypoints),
        loops=int(args.loops),
        hold_s=float(args.hold_s),
        timeout_s=float(args.timeout_s),
        pos_tolerance_mm=float(args.pos_tolerance_mm),
        ori_tolerance_deg=float(args.ori_tolerance_deg),
    )
    validate_sequence_workspace(sequence, start_pos, cfg['pos_min'], cfg['pos_max'])

    runner = PoseWaypointRunner({sequence.name: sequence}, log_dir=args.log_dir or '')
    runner.start(start_pos, start_rot, now=time.monotonic())

    print(
        '[figure8] Start pose source='
        f'{start_source} pos={np.array2string(start_pos, precision=4)} '
        f'quat={np.array2string(start_rot.as_quat(), precision=4)}'
    )
    print(
        '[figure8] Sequence='
        f'{sequence.name} loops={args.loops} waypoints_per_loop={args.waypoints} '
        f'amp_y={args.amp_y:.3f}m amp_z={args.amp_z:.3f}m'
    )

    last_pose_pos = start_pos.copy()
    last_pose_rot = start_rot
    last_status_line = ''

    try:
        while True:
            now = time.monotonic()
            if client is not None:
                feedback = client.feedback
                last_pose_pos, last_pose_rot, _ = pose_from_feedback(
                    feedback,
                    last_pose_pos,
                    last_pose_rot,
                )
            else:
                target_pos, target_rot = runner.current_target_pose
                if target_pos is None or target_rot is None:
                    target_pos = last_pose_pos
                    target_rot = last_pose_rot
                feedback = _feedback_from_pose(target_pos, target_rot, ik_result='SIM')
                last_pose_pos, last_pose_rot = target_pos.copy(), target_rot

            cmd_pos, cmd_rot = runner.tick(feedback, last_pose_pos, last_pose_rot, now=now)

            if client is not None:
                quat = cmd_rot.as_quat()
                client.send_pose(
                    float(cmd_pos[0]),
                    float(cmd_pos[1]),
                    float(cmd_pos[2]),
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                )

            status_line = runner.status_text
            if status_line != last_status_line:
                print(f'[figure8] {status_line}')
                last_status_line = status_line

            if not runner.is_active:
                break

            time.sleep(send_dt)
    except KeyboardInterrupt:
        print('[figure8] Interrupted; cancelling current sequence.')
        runner.cancel('keyboard_interrupt', {}, last_pose_pos, last_pose_rot, now=time.monotonic())
    finally:
        if client is not None:
            client.stop()

    if runner.status == RunnerStatus.SUCCEEDED:
        if runner.last_log_path:
            print(f'[figure8] Complete. Log saved to {runner.last_log_path}')
        else:
            print('[figure8] Complete.')
        return 0

    if runner.status == RunnerStatus.CANCELLED:
        if runner.last_log_path:
            print(f'[figure8] Cancelled. Log saved to {runner.last_log_path}')
        else:
            print('[figure8] Cancelled.')
        return 130

    if runner.last_log_path:
        print(f'[figure8] Failed: {runner.status_text} | log={runner.last_log_path}', file=sys.stderr)
    else:
        print(f'[figure8] Failed: {runner.status_text}', file=sys.stderr)
    return 1


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
