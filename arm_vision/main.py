#!/usr/bin/env python3
"""
arm_vision — standalone AprilTag-based teleoperation client.

Subcommands:
  calibrate-camera   Calibrate the webcam using a checkerboard pattern.
  calibrate-cube     Identify the 5 AprilTag faces and measure cube geometry.
  run                Detect cube pose and stream EE targets to the ROS node.

Usage examples:
  python main.py calibrate-camera
  python main.py calibrate-cube
  python main.py run
  python main.py run --host 192.168.1.100 --port 9999 --show
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ─────────────────────────────────────────────────────────────────────────────
# calibrate-camera
# ─────────────────────────────────────────────────────────────────────────────

def cmd_calibrate_camera(args: argparse.Namespace):
    from arm_vision.camera_cal import run
    run(
        device=args.device,
        cols=args.cols,
        rows=args.rows,
        square_size=args.square_size,
        output=args.output,
    )


# ─────────────────────────────────────────────────────────────────────────────
# calibrate-cube
# ─────────────────────────────────────────────────────────────────────────────

def cmd_calibrate_cube(args: argparse.Namespace):
    from arm_vision.cube_cal import run
    run(
        device=args.device,
        tag_family=args.tag_family,
        tag_size=args.tag_size,
        camera_cal=args.camera_cal,
        output=args.output,
    )


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace):
    from arm_vision.camera_cal import load as load_camera_cal
    from arm_vision.cube_cal   import load as load_cube_cfg
    from arm_vision.detector   import CubeDetector
    from arm_vision.workspace  import WorkspaceMapper
    from arm_vision.socket_client import PoseSocketClient

    print('Loading calibration files…')
    K, dist = load_camera_cal(args.camera_cal)
    cube_cfg = load_cube_cfg(args.cube_config)

    detector = CubeDetector(K, dist, cube_cfg)
    mapper   = WorkspaceMapper(args.workspace_config)

    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)
    cap.set(cv2.CAP_PROP_GAIN, 100)
    cap.set(cv2.CAP_PROP_CONTRAST, 200)
    print(f"Exposure:  {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Gain:      {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"Contrast:  {cap.get(cv2.CAP_PROP_CONTRAST)}")
    if not cap.isOpened():
        sys.exit(f'[ERROR] Cannot open camera device {args.device}')

    if args.no_socket:
        print('Socket disabled — running in local visualization mode.')
        client = None
    else:
        print(f'Connecting to ROS node at {args.host}:{args.port}…')
        client = PoseSocketClient(host=args.host, port=args.port)
        client.start()

    print('Running — press Q or ESC in the preview window to stop.')
    print()

    frame_count = 0
    send_count  = 0
    t0 = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            cube_pos, cube_quat, raw = detector.detect(frame)

            if cube_pos is not None:
                ee_pos, ee_quat = mapper.map(cube_pos, cube_quat)
                if client is not None:
                    client.send_pose(
                        x=float(ee_pos[0]),  y=float(ee_pos[1]),  z=float(ee_pos[2]),
                        qx=float(ee_quat[0]), qy=float(ee_quat[1]),
                        qz=float(ee_quat[2]), qw=float(ee_quat[3]),
                    )
                send_count += 1

            frame_count += 1

            if args.show:
                vis = detector.draw(frame, raw)

                status = 'TRACKING' if cube_pos is not None else 'NO CUBE'
                colour = (0, 220, 0) if cube_pos is not None else (0, 0, 220)
                cv2.putText(vis, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

                if cube_pos is not None:
                    ee_pos_disp = mapper.map_position(cube_pos)
                    cv2.putText(
                        vis,
                        f'EE target: ({ee_pos_disp[0]:.3f}, '
                        f'{ee_pos_disp[1]:.3f}, {ee_pos_disp[2]:.3f}) m',
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 0), 2,
                    )

                    cv2.putText(
                        vis,
                        f'Cube Pose: x={cube_pos[0]:.3f} y={cube_pos[1]:.3f} z={cube_pos[2]:.3f}',
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2
                    )

                    r = Rotation.from_quat(ee_quat)
                    yaw, pitch, roll = r.as_euler('zyx', degrees=True)

                    cv2.putText(
                        vis,
                        f'EE Quat: {ee_quat[0]:.2f}, {ee_quat[1]:.2f}, {ee_quat[2]:.2f}, {ee_quat[3]:.2f}',
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )
                    cv2.putText(
                        vis,
                        f'EE RPY: {roll:.1f}, {pitch:.1f}, {yaw:.1f}',
                        (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )


                elapsed = time.monotonic() - t0
                fps = frame_count / max(elapsed, 1e-3)
                cv2.putText(vis, f'FPS: {fps:.1f}  sent: {send_count}',
                            (10, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

                cv2.imshow('arm_vision', vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break

            else:
                # Headless: print status every 5 seconds
                elapsed = time.monotonic() - t0
                if frame_count % 150 == 0:
                    fps = frame_count / max(elapsed, 1e-3)
                    status = (f'cube @ ({cube_pos[0]:.3f}, {cube_pos[1]:.3f}, '
                              f'{cube_pos[2]:.3f}) m')  if cube_pos is not None else 'no cube'
                    print(f'[{elapsed:6.1f}s] {fps:.1f} fps  {status}')

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if client is not None:
            client.stop()
        print(f'\nStopped.  Sent {send_count} pose messages.')


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='arm_vision',
        description='AprilTag cube teleoperation client (no ROS required).',
    )
    sub = p.add_subparsers(dest='command', required=True)

    # ── calibrate-camera ──────────────────────────────────────────────────────
    cc = sub.add_parser('calibrate-camera',
                        help='Calibrate camera with a checkerboard.')
    cc.add_argument('--device',      type=int,   default=0)
    cc.add_argument('--cols',        type=int,   default=8,
                    help='Inner corners along wide axis (default: 9)')
    cc.add_argument('--rows',        type=int,   default=5,
                    help='Inner corners along short axis (default: 6)')
    cc.add_argument('--square-size', type=float, default=0.025,
                    dest='square_size',
                    help='Physical square side in metres (default: 0.025)')
    cc.add_argument('--output',      default='data/camera_calibration.yaml')

    # ── calibrate-cube ────────────────────────────────────────────────────────
    cb = sub.add_parser('calibrate-cube',
                        help='Calibrate 5-face AprilTag cube geometry.')
    cb.add_argument('--device',      type=int,   default=0)
    cb.add_argument('--tag-family',  default='tag36h11', dest='tag_family')
    cb.add_argument('--tag-size', type=float, default=0.032,
                    dest='tag_size',
                    help='AprilTag black-square side in metres (default: 0.032)')
    cb.add_argument('--camera-cal',  default='data/camera_calibration.yaml',
                    dest='camera_cal')
    cb.add_argument('--output',      default='data/cube_config.yaml')

    # ── run ───────────────────────────────────────────────────────────────────
    rn = sub.add_parser('run', help='Stream cube pose to ROS socket_teleop_node.')
    rn.add_argument('--device',           type=int,   default=0)
    rn.add_argument('--host',             default='127.0.0.1',
                    help='ROS host (default: 127.0.0.1)')
    rn.add_argument('--port',             type=int,   default=9999)
    rn.add_argument('--camera-cal',       default='data/camera_calibration.yaml',
                    dest='camera_cal')
    rn.add_argument('--cube-config',      default='data/cube_config.yaml',
                    dest='cube_config')
    rn.add_argument('--workspace-config', default='config/workspace.yaml',
                    dest='workspace_config')
    rn.add_argument('--show',             action='store_true',
                    help='Display live preview window')
    rn.add_argument('--no-socket',        action='store_true', dest='no_socket',
                    help='Skip socket connection (local visualization only)')

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        'calibrate-camera': cmd_calibrate_camera,
        'calibrate-cube':   cmd_calibrate_cube,
        'run':              cmd_run,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
