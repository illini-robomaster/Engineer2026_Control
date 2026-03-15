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
import csv
import os
import sys
import threading
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
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_euler_gauges(img, roll: float, pitch: float, yaw: float,
                       x: int = 10, y: int = 185) -> None:
    """Draw three horizontal bar gauges for roll, pitch, and yaw."""
    bar_w   = 200   # total bar width in pixels
    bar_h   = 12    # bar height in pixels
    gap     = 22    # vertical gap between bars
    labels  = [('Roll',  roll,  180.0, (100, 200, 255)),
               ('Pitch', pitch,  90.0, (100, 255, 150)),
               ('Yaw',   yaw,   180.0, (255, 180, 80))]

    for i, (name, angle, full_range, colour) in enumerate(labels):
        cy = y + i * gap

        # Background track
        cv2.rectangle(img, (x, cy), (x + bar_w, cy + bar_h), (60, 60, 60), -1)

        # Centre tick (zero reference)
        mid = x + bar_w // 2
        cv2.line(img, (mid, cy), (mid, cy + bar_h), (160, 160, 160), 1)

        # Filled portion — clamp to [-full_range, full_range]
        clamped  = max(-full_range, min(full_range, angle))
        fraction = clamped / full_range          # -1 … +1
        fill_px  = int(fraction * (bar_w // 2))  # signed pixels from centre

        if fill_px >= 0:
            cv2.rectangle(img, (mid, cy), (mid + fill_px, cy + bar_h), colour, -1)
        else:
            cv2.rectangle(img, (mid + fill_px, cy), (mid, cy + bar_h), colour, -1)

        # Label + value
        cv2.putText(img, f'{name}: {angle:+.1f}°',
                    (x + bar_w + 6, cy + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


def _draw_reach_gauge(img, reach_frac: float,
                      x: int = 10, y: int = 30, h: int = 200) -> None:
    """Vertical reach gauge on the right side of the frame.

    Green at the bottom (safe), yellow in the middle, red at top (limit).
    A marker shows the current reach percentage.
    """
    w = 20
    clamped = max(0.0, min(1.0, reach_frac))

    # Background track
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)

    # Gradient fill: bottom = green, top = red
    fill_h = int(clamped * h)
    if fill_h > 0:
        for i in range(fill_h):
            frac = i / max(h - 1, 1)
            r_c = int(frac * 255)
            g_c = int((1.0 - frac) * 220)
            py = y + h - 1 - i
            cv2.line(img, (x + 1, py), (x + w - 1, py), (0, g_c, r_c), 1)

    # Marker triangle
    marker_y = y + h - fill_h
    pts = np.array([
        [x - 6, marker_y - 4],
        [x - 6, marker_y + 4],
        [x,     marker_y],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))

    # Percentage label
    pct = int(clamped * 100)
    if clamped < 0.70:
        col = (0, 200, 0)
    elif clamped < 0.90:
        col = (0, 200, 255)
    else:
        col = (0, 0, 255)
    cv2.putText(img, f'{pct}%', (x - 8, y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.putText(img, 'REACH', (x - 10, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace):
    from arm_vision.camera_cal import load as load_camera_cal
    from arm_vision.cube_cal   import load as load_cube_cfg
    from arm_vision.detector   import CubeDetector
    from arm_vision.workspace  import WorkspaceMapper
    from arm_vision.pose_filter import PoseFilter
    from arm_vision.socket_client import PoseSocketClient

    import yaml as _yaml

    print('Loading calibration files…')
    K, dist = load_camera_cal(args.camera_cal)
    cube_cfg = load_cube_cfg(args.cube_config)

    # Camera intrinsics for safe-zone overlay projection
    _fx, _fy = float(K[0, 0]), float(K[1, 1])
    _cx, _cy = float(K[0, 2]), float(K[1, 2])

    detector = CubeDetector(K, dist, cube_cfg)
    mapper   = WorkspaceMapper(args.workspace_config)

    with open(args.workspace_config) as _f:
        _ws_cfg = _yaml.safe_load(_f)['workspace']
    _filt_cfg = _ws_cfg.get('filter', {})
    pose_filter = PoseFilter(
        process_noise=float(_filt_cfg.get('process_noise', 0.02)),
        meas_noise=float(_filt_cfg.get('meas_noise', 0.005)),
        max_vel=float(_filt_cfg.get('max_vel', 0.5)),
        max_coast_s=float(_filt_cfg.get('max_coast_s', 0.5)),
    )

    import platform
    backend = cv2.CAP_V4L2 if platform.system() == 'Linux' else cv2.CAP_AVFOUNDATION
    cap = cv2.VideoCapture(args.device, backend)
    if not cap.isOpened():
        sys.exit(f'[ERROR] Cannot open camera device {args.device}')
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, -8)
    cap.set(cv2.CAP_PROP_GAIN, 100)
    cap.set(cv2.CAP_PROP_CONTRAST, 200)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    print(f"Exposure:  {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Gain:      {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"Contrast:  {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"FPS:       {cap.get(cv2.CAP_PROP_FPS)}")

    if args.no_socket:
        print('Socket disabled — running in local visualization mode.')
        client = None
    else:
        print(f'Connecting to ROS node at {args.host}:{args.port}…')
        client = PoseSocketClient(host=args.host, port=args.port)
        client.start()

    # ── Detection thread ───────────────────────────────────────────────────
    # Camera capture + AprilTag detection run in a background thread.
    # The send loop (main thread) runs at send_rate_hz, polling the latest
    # detection and feeding Kalman predictions between detections.
    det_lock = threading.Lock()
    det_state = {'pos': None, 'quat': None, 'raw': [], 'frame': None,
                 'seq': 0, 'running': True, 'cam_fail': 0,
                 'det_ms': 0.0, 'n_known': 0}

    def _detection_loop():
        while det_state['running']:
            # ── Drain camera buffer to get the freshest frame ─────────
            # OpenCV buffers 3-5 frames internally; reading sequentially
            # would process stale data (50-80 ms behind reality at 60 fps).
            # Grab-without-decode drains the buffer cheaply, then we decode
            # only the last frame.
            ret = False
            for _ in range(5):
                grabbed = cap.grab()
                if not grabbed:
                    break
                ret = True
            if ret:
                ret, frame = cap.retrieve()

            if not ret:
                det_state['cam_fail'] += 1
                if det_state['cam_fail'] == 10:
                    print('[ERROR] Camera is not returning frames. '
                          'On macOS, check System Settings → Privacy & Security → Camera '
                          'and grant access to your terminal app, then restart.')
                time.sleep(0.001)
                continue
            det_state['cam_fail'] = 0
            t_det_start = time.monotonic()
            cube_pos, cube_quat, raw = detector.detect(frame)
            det_elapsed_ms = (time.monotonic() - t_det_start) * 1000.0
            n_known = sum(1 for d in raw if hasattr(d, 'tag_id'))
            with det_lock:
                det_state['pos']     = cube_pos
                det_state['quat']    = cube_quat
                det_state['raw']     = raw
                det_state['frame']   = frame
                det_state['seq']    += 1
                det_state['det_ms']  = det_elapsed_ms
                det_state['n_known'] = n_known

    det_thread = threading.Thread(target=_detection_loop, daemon=True)
    det_thread.start()

    # ── Diagnostic CSV logger ────────────────────────────────────────────────
    log_file = None
    log_writer = None
    if args.log:
        log_path = args.log
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        log_file = open(log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            't_mono',           # monotonic timestamp (s)
            'dt_ms',            # time since last tick (ms)
            'state',            # TRACK / COAST / NONE
            'n_tags',           # number of known tags detected
            'det_ms',           # detection latency (ms) — 0 if no new frame
            'cube_x', 'cube_y', 'cube_z',   # raw detection (cam frame)
            'filt_x', 'filt_y', 'filt_z',   # Kalman output (cam frame)
            'ee_x',   'ee_y',   'ee_z',     # mapped EE target (robot frame)
            'reach',            # reach fraction 0..1
            'kf_vx',  'kf_vy',  'kf_vz',   # Kalman velocity estimate
            'q_boost',          # Kalman direction-change boost (1.0 = none)
        ])
        print(f'Diagnostic CSV → {log_path}')

    # ── Send loop (main thread) ────────────────────────────────────────────
    send_dt    = 1.0 / args.send_rate
    display_dt = 1.0 / 30.0    # cap display at 30 fps to not slow the send loop

    send_count   = 0
    detect_count = 0
    coast_count  = 0
    tick         = 0
    last_seq     = -1
    t0           = time.monotonic()
    t_prev       = t0
    t_last_disp  = t0

    print(f'Running — send rate {args.send_rate} Hz.')
    print(f'  Cube at cam_origin → arm at ee_home.  Deltas scaled by gain.')
    print(f'  H = re-home (clutch): resets reference, arm stays in place.')
    print(f'  Q / ESC = stop.')
    print()

    try:
        while det_state['running']:
            t_now  = time.monotonic()
            dt     = t_now - t_prev
            t_prev = t_now

            # ── Poll latest detection ──────────────────────────────────────
            with det_lock:
                d_pos   = det_state['pos']
                d_quat  = det_state['quat']
                d_raw   = det_state['raw']
                d_frame = det_state['frame']
                d_seq   = det_state['seq']
                d_det_ms  = det_state['det_ms']
                d_n_known = det_state['n_known']

            new_detection = (d_seq != last_seq)
            last_seq = d_seq

            # Feed Kalman: measurement on new detection, predict-only otherwise
            if new_detection and d_pos is not None:
                filt_pos, filt_quat = pose_filter.step(d_pos, d_quat, dt)
                detect_count += 1
            elif new_detection and d_pos is None:
                # Detection ran but cube wasn't found — predict only
                filt_pos, filt_quat = pose_filter.step(None, None, dt)
                coast_count += 1
            else:
                # No new frame — pure Kalman prediction
                filt_pos, filt_quat = pose_filter.step(None, None, dt)

            # ── Map + send ─────────────────────────────────────────────────
            ee_pos = ee_quat = None
            reach_frac = 0.0
            if filt_pos is not None:
                ee_pos, ee_quat, reach_frac = mapper.map(filt_pos, filt_quat)
                if client is not None:
                    client.send_pose(
                        x=float(ee_pos[0]),  y=float(ee_pos[1]),  z=float(ee_pos[2]),
                        qx=float(ee_quat[0]), qy=float(ee_quat[1]),
                        qz=float(ee_quat[2]), qw=float(ee_quat[3]),
                    )
                send_count += 1

            tick += 1

            # ── CSV diagnostic log ──────────────────────────────────────────
            if log_writer is not None:
                if new_detection and d_pos is not None:
                    state_str = 'TRACK'
                elif filt_pos is not None:
                    state_str = 'COAST'
                else:
                    state_str = 'NONE'
                kf_vel = pose_filter.velocity
                q_boost = pose_filter.last_q_boost
                log_writer.writerow([
                    f'{t_now:.4f}',
                    f'{dt*1000:.2f}',
                    state_str,
                    d_n_known if new_detection else '',
                    f'{d_det_ms:.1f}' if new_detection else '',
                    f'{d_pos[0]:.5f}' if d_pos is not None else '',
                    f'{d_pos[1]:.5f}' if d_pos is not None else '',
                    f'{d_pos[2]:.5f}' if d_pos is not None else '',
                    f'{filt_pos[0]:.5f}' if filt_pos is not None else '',
                    f'{filt_pos[1]:.5f}' if filt_pos is not None else '',
                    f'{filt_pos[2]:.5f}' if filt_pos is not None else '',
                    f'{ee_pos[0]:.5f}' if ee_pos is not None else '',
                    f'{ee_pos[1]:.5f}' if ee_pos is not None else '',
                    f'{ee_pos[2]:.5f}' if ee_pos is not None else '',
                    f'{reach_frac:.4f}',
                    f'{kf_vel[0]:.4f}', f'{kf_vel[1]:.4f}', f'{kf_vel[2]:.4f}',
                    f'{q_boost:.1f}',
                ])

            # ── Display (capped at 30 fps) ─────────────────────────────────
            if args.show and d_frame is not None and (t_now - t_last_disp) >= display_dt:
                t_last_disp = t_now
                vis = detector.draw(d_frame, d_raw)
                h_img, w_img = vis.shape[:2]

                detected = new_detection and d_pos is not None
                coasting = (not detected) and (filt_pos is not None)
                if coasting:
                    status, colour = 'COASTING', (0, 180, 255)
                elif detected:
                    status, colour = 'TRACKING', (0, 220, 0)
                else:
                    status, colour = 'NO CUBE', (0, 0, 220)
                cv2.putText(vis, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)

                if ee_pos is not None:
                    cv2.putText(
                        vis,
                        f'EE target: ({ee_pos[0]:.3f}, '
                        f'{ee_pos[1]:.3f}, {ee_pos[2]:.3f}) m',
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 0), 2,
                    )

                    if d_pos is not None:
                        cv2.putText(
                            vis,
                            f'Cube: x={d_pos[0]:.3f} y={d_pos[1]:.3f} z={d_pos[2]:.3f}',
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2
                        )

                    r = Rotation.from_quat(ee_quat)
                    yaw, pitch, roll = r.as_euler('zyx', degrees=True)
                    cv2.putText(vis,
                                f'RPY: {roll:.1f}, {pitch:.1f}, {yaw:.1f}',
                                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    _draw_euler_gauges(vis, roll, pitch, yaw, x=10, y=165)

                # ── Safe zone overlay ─────────────────────────────────
                # Shows how far the cube can move before hitting limits.
                # The rectangle is centred on the cube and shrinks as the
                # arm approaches its reach boundary.
                if filt_pos is not None and filt_pos[2] > 0.05:
                    cam_z = float(filt_pos[2])
                    sx_lo, sx_hi, sy_lo, sy_hi = mapper.safe_zone_cam(filt_pos, cam_z)
                    # Project camera metres → pixels
                    u_lo = int(_fx * sx_lo / cam_z + _cx)
                    u_hi = int(_fx * sx_hi / cam_z + _cx)
                    v_lo = int(_fy * sy_lo / cam_z + _cy)
                    v_hi = int(_fy * sy_hi / cam_z + _cy)
                    # Clamp to image bounds
                    u_lo = max(0, min(u_lo, w_img - 1))
                    u_hi = max(0, min(u_hi, w_img - 1))
                    v_lo = max(0, min(v_lo, h_img - 1))
                    v_hi = max(0, min(v_hi, h_img - 1))

                    # Colour based on reach fraction
                    if reach_frac < 0.70:
                        zone_col = (0, 200, 0)       # green — plenty of room
                    elif reach_frac < 0.90:
                        zone_col = (0, 200, 255)     # yellow — approaching limit
                    else:
                        zone_col = (0, 0, 255)       # red — at/beyond limit

                    cv2.rectangle(vis, (u_lo, v_lo), (u_hi, v_hi), zone_col, 2)

                    # Red border flash when at the limit
                    if reach_frac >= 0.90:
                        cv2.rectangle(vis, (0, 0),
                                      (w_img - 1, h_img - 1), (0, 0, 255), 3)


                # ── Reach gauge bar ───────────────────────────────────
                _draw_reach_gauge(vis, reach_frac, x=w_img - 50, y=30, h=200)

                elapsed = time.monotonic() - t0
                send_hz = send_count / max(elapsed, 1e-3)
                det_pct = detect_count / max(tick, 1) * 100
                cv2.putText(vis,
                            f'send: {send_hz:.0f}Hz  det: {det_pct:.0f}%  coast: {coast_count}',
                            (10, vis.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

                cv2.putText(vis, 'H=re-home',
                            (w_img - 130, h_img - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)

                cv2.imshow('arm_vision', vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                if key in (ord('h'), ord('H')):
                    if filt_pos is not None and ee_pos is not None:
                        mapper.re_home(filt_pos, filt_quat, ee_pos)
                        print(f'[home] Re-homed — arm stays at '
                              f'({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})')

            elif not args.show:
                # Headless: print status every 5 seconds
                elapsed = time.monotonic() - t0
                if tick % (args.send_rate * 5) == 0 and tick > 0:
                    send_hz = send_count / max(elapsed, 1e-3)
                    det_pct = detect_count / max(tick, 1) * 100
                    if filt_pos is not None:
                        s = (f'filt @ ({filt_pos[0]:.3f}, {filt_pos[1]:.3f}, '
                             f'{filt_pos[2]:.3f}) m')
                    else:
                        s = 'no cube'
                    print(f'[{elapsed:6.1f}s] send={send_hz:.0f}Hz  det={det_pct:.0f}%  {s}')

            # ── Rate limit ─────────────────────────────────────────────────
            sleep_t = send_dt - (time.monotonic() - t_now)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        pass
    finally:
        det_state['running'] = False
        det_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        if client is not None:
            client.stop()
        if log_file is not None:
            log_file.close()
            print(f'Diagnostic CSV written → {args.log}')
        elapsed = time.monotonic() - t0
        det_pct = detect_count / max(tick, 1) * 100
        print(f'\nStopped.  {send_count} poses sent at ~{send_count/max(elapsed,1e-3):.0f}Hz, '
              f'{detect_count} detections ({det_pct:.0f}%), '
              f'{coast_count} coasted in {elapsed:.1f}s.')


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
    rn.add_argument('--fps',              type=int,   default=60,
                    help='Target camera framerate (default: 60)')
    rn.add_argument('--no-socket',        action='store_true', dest='no_socket',
                    help='Skip socket connection (local visualization only)')
    rn.add_argument('--send-rate',        type=int,   default=100,
                    dest='send_rate',
                    help='Kalman filter / send loop rate in Hz (default: 100)')
    rn.add_argument('--log',              type=str,   default=None,
                    help='Write diagnostic CSV to this path (e.g. logs/diag.csv)')

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
