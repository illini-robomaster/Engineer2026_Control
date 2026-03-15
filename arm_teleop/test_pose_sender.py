#!/usr/bin/env python3
"""
Fake teleop input — sends synthetic 6D poses over TCP to socket_teleop_node,
ik_teleop_node, or moveit_teleop_node without any computer vision.

Modes:
  static   — hold one fixed pose (good first sanity check)
  sweep    — slow sinusoidal motion along one axis
  circle   — circle in the XY plane around the neutral point
  figure8  — figure-8 (Lissajous 1:2) around the home/neutral point
  step     — discrete position steps (press Enter to advance)
  keyboard — live WASD-style nudge control

Usage:
  python test_pose_sender.py                          # static, default neutral
  python test_pose_sender.py --mode sweep --axis z
  python test_pose_sender.py --mode circle --radius 0.05
  python test_pose_sender.py --mode figure8 --radius 0.05 --plane yz
  python test_pose_sender.py --mode step
  python test_pose_sender.py --mode keyboard
  python test_pose_sender.py --host 192.168.1.10 --port 9999
"""

import argparse
import json
import math
import socket
import sys
import time


# ── Default neutral EE pose — derived from URDF FK at joint_angles=[0]*6 ──────
# Run:  python3 -c "... PyKDL FK ..."  to recompute if URDF changes.
# FK result: x=0.526073  y=-0.000001  z=0.396049
#            qx=0.707069  qy=-0.002084  qz=0.002014  qw=0.707139
DEFAULT_X = 0.526
DEFAULT_Y = 0.000
DEFAULT_Z = 0.396

# Quaternion ≈ 90° rotation around world X — matches URDF FK at home config.
DEFAULT_QX =  0.707
DEFAULT_QY =  0.000
DEFAULT_QZ =  0.000
DEFAULT_QW =  0.707


# ─────────────────────────────────────────────────────────────────────────────

def send(sock: socket.socket, x, y, z, qx=DEFAULT_QX, qy=DEFAULT_QY,
         qz=DEFAULT_QZ, qw=DEFAULT_QW):
    msg = json.dumps({'x': x, 'y': y, 'z': z,
                      'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw}) + '\n'
    sock.sendall(msg.encode())


def connect(host: str, port: int) -> socket.socket:
    print(f'Connecting to {host}:{port} …', end=' ', flush=True)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.settimeout(2.0)
    print('connected.')
    return s


# ── Modes ─────────────────────────────────────────────────────────────────────

def mode_static(sock, args):
    x, y, z = args.x, args.y, args.z
    print(f'Static pose: ({x:.3f}, {y:.3f}, {z:.3f})  — Ctrl-C to stop')
    rate = 1.0 / args.hz
    try:
        while True:
            send(sock, x, y, z)
            time.sleep(rate)
    except KeyboardInterrupt:
        pass


def mode_sweep(sock, args):
    axis  = args.axis.lower()
    amp   = args.amplitude
    freq  = args.freq
    rate  = 1.0 / args.hz
    x0, y0, z0 = args.x, args.y, args.z
    print(f'Sweep: {axis}-axis  amp={amp:.3f} m  freq={freq:.2f} Hz  — Ctrl-C to stop')
    t0 = time.monotonic()
    try:
        while True:
            t   = time.monotonic() - t0
            delta = amp * math.sin(2 * math.pi * freq * t)
            x = x0 + (delta if axis == 'x' else 0.0)
            y = y0 + (delta if axis == 'y' else 0.0)
            z = z0 + (delta if axis == 'z' else 0.0)
            send(sock, x, y, z)
            print(f'\r  pos=({x:+.3f}, {y:+.3f}, {z:+.3f})', end='', flush=True)
            time.sleep(rate)
    except KeyboardInterrupt:
        print()


def mode_circle(sock, args):
    r     = args.radius
    freq  = args.freq
    rate  = 1.0 / args.hz
    x0, y0, z0 = args.x, args.y, args.z
    print(f'Circle: r={r:.3f} m  freq={freq:.2f} Hz  plane=XY  — Ctrl-C to stop')
    t0 = time.monotonic()
    try:
        while True:
            t   = time.monotonic() - t0
            ang = 2 * math.pi * freq * t
            x   = x0 + r * math.cos(ang)
            y   = y0 + r * math.sin(ang)
            z   = z0
            send(sock, x, y, z)
            print(f'\r  pos=({x:+.3f}, {y:+.3f}, {z:+.3f})  ang={math.degrees(ang)%360:.1f}°',
                  end='', flush=True)
            time.sleep(rate)
    except KeyboardInterrupt:
        print()


def mode_figure8(sock, args):
    """
    Trace a figure-8 (Lissajous 1:2) around the neutral home position.

    The parametrisation uses two sinusoids with a 1:2 frequency ratio:
        slow_axis(t) = radius * sin(omega * t)
        fast_axis(t) = radius * sin(2 * omega * t)

    This produces a figure-8 whose lobes are symmetric around the center.
    Orientation is held constant at the home quaternion throughout.

    --plane chooses which two Cartesian axes carry the motion:
        xy  (default) — horizontal figure-8 (forward-back and side-to-side)
        xz  — forward-back and up-down
        yz  — side-to-side and up-down
    """
    r    = args.radius
    freq = args.freq
    rate = 1.0 / args.hz
    x0, y0, z0 = args.x, args.y, args.z
    plane = args.plane.lower()

    omega = 2 * math.pi * freq
    print(
        f'Figure-8: r={r:.3f} m  freq={freq:.2f} Hz  plane={plane}  '
        f'centre=({x0:.3f},{y0:.3f},{z0:.3f})  — Ctrl-C to stop'
    )
    t0 = time.monotonic()
    try:
        while True:
            t   = time.monotonic() - t0
            slow = r * math.sin(omega * t)
            fast = r * math.sin(2 * omega * t)

            if plane == 'xy':
                x, y, z = x0 + slow, y0 + fast, z0
            elif plane == 'xz':
                x, y, z = x0 + slow, y0, z0 + fast
            else:  # yz (default)
                x, y, z = x0, y0 + slow, z0 + fast

            send(sock, x, y, z)
            print(
                f'\r  pos=({x:+.3f}, {y:+.3f}, {z:+.3f})'
                f'  t={t:.1f}s',
                end='', flush=True,
            )
            time.sleep(rate)
    except KeyboardInterrupt:
        print()


def mode_step(sock, args):
    """Press Enter to step through a list of waypoints."""
    waypoints = [
        (args.x,        args.y,        args.z,        'home'),
        (args.x + 0.05, args.y,        args.z,        '+5cm X'),
        (args.x - 0.05, args.y,        args.z,        '-5cm X'),
        (args.x,        args.y + 0.05, args.z,        '+5cm Y'),
        (args.x,        args.y - 0.05, args.z,        '-5cm Y'),
        (args.x,        args.y,        args.z + 0.05, '+5cm Z'),
        (args.x,        args.y,        args.z - 0.05, '-5cm Z'),
        (args.x,        args.y,        args.z,        'home'),
    ]
    rate = 1.0 / args.hz
    print('Step mode — press Enter to advance to next waypoint, Ctrl-C to quit.')
    for x, y, z, label in waypoints:
        try:
            input(f'  → [{label}]  ({x:.3f}, {y:.3f}, {z:.3f})  press Enter …')
        except KeyboardInterrupt:
            break
        print(f'    Sending ({x:.3f}, {y:.3f}, {z:.3f}) for 2 s …')
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            send(sock, x, y, z)
            time.sleep(rate)


def mode_keyboard(sock, args):
    """WASD / QE nudge control — no curses, just line-buffered Enter presses."""
    STEP = args.step
    x, y, z = args.x, args.y, args.z
    rate = 1.0 / args.hz

    print(f'Keyboard nudge (step={STEP*100:.1f} cm). Commands:')
    print('  w/s  → X +/-      a/d → Y +/-      q/e → Z +/-')
    print('  r    → reset to home     p → print current     Ctrl-C → quit')

    import threading

    cmd_queue = []
    lock      = threading.Lock()
    stop_flag = threading.Event()

    def input_thread():
        while not stop_flag.is_set():
            try:
                c = input().strip().lower()
                with lock:
                    cmd_queue.append(c)
            except (EOFError, KeyboardInterrupt):
                stop_flag.set()

    t = threading.Thread(target=input_thread, daemon=True)
    t.start()

    try:
        while not stop_flag.is_set():
            with lock:
                cmds = cmd_queue[:]
                cmd_queue.clear()
            for c in cmds:
                if   c == 'w': x += STEP
                elif c == 's': x -= STEP
                elif c == 'a': y += STEP
                elif c == 'd': y -= STEP
                elif c == 'q': z += STEP
                elif c == 'e': z -= STEP
                elif c == 'r': x, y, z = args.x, args.y, args.z
                elif c == 'p':
                    print(f'  current: ({x:.3f}, {y:.3f}, {z:.3f})')
            send(sock, x, y, z)
            time.sleep(rate)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
    print(f'\nStopped at ({x:.3f}, {y:.3f}, {z:.3f})')


# ── Main ──────────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser(description='Fake teleop pose sender for testing.')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=9999)
    p.add_argument('--mode', default='static',
                   choices=['static', 'sweep', 'circle', 'figure8', 'step', 'keyboard'])
    p.add_argument('--hz',   type=float, default=30.0,
                   help='Send rate Hz (default: 30)')

    # Neutral pose
    p.add_argument('--x', type=float, default=DEFAULT_X)
    p.add_argument('--y', type=float, default=DEFAULT_Y)
    p.add_argument('--z', type=float, default=DEFAULT_Z)

    # Sweep options
    p.add_argument('--axis',      default='x', choices=['x', 'y', 'z'])
    p.add_argument('--amplitude', type=float, default=0.05,
                   help='Sweep amplitude in metres (default: 0.05)')
    p.add_argument('--freq',      type=float, default=0.2,
                   help='Motion frequency Hz (default: 0.2 = one cycle per 5 s)')

    # Circle / figure-8 options
    p.add_argument('--radius', type=float, default=0.04,
                   help='Circle/figure-8 radius in metres (default: 0.04)')
    p.add_argument('--plane', default='xy', choices=['xy', 'xz', 'yz'],
                   help='Plane for figure-8 motion: xy | xz | yz (default: xy = horizontal)')

    # Keyboard options
    p.add_argument('--step', type=float, default=0.01,
                   help='Keyboard nudge step in metres (default: 0.01)')
    return p.parse_args()


def main():
    args = build_args()
    try:
        sock = connect(args.host, args.port)
    except ConnectionRefusedError:
        sys.exit(f'[ERROR] Connection refused — is the arm bringup running?')
    except OSError as e:
        sys.exit(f'[ERROR] {e}')

    dispatch = {
        'static':   mode_static,
        'sweep':    mode_sweep,
        'circle':   mode_circle,
        'figure8':  mode_figure8,
        'step':     mode_step,
        'keyboard': mode_keyboard,
    }
    try:
        dispatch[args.mode](sock, args)
    finally:
        sock.close()
        print('Socket closed.')


if __name__ == '__main__':
    main()
