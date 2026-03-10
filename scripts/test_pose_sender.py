#!/usr/bin/env python3
"""
test_pose_sender.py — mock arm_vision TCP sender for testing the full pipeline.

Simulates what arm_vision/socket_client.py would send, allowing you to test:
  socket_teleop_node → MoveIt Servo → arm_controller → (uart_bridge_node)

Protocol: newline-delimited JSON on TCP port 9999
  {"x": 0.1, "y": 0.2, "z": 0.3, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}

Usage:
  # Interactive mode (default): type target poses at the prompt
  python3 scripts/test_pose_sender.py

  # Demo mode: automatically steps through preset waypoints
  python3 scripts/test_pose_sender.py --demo

  # Hold a fixed position
  python3 scripts/test_pose_sender.py --hold 0.426 0.0 0.395

  # Custom host/port
  python3 scripts/test_pose_sender.py --host 192.168.1.100 --port 9999

The arm should already be running:
  ./scripts/run_ros2_humble_moveit_control.sh
"""

import argparse
import json
import math
import socket
import sys
import time

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = '\033[0m'
BOLD   = '\033[1m'
CYAN   = '\033[96m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'

# ── Nominal EE origin (from workspace.yaml) — safe starting target ───────────
DEFAULT_X, DEFAULT_Y, DEFAULT_Z = 0.426, 0.0, 0.395

# ── Preset demo waypoints [x, y, z, qx, qy, qz, qw] ──────────────────────────
DEMO_WAYPOINTS = [
    # (label, x,     y,     z,     qx,   qy,   qz,   qw,  hold_s)
    ("home",       DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 3.0),
    ("+X forward", DEFAULT_X+0.08,  DEFAULT_Y,       DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 3.0),
    ("home",       DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 2.0),
    ("+Y left",    DEFAULT_X,       DEFAULT_Y+0.08,  DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 3.0),
    ("home",       DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 2.0),
    ("-Y right",   DEFAULT_X,       DEFAULT_Y-0.08,  DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 3.0),
    ("home",       DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 2.0),
    ("+Z up",      DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z+0.08,  0.0, 0.0, 0.0, 1.0, 3.0),
    ("home",       DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 2.0),
    ("-Z down",    DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z-0.08,  0.0, 0.0, 0.0, 1.0, 3.0),
    ("home",       DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       0.0, 0.0, 0.0, 1.0, 3.0),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame(x, y, z, qx, qy, qz, qw) -> str:
    return json.dumps({'x': x, 'y': y, 'z': z,
                       'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw}) + '\n'


def _connect(host: str, port: int) -> socket.socket:
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3.0)
            s.connect((host, port))
            s.settimeout(None)
            print(f'{GREEN}Connected to {host}:{port}{RESET}')
            return s
        except OSError as e:
            print(f'{YELLOW}Waiting for socket_teleop_node... ({e}){RESET}')
            time.sleep(1.0)


def _send(sock: socket.socket, frame: str) -> bool:
    try:
        sock.sendall(frame.encode())
        return True
    except OSError:
        return False


def _fmt(x, y, z, qx, qy, qz, qw) -> str:
    return (f'x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  '
            f'qx={qx:+.3f}  qy={qy:+.3f}  qz={qz:+.3f}  qw={qw:+.3f}')


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_demo(host: str, port: int) -> None:
    print(f'{BOLD}{CYAN}=== Demo mode ==={RESET}')
    print('Stepping through preset waypoints. Press Ctrl-C to stop.\n')
    sock = _connect(host, port)
    rate = 1 / 30.0  # 30 Hz send rate

    try:
        for label, x, y, z, qx, qy, qz, qw, hold in DEMO_WAYPOINTS:
            print(f'{BOLD}→ {label:<14}{RESET}  {_fmt(x, y, z, qx, qy, qz, qw)}', flush=True)
            deadline = time.monotonic() + hold
            frame = _make_frame(x, y, z, qx, qy, qz, qw)
            while time.monotonic() < deadline:
                if not _send(sock, frame):
                    print(f'{YELLOW}Connection lost, reconnecting...{RESET}')
                    sock = _connect(host, port)
                time.sleep(rate)
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Interrupted.{RESET}')
    finally:
        sock.close()


def run_hold(host: str, port: int, x: float, y: float, z: float) -> None:
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    print(f'{BOLD}{CYAN}=== Hold mode ==={RESET}')
    print(f'Sending fixed target at {_fmt(x, y, z, qx, qy, qz, qw)}')
    print('Press Ctrl-C to stop.\n')
    sock = _connect(host, port)
    frame = _make_frame(x, y, z, qx, qy, qz, qw)
    rate = 1 / 30.0
    try:
        while True:
            if not _send(sock, frame):
                sock = _connect(host, port)
                frame = _make_frame(x, y, z, qx, qy, qz, qw)
            time.sleep(rate)
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Stopped.{RESET}')
    finally:
        sock.close()


def run_interactive(host: str, port: int) -> None:
    print(f'{BOLD}{CYAN}=== Interactive mode ==={RESET}')
    print('Enter a target pose and the arm will move toward it (P-controller).')
    print('Formats:')
    print(f'  {BOLD}x y z{RESET}               — position only (quaternion = identity)')
    print(f'  {BOLD}x y z  qx qy qz qw{RESET}  — full 6D pose')
    print(f'  {BOLD}home{RESET}                 — go to ee_origin {DEFAULT_X} {DEFAULT_Y} {DEFAULT_Z}')
    print(f'  {BOLD}stop{RESET}                 — disconnect (arm halts via timeout)')
    print(f'  {BOLD}Ctrl-C{RESET}               — quit\n')

    sock = _connect(host, port)
    rate = 1 / 30.0
    current_frame = _make_frame(DEFAULT_X, DEFAULT_Y, DEFAULT_Z, 0, 0, 0, 1)

    import threading

    stop_event = threading.Event()

    def sender():
        nonlocal current_frame
        while not stop_event.is_set():
            if current_frame and not _send(sock, current_frame):
                print(f'\n{YELLOW}Connection lost, reconnecting...{RESET}', flush=True)
                try:
                    new_s = _connect(host, port)
                    sock.__class__ = new_s.__class__
                    sock._sock = new_s._sock
                except Exception:
                    pass
            time.sleep(rate)

    t = threading.Thread(target=sender, daemon=True)
    t.start()

    try:
        while True:
            try:
                raw = input(f'{CYAN}pose>{RESET} ').strip()
            except EOFError:
                break
            if not raw:
                continue
            if raw.lower() == 'stop':
                current_frame = None
                print(f'{YELLOW}Stopped sending. Socket closed. Arm will halt via timeout.{RESET}')
                time.sleep(0.5)
                sock.close()
                break
            if raw.lower() == 'home':
                raw = f'{DEFAULT_X} {DEFAULT_Y} {DEFAULT_Z}'

            parts = raw.split()
            try:
                if len(parts) == 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
                elif len(parts) == 7:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    qx, qy, qz, qw = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                else:
                    print(f'{RED}Need 3 or 7 numbers. Example: 0.426 0.0 0.395{RESET}')
                    continue
            except ValueError:
                print(f'{RED}Invalid number. Example: 0.426 0.0 0.395{RESET}')
                continue

            current_frame = _make_frame(x, y, z, qx, qy, qz, qw)
            print(f'  → sending  {_fmt(x, y, z, qx, qy, qz, qw)}', flush=True)

    except KeyboardInterrupt:
        print(f'\n{YELLOW}Quit.{RESET}')
    finally:
        stop_event.set()
        try:
            sock.close()
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Mock arm_vision TCP sender — test the full ROS pipeline without vision.')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Host running socket_teleop_node (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9999,
                        help='TCP port (default: 9999)')
    parser.add_argument('--demo', action='store_true',
                        help='Run preset waypoint demo sequence')
    parser.add_argument('--hold', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        help='Hold a fixed EE target position continuously')
    args = parser.parse_args()

    print(f'\n{BOLD}Pipeline Test Sender{RESET}')
    print(f'ee_origin (nominal): x={DEFAULT_X}  y={DEFAULT_Y}  z={DEFAULT_Z}')
    print(f'Target: {args.host}:{args.port}\n')

    if args.demo:
        run_demo(args.host, args.port)
    elif args.hold:
        run_hold(args.host, args.port, *args.hold)
    else:
        run_interactive(args.host, args.port)


if __name__ == '__main__':
    main()
