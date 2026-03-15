#!/usr/bin/env python3
"""
test_pose_sender.py — mock arm_vision TCP sender for testing the full pipeline.

Usage:
  python3 scripts/test_pose_sender.py [mode] [options]

Modes:
  (none)      Interactive — type target poses at the prompt  (default)
  demo        Step through preset axis-sweep waypoints
  figure8     Continuous figure-8 in the Y-Z plane
  hold X Y Z  Hold a fixed EE position

Options:
  --host      TCP host  (default: 127.0.0.1)
  --port      TCP port  (default: 9999)
  --amp-y M   figure8 Y amplitude in metres  (default: 0.06)
  --amp-z M   figure8 Z amplitude in metres  (default: 0.04)
  --quat QX QY QZ QW  fixed orientation for figure8  (default: home FK pose)
"""

import argparse
import json
import math
import random
import socket
import sys
import threading
import time
from pathlib import Path

# ── FK helpers (pure numpy — no ROS needed) ───────────────────────────────────

def _rpy_xyz_to_mat4(rpy, xyz):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    import numpy as np
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,              cp*cr],
    ])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = xyz
    return T

def _axis_angle_mat4(axis, angle):
    import numpy as np
    ax, ay, az = axis
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [t*ax*ax + c,     t*ax*ay - s*az,  t*ax*az + s*ay],
        [t*ax*ay + s*az,  t*ay*ay + c,     t*ay*az - s*ax],
        [t*ax*az - s*ay,  t*ay*az + s*ax,  t*az*az + c],
    ])
    return T

def _load_chain():
    """Parse URDF and return (chain_data, lowers, uppers).

    chain_data entries: {type, T_fixed, axis}
    lowers/uppers: numpy arrays of joint limits in chain order.
    """
    import numpy as np
    # Locate URDF relative to this script
    script_dir = Path(__file__).resolve().parent
    candidates = list(script_dir.parent.glob('**/*.urdf'))
    if not candidates:
        raise FileNotFoundError('No .urdf file found under ' + str(script_dir.parent))
    urdf_path = candidates[0]

    # urdf_parser_py ships with ROS Humble
    ros_site = '/opt/ros/humble/lib/python3.10/site-packages'
    if ros_site not in sys.path:
        sys.path.insert(0, ros_site)
    from urdf_parser_py import urdf as urdf_parser

    model = urdf_parser.URDF.from_xml_string(urdf_path.read_bytes())
    jfc = {j.child: j for j in model.joints}
    path = []
    cur = 'End_Effector'
    while cur != 'base_link':
        j = jfc[cur]; path.append(j); cur = j.parent
    path.reverse()

    chain, lowers, uppers = [], [], []
    for j in path:
        xyz = list(j.origin.xyz) if j.origin and j.origin.xyz else [0., 0., 0.]
        rpy = list(j.origin.rpy) if j.origin and j.origin.rpy else [0., 0., 0.]
        axis = np.array(list(j.axis) if j.axis else [1., 0., 0.], dtype=float)
        n = np.linalg.norm(axis)
        if n > 1e-10: axis /= n
        chain.append({'type': j.type, 'T_fixed': _rpy_xyz_to_mat4(rpy, xyz), 'axis': axis})
        if j.type in ('revolute', 'continuous'):
            lowers.append(float(j.limit.lower) if j.limit else -math.pi)
            uppers.append(float(j.limit.upper) if j.limit else  math.pi)
    return chain, np.array(lowers), np.array(uppers)

def _fk_pos(chain, q):
    """Return EE position (x, y, z) for joint array q."""
    import numpy as np
    T = np.eye(4); qi = 0
    for seg in chain:
        T = T @ seg['T_fixed']
        if seg['type'] in ('revolute', 'continuous'):
            T = T @ _axis_angle_mat4(seg['axis'], q[qi]); qi += 1
    return T[:3, 3]


# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = '\033[0m'
BOLD   = '\033[1m'
CYAN   = '\033[96m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'

# ── Nominal EE origin ─────────────────────────────────────────────────────────
DEFAULT_X, DEFAULT_Y, DEFAULT_Z = 0.526, 0.0, 0.396

# Home EE orientation from FK at q=[0]*6 (≈90° around X, computed from URDF).
# Identity (0,0,0,1) would force the arm to rotate 90° from its natural pose.
HOME_QX, HOME_QY, HOME_QZ, HOME_QW = 0.7071, 0.0, 0.0, 0.7071

# ── Preset demo waypoints ─────────────────────────────────────────────────────
DEMO_WAYPOINTS = [
    # (label,        x,               y,               z,               hold_s)
    ("home",         DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       3.0),
    ("+X forward",   DEFAULT_X+0.08,  DEFAULT_Y,       DEFAULT_Z,       3.0),
    ("home",         DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       2.0),
    ("+Y left",      DEFAULT_X,       DEFAULT_Y+0.08,  DEFAULT_Z,       3.0),
    ("home",         DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       2.0),
    ("-Y right",     DEFAULT_X,       DEFAULT_Y-0.08,  DEFAULT_Z,       3.0),
    ("home",         DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       2.0),
    ("+Z up",        DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z+0.08,  3.0),
    ("home",         DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       2.0),
    ("-Z down",      DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z-0.08,  3.0),
    ("home",         DEFAULT_X,       DEFAULT_Y,       DEFAULT_Z,       3.0),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame(x, y, z, qx=HOME_QX, qy=HOME_QY, qz=HOME_QZ, qw=HOME_QW) -> str:
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
            print(f'{YELLOW}Waiting for ik_teleop_node... ({e}){RESET}')
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
    print(f'{BOLD}{CYAN}=== Demo: axis sweep ==={RESET}')
    print('Stepping through preset waypoints. Press Ctrl-C to stop.\n')
    sock = _connect(host, port)
    try:
        for label, x, y, z, hold in DEMO_WAYPOINTS:
            frame = _make_frame(x, y, z)
            print(f'{BOLD}→ {label:<14}{RESET}  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}',
                  flush=True)
            deadline = time.monotonic() + hold
            while time.monotonic() < deadline:
                if not _send(sock, frame):
                    print(f'{YELLOW}Connection lost, reconnecting...{RESET}')
                    sock = _connect(host, port)
                time.sleep(1 / 30.0)
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Interrupted.{RESET}')
    finally:
        sock.close()


def run_figure8(host: str, port: int,
                amp_y: float = 0.06, amp_z: float = 0.04,
                quat: tuple = (HOME_QX, HOME_QY, HOME_QZ, HOME_QW)) -> None:
    """Lissajous 1:2 figure-8 in the Y-Z plane. X and orientation stay fixed."""
    qx, qy, qz, qw = quat
    print(f'{BOLD}{CYAN}=== Figure-8 (Y-Z plane) ==={RESET}')
    print(f'amp_y={amp_y:.3f} m  amp_z={amp_z:.3f} m  '
          f'centre=({DEFAULT_X}, {DEFAULT_Y}, {DEFAULT_Z})')
    print(f'orientation locked: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})')
    print('Press Ctrl-C to stop.\n')
    sock = _connect(host, port)
    omega = 0.5   # rad/s — one full loop ~12 s
    t0 = time.monotonic()
    try:
        while True:
            t = (time.monotonic() - t0) * omega
            y = DEFAULT_Y + amp_y * math.sin(t)
            z = DEFAULT_Z + amp_z * math.sin(2 * t)
            if not _send(sock, _make_frame(DEFAULT_X, y, z, qx, qy, qz, qw)):
                print(f'{YELLOW}Connection lost, reconnecting...{RESET}')
                sock = _connect(host, port)
            print(f'\r  y={y:+.3f}  z={z:+.3f}', end='', flush=True)
            time.sleep(1 / 30.0)
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Stopped.{RESET}')
    finally:
        sock.close()


def run_explore(host: str, port: int, hold: float = 4.0) -> None:
    """Send random reachable positions, holding each for `hold` seconds.

    Workspace bounds derived from URDF joint limits and arm geometry:
      x: [0.20, 0.65]   (reach forward/back)
      y: [-0.40, 0.40]  (side reach)
      z: [0.10, 0.65]   (height)
    Orientation stays fixed at the home FK pose throughout.
    """
    X_LO, X_HI = 0.20, 0.65
    Y_LO, Y_HI = -0.40, 0.40
    Z_LO, Z_HI = 0.10, 0.65

    print(f'{BOLD}{CYAN}=== Explore ==={RESET}')
    print(f'x=[{X_LO},{X_HI}]  y=[{Y_LO},{Y_HI}]  z=[{Z_LO},{Z_HI}]')
    print(f'hold={hold:.1f}s per target  orientation locked to home FK pose')
    print('Press Ctrl-C to stop.\n')
    sock = _connect(host, port)
    n = 0
    try:
        while True:
            x = random.uniform(X_LO, X_HI)
            y = random.uniform(Y_LO, Y_HI)
            z = random.uniform(Z_LO, Z_HI)
            frame = _make_frame(x, y, z)
            n += 1
            print(f'{BOLD}[{n:3d}]{RESET}  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}',
                  flush=True)
            deadline = time.monotonic() + hold
            while time.monotonic() < deadline:
                if not _send(sock, frame):
                    print(f'{YELLOW}Connection lost, reconnecting...{RESET}')
                    sock = _connect(host, port)
                time.sleep(1 / 30.0)
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Stopped after {n} targets.{RESET}')
    finally:
        sock.close()


def run_explore_edge(host: str, port: int,
                     step: float = 0.02,
                     steps_per_ray: int = 30,
                     hold_per_step: float = 1.5) -> None:
    """Probe the workspace boundary by marching outward along random rays.

    Algorithm:
      1. Pick a random 3D unit direction from the current position.
      2. March in steps of `step` metres, holding each position for
         `hold_per_step` seconds so the arm has time to track.
      3. After `steps_per_ray` steps (or when we want a new direction),
         return to home and pick a new random direction.

    Because each step is small, the IK always has a nearby seed from the
    previous step and converges reliably.  The arm naturally stalls at the
    true workspace boundary rather than jumping to unreachable positions.
    """
    import numpy as np

    print(f'{BOLD}{CYAN}=== Explore edge (line search) ==={RESET}')
    print(f'step={step*100:.0f} cm  {steps_per_ray} steps/ray  '
          f'{hold_per_step:.1f}s/step')
    print('Press Ctrl-C to stop.\n')

    sock = _connect(host, port)
    ray = 0

    def _march(x0, y0, z0, dx, dy, dz):
        """March from (x0,y0,z0) along unit direction (dx,dy,dz)."""
        for s in range(1, steps_per_ray + 1):
            x = x0 + dx * step * s
            y = y0 + dy * step * s
            z = z0 + dz * step * s
            frame = _make_frame(x, y, z)
            print(f'  step {s:2d}  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}',
                  flush=True)
            deadline = time.monotonic() + hold_per_step
            while time.monotonic() < deadline:
                if not _send(sock, frame):
                    raise OSError('connection lost')
                time.sleep(1 / 30.0)

    def _go_home():
        frame = _make_frame(DEFAULT_X, DEFAULT_Y, DEFAULT_Z)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            _send(sock, frame)
            time.sleep(1 / 30.0)

    try:
        while True:
            ray += 1
            # Random unit direction, biased to avoid straight down (z=-1)
            theta = random.uniform(0, 2 * math.pi)   # azimuth
            phi   = random.uniform(0.1, math.pi)      # polar (avoid singularity at top/bottom)
            dx = math.sin(phi) * math.cos(theta)
            dy = math.sin(phi) * math.sin(theta)
            dz = math.cos(phi)
            print(f'\n{BOLD}Ray {ray}{RESET}  '
                  f'dir=({dx:+.2f},{dy:+.2f},{dz:+.2f})', flush=True)
            try:
                _march(DEFAULT_X, DEFAULT_Y, DEFAULT_Z, dx, dy, dz)
            except OSError:
                print(f'{YELLOW}Connection lost, reconnecting...{RESET}')
                sock = _connect(host, port)
                continue
            print(f'  → returning home', flush=True)
            _go_home()
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Stopped after {ray} rays.{RESET}')
    finally:
        sock.close()


def run_hold(host: str, port: int, x: float, y: float, z: float) -> None:
    frame = _make_frame(x, y, z)
    print(f'{BOLD}{CYAN}=== Hold ==={RESET}  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}')
    print('Press Ctrl-C to stop.\n')
    sock = _connect(host, port)
    try:
        while True:
            if not _send(sock, frame):
                sock = _connect(host, port)
            time.sleep(1 / 30.0)
    except KeyboardInterrupt:
        print(f'\n{YELLOW}Stopped.{RESET}')
    finally:
        sock.close()


def run_interactive(host: str, port: int) -> None:
    print(f'{BOLD}{CYAN}=== Interactive ==={RESET}')
    print(f'  {BOLD}x y z{RESET}            — move to position (home orientation)')
    print(f'  {BOLD}x y z qx qy qz qw{RESET} — full 6D pose')
    print(f'  {BOLD}home{RESET}              — go to ({DEFAULT_X}, {DEFAULT_Y}, {DEFAULT_Z})')
    print(f'  {BOLD}stop{RESET} / {BOLD}Ctrl-C{RESET}    — quit\n')

    sock = _connect(host, port)
    current_frame = _make_frame(DEFAULT_X, DEFAULT_Y, DEFAULT_Z)
    stop_event = threading.Event()

    def _sender():
        while not stop_event.is_set():
            if current_frame:
                _send(sock, current_frame)
            time.sleep(1 / 30.0)

    threading.Thread(target=_sender, daemon=True).start()

    try:
        while True:
            try:
                raw = input(f'{CYAN}pose>{RESET} ').strip()
            except EOFError:
                break
            if not raw:
                continue
            if raw.lower() in ('stop', 'quit', 'q'):
                break
            if raw.lower() == 'home':
                raw = f'{DEFAULT_X} {DEFAULT_Y} {DEFAULT_Z}'

            parts = raw.split()
            try:
                if len(parts) == 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    current_frame = _make_frame(x, y, z)
                elif len(parts) == 7:
                    vals = [float(p) for p in parts]
                    current_frame = _make_frame(*vals[:3], *vals[3:])
                else:
                    print(f'{RED}Need 3 or 7 numbers.{RESET}')
                    continue
            except ValueError:
                print(f'{RED}Invalid number.{RESET}')
                continue
            print(f'  → {current_frame.strip()}', flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        sock.close()
        print(f'\n{YELLOW}Quit.{RESET}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Mock arm_vision TCP sender.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Modes: (none)=interactive  demo  figure8  hold X Y Z')
    parser.add_argument('mode', nargs='?', default='interactive',
                        choices=['interactive', 'demo', 'figure8', 'hold',
                                 'explore', 'explore-edge'],
                        help='Run mode (default: interactive)')
    parser.add_argument('hold_xyz', nargs='*', type=float, metavar='COORD',
                        help='X Y Z for hold mode')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--amp-y', type=float, default=0.06, metavar='M')
    parser.add_argument('--amp-z', type=float, default=0.04, metavar='M')
    parser.add_argument('--hold-time', type=float, default=4.0, metavar='S',
                        help='explore/explore-edge: seconds per target (default: 4.0)')
    parser.add_argument('--step', type=float, default=0.02, metavar='M',
                        help='explore-edge: step size in metres per march step (default: 0.02)')
    parser.add_argument('--steps-per-ray', type=int, default=30, metavar='N',
                        help='explore-edge: steps before picking a new direction (default: 30)')
    parser.add_argument('--hold-per-step', type=float, default=1.5, metavar='S',
                        help='explore-edge: seconds to hold each step (default: 1.5)')
    parser.add_argument('--quat', nargs=4, type=float,
                        metavar=('QX', 'QY', 'QZ', 'QW'),
                        default=[HOME_QX, HOME_QY, HOME_QZ, HOME_QW])
    args = parser.parse_args()

    print(f'\n{BOLD}Test Pose Sender{RESET}  {args.host}:{args.port}\n')

    if args.mode == 'demo':
        run_demo(args.host, args.port)
    elif args.mode == 'explore':
        run_explore(args.host, args.port, args.hold_time)
    elif args.mode == 'explore-edge':
        run_explore_edge(args.host, args.port,
                         args.step, args.steps_per_ray, args.hold_per_step)
    elif args.mode == 'figure8':
        run_figure8(args.host, args.port, args.amp_y, args.amp_z, tuple(args.quat))
    elif args.mode == 'hold':
        if len(args.hold_xyz) != 3:
            parser.error('hold mode requires exactly 3 coordinates: X Y Z')
        run_hold(args.host, args.port, *args.hold_xyz)
    else:
        run_interactive(args.host, args.port)


if __name__ == '__main__':
    main()
