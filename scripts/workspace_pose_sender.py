#!/usr/bin/env python3
"""
Deterministic TCP pose sender for sweeping the URDF-derived task space.

This script builds a reachable workspace shell directly from URDF FK samples,
then plays a repeatable out-and-back probe on each selected ray:

  home -> reachable shell pose -> slightly outside shell -> shell -> home

Each transmitted frame is logged together with the latest feedback from
ik_teleop_node so failed regions can be correlated with:
  - IK result type (6D / POS_ONLY / SNAP / FAIL)
  - winning seed and pool
  - best failed 6D residuals / sigma_min
  - FK position / quaternion returned by the node

Examples:
  python3 scripts/workspace_pose_sender.py
  python3 scripts/workspace_pose_sender.py --max-shell-points 20 --outside-margin-mm 12
  python3 scripts/workspace_pose_sender.py --host 172.16.51.47 --port 9999 --loop
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _rpy_xyz_to_mat4(rpy, xyz):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def _axis_angle_mat4(axis, angle):
    ax, ay, az = axis
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
        [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
        [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
    ])
    return T


def _mat3_to_quat(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12))
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12))
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12))
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-9:
        quat /= norm
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm <= 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    q = np.array(q, dtype=float) / norm
    if q[3] < 0.0:
        q *= -1.0
    return q


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _quat_normalize((1.0 - t) * q0 + t * q1)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return _quat_normalize(s0 * q0 + s1 * q1)


def _load_chain(urdf_path: Path, base: str = 'base_link', tip: str = 'End_Effector'):
    ros_site = '/opt/ros/humble/lib/python3.10/site-packages'
    if ros_site not in sys.path:
        sys.path.insert(0, ros_site)
    from urdf_parser_py import urdf as urdf_parser

    model = urdf_parser.URDF.from_xml_string(urdf_path.read_bytes())
    joint_from_child = {j.child: j for j in model.joints}
    path = []
    current = tip
    while current != base:
        joint = joint_from_child.get(current)
        if joint is None:
            raise RuntimeError(f'No URDF path from {base!r} to {tip!r}')
        path.append(joint)
        current = joint.parent
    path.reverse()

    chain = []
    lowers = []
    uppers = []
    for j in path:
        xyz = list(j.origin.xyz) if j.origin and j.origin.xyz else [0.0, 0.0, 0.0]
        rpy = list(j.origin.rpy) if j.origin and j.origin.rpy else [0.0, 0.0, 0.0]
        axis = np.array(list(j.axis) if j.axis else [1.0, 0.0, 0.0], dtype=float)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-10:
            axis /= axis_norm
        lower = float(j.limit.lower) if j.limit and j.limit.lower is not None else -math.pi
        upper = float(j.limit.upper) if j.limit and j.limit.upper is not None else math.pi
        chain.append({
            'type': j.type,
            'T_fixed': _rpy_xyz_to_mat4(rpy, xyz),
            'axis': axis,
            'lower': lower,
            'upper': upper,
        })
        if j.type in ('revolute', 'continuous'):
            lowers.append(lower)
            uppers.append(upper)
    return chain, np.array(lowers, dtype=float), np.array(uppers, dtype=float)


def _fk_pose(chain, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.eye(4)
    qi = 0
    for seg in chain:
        T = T @ seg['T_fixed']
        if seg['type'] in ('revolute', 'continuous'):
            T = T @ _axis_angle_mat4(seg['axis'], q[qi])
            qi += 1
    return T[:3, 3].copy(), _mat3_to_quat(T[:3, :3].copy())


def _halton(index: int, base: int) -> float:
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def _halton_point(index: int, dims: int) -> np.ndarray:
    bases = [2, 3, 5, 7, 11, 13, 17, 19]
    if dims > len(bases):
        raise ValueError(f'Halton only configured for up to {len(bases)} dims')
    return np.array([_halton(index, base) for base in bases[:dims]], dtype=float)


@dataclass(frozen=True)
class WorkspacePose:
    label: str
    q: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    radius: float
    azimuth: float
    elevation: float


def _shell_bin_key(vec: np.ndarray, az_bins: int, el_bins: int) -> tuple[int, int]:
    radius = float(np.linalg.norm(vec))
    if radius < 1e-9:
        return (0, 0)
    az = math.atan2(vec[1], vec[0])
    el = math.asin(max(-1.0, min(1.0, vec[2] / radius)))
    az_idx = min(az_bins - 1, max(0, int((az + math.pi) / (2.0 * math.pi) * az_bins)))
    el_idx = min(el_bins - 1, max(0, int((el + math.pi / 2.0) / math.pi * el_bins)))
    return az_idx, el_idx


def _generate_workspace_shell(
    chain,
    lowers: np.ndarray,
    uppers: np.ndarray,
    sample_count: int,
    az_bins: int,
    el_bins: int,
    max_shell_points: int,
) -> tuple[WorkspacePose, list[WorkspacePose]]:
    home_q = np.zeros_like(lowers)
    home_pos, home_quat = _fk_pose(chain, home_q)
    home = WorkspacePose(
        label='home',
        q=home_q,
        pos=home_pos,
        quat=home_quat,
        radius=0.0,
        azimuth=0.0,
        elevation=0.0,
    )

    bins: dict[tuple[int, int], WorkspacePose] = {}
    for i in range(1, sample_count + 1):
        alpha = _halton_point(i, len(lowers))
        q = lowers + alpha * (uppers - lowers)
        pos, quat = _fk_pose(chain, q)
        vec = pos - home_pos
        radius = float(np.linalg.norm(vec))
        if radius < 0.03:
            continue
        azimuth = math.atan2(vec[1], vec[0])
        elevation = math.asin(max(-1.0, min(1.0, vec[2] / radius)))
        pose = WorkspacePose(
            label=f'sample_{i:04d}',
            q=q,
            pos=pos,
            quat=quat,
            radius=radius,
            azimuth=azimuth,
            elevation=elevation,
        )
        key = _shell_bin_key(vec, az_bins, el_bins)
        current = bins.get(key)
        if current is None or pose.radius > current.radius:
            bins[key] = pose

    shell = sorted(bins.values(), key=lambda p: (p.azimuth, p.elevation, p.radius))
    if not shell:
        raise RuntimeError('No workspace shell samples found; check URDF/joint limits')

    if len(shell) > max_shell_points:
        idxs = np.linspace(0, len(shell) - 1, num=max_shell_points, dtype=int)
        shell = [shell[i] for i in idxs]

    return home, shell


class FeedbackReceiver:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._feedback: dict = {}
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def feedback(self) -> dict:
        with self._lock:
            return dict(self._feedback)

    def _recv_loop(self) -> None:
        buf = ''
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(4096).decode('utf-8', errors='replace')
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                time.sleep(0.05)
                continue
            buf += chunk
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    self._feedback = msg


def _send_pose(sock: socket.socket, pos: np.ndarray, quat: np.ndarray) -> None:
    frame = json.dumps({
        'x': float(pos[0]),
        'y': float(pos[1]),
        'z': float(pos[2]),
        'qx': float(quat[0]),
        'qy': float(quat[1]),
        'qz': float(quat[2]),
        'qw': float(quat[3]),
    }) + '\n'
    sock.sendall(frame.encode('utf-8'))


def _connect(host: str, port: int) -> socket.socket:
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            sock.connect((host, port))
            sock.settimeout(0.1)
            print(f'[workspace_pose_sender] Connected to {host}:{port}')
            return sock
        except OSError as exc:
            print(f'[workspace_pose_sender] Waiting for TCP server: {exc}')
            time.sleep(1.0)


def _ensure_log_writer(log_dir: str) -> tuple[Optional[object], Optional[csv.writer], str]:
    if not log_dir:
        return None, None, ''
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(log_dir, f'workspace_probe_{ts}.csv')
    handle = open(path, 'w', newline='', buffering=1)
    writer = csv.writer(handle)
    writer.writerow([
        'mono_s',
        'phase',
        'target_kind',
        'label',
        'step_idx',
        'step_total',
        'cmd_x',
        'cmd_y',
        'cmd_z',
        'cmd_qx',
        'cmd_qy',
        'cmd_qz',
        'cmd_qw',
        'ik_ok',
        'ik_result',
        'ik_seed',
        'ik_seed_kind',
        'ik_pool',
        'ik_fail_cause',
        'fk_x',
        'fk_y',
        'fk_z',
        'fk_qx',
        'fk_qy',
        'fk_qz',
        'fk_qw',
        'ik_continuity_cost',
        'ik_joint_state_cost',
        'ik_posture_cost',
        'ik_max_jump_rad',
        'ik6d_time_budget_hit',
        'ik6d_best_failed_seed',
        'ik6d_best_failed_seed_kind',
        'ik6d_best_failed_reason',
        'ik6d_best_failed_class',
        'ik6d_best_failed_pos_err_mm',
        'ik6d_best_failed_ori_err_deg',
        'ik6d_best_failed_sigma_min',
        'ik6d_best_failed_singular',
    ])
    return handle, writer, path


def _log_row(
    writer: Optional[csv.writer],
    *,
    phase: str,
    target_kind: str,
    label: str,
    step_idx: int,
    step_total: int,
    pos: np.ndarray,
    quat: np.ndarray,
    feedback: dict,
) -> None:
    if writer is None:
        return
    writer.writerow([
        f'{time.monotonic():.6f}',
        phase,
        target_kind,
        label,
        step_idx,
        step_total,
        f'{pos[0]:.6f}',
        f'{pos[1]:.6f}',
        f'{pos[2]:.6f}',
        f'{quat[0]:.6f}',
        f'{quat[1]:.6f}',
        f'{quat[2]:.6f}',
        f'{quat[3]:.6f}',
        feedback.get('ik_ok', ''),
        feedback.get('ik_result', ''),
        feedback.get('ik_seed', ''),
        feedback.get('ik_seed_kind', ''),
        feedback.get('ik_pool', ''),
        feedback.get('ik_fail_cause', ''),
        feedback.get('fk_x', ''),
        feedback.get('fk_y', ''),
        feedback.get('fk_z', ''),
        feedback.get('fk_qx', ''),
        feedback.get('fk_qy', ''),
        feedback.get('fk_qz', ''),
        feedback.get('fk_qw', ''),
        feedback.get('ik_continuity_cost', ''),
        feedback.get('ik_joint_state_cost', ''),
        feedback.get('ik_posture_cost', ''),
        feedback.get('ik_max_jump_rad', ''),
        feedback.get('ik6d_time_budget_hit', ''),
        feedback.get('ik6d_best_failed_seed', ''),
        feedback.get('ik6d_best_failed_seed_kind', ''),
        feedback.get('ik6d_best_failed_reason', ''),
        feedback.get('ik6d_best_failed_class', ''),
        feedback.get('ik6d_best_failed_pos_err_mm', ''),
        feedback.get('ik6d_best_failed_ori_err_deg', ''),
        feedback.get('ik6d_best_failed_sigma_min', ''),
        feedback.get('ik6d_best_failed_singular', ''),
    ])


def _play_segment(
    sock: socket.socket,
    feedback_rx: FeedbackReceiver,
    writer: Optional[csv.writer],
    *,
    phase: str,
    target_kind: str,
    label: str,
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    end_pos: np.ndarray,
    end_quat: np.ndarray,
    duration_s: float,
    hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = max(1, int(round(duration_s * hz)))
    for step in range(1, steps + 1):
        t = step / steps
        pos = (1.0 - t) * start_pos + t * end_pos
        quat = _quat_slerp(start_quat, end_quat, t)
        _send_pose(sock, pos, quat)
        _log_row(
            writer,
            phase=phase,
            target_kind=target_kind,
            label=label,
            step_idx=step,
            step_total=steps,
            pos=pos,
            quat=quat,
            feedback=feedback_rx.feedback,
        )
        time.sleep(1.0 / hz)
    return end_pos.copy(), end_quat.copy()


def _hold_pose(
    sock: socket.socket,
    feedback_rx: FeedbackReceiver,
    writer: Optional[csv.writer],
    *,
    phase: str,
    target_kind: str,
    label: str,
    pos: np.ndarray,
    quat: np.ndarray,
    hold_s: float,
    hz: float,
) -> None:
    steps = max(1, int(round(hold_s * hz)))
    for step in range(1, steps + 1):
        _send_pose(sock, pos, quat)
        _log_row(
            writer,
            phase=phase,
            target_kind=target_kind,
            label=label,
            step_idx=step,
            step_total=steps,
            pos=pos,
            quat=quat,
            feedback=feedback_rx.feedback,
        )
        time.sleep(1.0 / hz)


def _outside_probe(home: WorkspacePose, shell_pose: WorkspacePose, outside_margin_m: float) -> np.ndarray:
    vec = shell_pose.pos - home.pos
    radius = float(np.linalg.norm(vec))
    if radius < 1e-9:
        return shell_pose.pos.copy()
    return shell_pose.pos + vec / radius * outside_margin_m


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Deterministic URDF-workspace probe sender for ik_teleop_node.',
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--urdf', default='robotic_arm_v4_urdf/urdf/robotic_arm_v4_urdf.urdf')
    parser.add_argument('--hz', type=float, default=30.0)
    parser.add_argument('--sample-count', type=int, default=768,
                        help='Halton FK samples used to estimate the reachable shell')
    parser.add_argument('--az-bins', type=int, default=10)
    parser.add_argument('--el-bins', type=int, default=6)
    parser.add_argument('--max-shell-points', type=int, default=16,
                        help='Maximum number of shell rays to probe per pass')
    parser.add_argument('--travel-time', type=float, default=1.0,
                        help='Seconds for interpolated travel between phases')
    parser.add_argument('--inside-hold', type=float, default=0.5,
                        help='Seconds to hold each reachable shell pose')
    parser.add_argument('--outside-hold', type=float, default=0.5,
                        help='Seconds to hold each slightly-outside probe')
    parser.add_argument('--home-hold', type=float, default=0.25,
                        help='Seconds to hold home between rays')
    parser.add_argument('--outside-margin-mm', type=float, default=15.0,
                        help='How far beyond the shell to probe along each ray')
    parser.add_argument('--loop', action='store_true',
                        help='Repeat the full probe cycle until interrupted')
    parser.add_argument('--log-dir', default='/tmp',
                        help='CSV log directory (default: /tmp, empty string disables)')
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    urdf_path = Path(args.urdf).resolve()
    chain, lowers, uppers = _load_chain(urdf_path)
    home, shell = _generate_workspace_shell(
        chain,
        lowers,
        uppers,
        sample_count=args.sample_count,
        az_bins=args.az_bins,
        el_bins=args.el_bins,
        max_shell_points=args.max_shell_points,
    )

    print(
        '[workspace_pose_sender] '
        f'home=({home.pos[0]:.3f},{home.pos[1]:.3f},{home.pos[2]:.3f}) '
        f'shell_points={len(shell)} '
        f'outside_margin={args.outside_margin_mm:.1f}mm'
    )
    for i, pose in enumerate(shell, start=1):
        print(
            f'  [{i:02d}] {pose.label} '
            f'pos=({pose.pos[0]:+.3f},{pose.pos[1]:+.3f},{pose.pos[2]:+.3f}) '
            f'r={pose.radius:.3f}m az={math.degrees(pose.azimuth):+.1f}deg '
            f'el={math.degrees(pose.elevation):+.1f}deg'
        )

    sock = _connect(args.host, args.port)
    feedback_rx = FeedbackReceiver(sock)
    feedback_rx.start()
    log_handle, log_writer, log_path = _ensure_log_writer(args.log_dir)
    if log_path:
        print(f'[workspace_pose_sender] log -> {log_path}')

    current_pos = home.pos.copy()
    current_quat = home.quat.copy()
    outside_margin_m = args.outside_margin_mm / 1000.0

    try:
        while True:
            _hold_pose(
                sock, feedback_rx, log_writer,
                phase='home_settle',
                target_kind='reachable',
                label='home',
                pos=home.pos,
                quat=home.quat,
                hold_s=max(args.home_hold, 1.0),
                hz=args.hz,
            )
            current_pos = home.pos.copy()
            current_quat = home.quat.copy()

            for index, pose in enumerate(shell, start=1):
                outside_pos = _outside_probe(home, pose, outside_margin_m)
                label = f'ray_{index:02d}'
                print(
                    f'[workspace_pose_sender] {label} '
                    f'reachable=({pose.pos[0]:+.3f},{pose.pos[1]:+.3f},{pose.pos[2]:+.3f}) '
                    f'outside=({outside_pos[0]:+.3f},{outside_pos[1]:+.3f},{outside_pos[2]:+.3f})'
                )
                current_pos, current_quat = _play_segment(
                    sock, feedback_rx, log_writer,
                    phase='to_shell',
                    target_kind='reachable',
                    label=label,
                    start_pos=current_pos,
                    start_quat=current_quat,
                    end_pos=pose.pos,
                    end_quat=pose.quat,
                    duration_s=args.travel_time,
                    hz=args.hz,
                )
                _hold_pose(
                    sock, feedback_rx, log_writer,
                    phase='hold_shell',
                    target_kind='reachable',
                    label=label,
                    pos=pose.pos,
                    quat=pose.quat,
                    hold_s=args.inside_hold,
                    hz=args.hz,
                )
                current_pos, current_quat = _play_segment(
                    sock, feedback_rx, log_writer,
                    phase='to_outside',
                    target_kind='outside',
                    label=label,
                    start_pos=current_pos,
                    start_quat=current_quat,
                    end_pos=outside_pos,
                    end_quat=pose.quat,
                    duration_s=args.travel_time * 0.6,
                    hz=args.hz,
                )
                _hold_pose(
                    sock, feedback_rx, log_writer,
                    phase='hold_outside',
                    target_kind='outside',
                    label=label,
                    pos=outside_pos,
                    quat=pose.quat,
                    hold_s=args.outside_hold,
                    hz=args.hz,
                )
                current_pos, current_quat = _play_segment(
                    sock, feedback_rx, log_writer,
                    phase='return_shell',
                    target_kind='reachable',
                    label=label,
                    start_pos=current_pos,
                    start_quat=current_quat,
                    end_pos=pose.pos,
                    end_quat=pose.quat,
                    duration_s=args.travel_time * 0.6,
                    hz=args.hz,
                )
                _hold_pose(
                    sock, feedback_rx, log_writer,
                    phase='settle_shell',
                    target_kind='reachable',
                    label=label,
                    pos=pose.pos,
                    quat=pose.quat,
                    hold_s=args.inside_hold * 0.5,
                    hz=args.hz,
                )
                current_pos, current_quat = _play_segment(
                    sock, feedback_rx, log_writer,
                    phase='return_home',
                    target_kind='reachable',
                    label=label,
                    start_pos=current_pos,
                    start_quat=current_quat,
                    end_pos=home.pos,
                    end_quat=home.quat,
                    duration_s=args.travel_time,
                    hz=args.hz,
                )
                _hold_pose(
                    sock, feedback_rx, log_writer,
                    phase='hold_home',
                    target_kind='reachable',
                    label=label,
                    pos=home.pos,
                    quat=home.quat,
                    hold_s=args.home_hold,
                    hz=args.hz,
                )

            if not args.loop:
                break
    except KeyboardInterrupt:
        print('[workspace_pose_sender] interrupted')
    finally:
        feedback_rx.stop()
        if log_handle is not None:
            log_handle.close()
        try:
            sock.close()
        except OSError:
            pass
        if log_path:
            print(f'[workspace_pose_sender] saved log: {log_path}')


if __name__ == '__main__':
    main()
