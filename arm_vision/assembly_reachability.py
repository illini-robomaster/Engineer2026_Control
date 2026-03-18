#!/usr/bin/env python3
"""
Assembly reachability checker.

Given a starting EE position (post-INSERT), traces the full assembly trajectory
(LIFT → P arc → Q arc) and tests IK feasibility at each step.  Prints a
reachability map showing which Q arc angles have valid IK solutions.

Usage:
    python assembly_reachability.py --x 0.50 --y 0.0 --z 0.40
    python assembly_reachability.py --x 0.50 --y 0.0 --z 0.40 --q-range 120
    python assembly_reachability.py --sweep   # scan a grid of starting positions

Requires: the URDF file and assembly_params.yaml.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ── FK / IK (copied from ik_teleop_node to avoid ROS dependency) ─────────────

def _rpy_xyz_to_mat4(rpy, xyz):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = xyz
    return T


def _axis_angle_mat4(axis, angle):
    ax, ay, az = axis
    c, s, t = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [t*ax*ax + c,    t*ax*ay - s*az, t*ax*az + s*ay],
        [t*ax*ay + s*az, t*ay*ay + c,    t*ay*az - s*ax],
        [t*ax*az - s*ay, t*ay*az + s*ax, t*az*az + c   ],
    ])
    return T


def _quat_to_mat3(q):
    qx, qy, qz, qw = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)],
    ])


def _build_chain_data(urdf_path, base='base_link', tip='End_Effector'):
    from urdf_parser_py import urdf as urdf_parser
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()
    model = urdf_parser.URDF.from_xml_string(robot_desc.encode('utf-8'))
    joint_from_child = {j.child: j for j in model.joints}
    path = []
    current = tip
    while current != base:
        joint = joint_from_child.get(current)
        if joint is None:
            raise RuntimeError(f'No path from {base} to {tip}')
        path.append(joint)
        current = joint.parent
    path.reverse()
    chain_data = []
    for j in path:
        xyz = list(j.origin.xyz) if j.origin and j.origin.xyz else [0., 0., 0.]
        rpy = list(j.origin.rpy) if j.origin and j.origin.rpy else [0., 0., 0.]
        raw_axis = list(j.axis) if j.axis else [1., 0., 0.]
        axis = np.array(raw_axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n > 1e-10:
            axis /= n
        lower = float(j.limit.lower) if j.limit and j.limit.lower is not None else -np.pi
        upper = float(j.limit.upper) if j.limit and j.limit.upper is not None else  np.pi
        chain_data.append({
            'name': j.name, 'type': j.type,
            'T_fixed': _rpy_xyz_to_mat4(rpy, xyz),
            'axis': axis, 'lower': lower, 'upper': upper,
        })
    return chain_data


def _fk_and_jac(chain_data, q):
    T = np.eye(4)
    origins, axes = [], []
    qi = 0
    for seg in chain_data:
        T = T @ seg['T_fixed']
        if seg['type'] in ('revolute', 'continuous'):
            origins.append(T[:3, 3].copy())
            axes.append(T[:3, :3] @ seg['axis'])
            T = T @ _axis_angle_mat4(seg['axis'], q[qi])
            qi += 1
    p_ee = T[:3, 3]
    R_ee = T[:3, :3].copy()
    n = len(q)
    J = np.zeros((6, n))
    for i in range(n):
        J[:3, i] = np.cross(axes[i], p_ee - origins[i])
        J[3:, i] = axes[i]
    return p_ee, R_ee, J


def _joint_limits(chain_data):
    lowers = np.array([s['lower'] for s in chain_data if s['type'] in ('revolute', 'continuous')])
    uppers = np.array([s['upper'] for s in chain_data if s['type'] in ('revolute', 'continuous')])
    return lowers, uppers


def solve_ik_6d(chain_data, target_pos, target_quat, seed, max_iters=500):
    """6D DLS IK. Returns joint list or None."""
    R_tgt = _quat_to_mat3(target_quat)
    q = np.array(seed, dtype=float)
    lowers, uppers = _joint_limits(chain_data)
    lam = 1e-3
    prev_err = float('inf')
    for _ in range(max_iters):
        p_ee, R_ee, J = _fk_and_jac(chain_data, q)
        err_pos = target_pos - p_ee
        R_err = R_tgt @ R_ee.T
        err_rot = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        pos_norm = float(np.linalg.norm(err_pos))
        ori_norm = float(np.linalg.norm(err_rot))
        if pos_norm < 1e-3 and ori_norm < 0.035:
            return list(q), pos_norm, ori_norm
        err_6 = np.concatenate([err_pos, err_rot])
        err_6d_norm = float(np.linalg.norm(err_6))
        if err_6d_norm >= prev_err:
            lam = min(lam * 10.0, 1e-1)
        else:
            lam = max(lam * 0.5, 1e-7)
        prev_err = err_6d_norm
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err_6)
        dq_norm = float(np.linalg.norm(dq))
        if dq_norm < 1e-9:
            break
        if dq_norm > 0.20:
            dq *= 0.20 / dq_norm
        q = np.clip(q + dq, lowers, uppers)
    # Near-miss
    p_f, R_f, _ = _fk_and_jac(chain_data, q)
    R_ef = R_tgt @ R_f.T
    e_rf = 0.5 * np.array([R_ef[2,1]-R_ef[1,2], R_ef[0,2]-R_ef[2,0], R_ef[1,0]-R_ef[0,1]])
    fp = float(np.linalg.norm(target_pos - p_f))
    fo = float(np.linalg.norm(e_rf))
    if fp < 2e-3 and fo < 0.052:
        return list(q), fp, fo
    return None, fp, fo


def try_ik_multi_seed(chain_data, target_pos, target_quat, prev_solution=None,
                      n_random=15, max_iters=300):
    """Try multiple seeds. Returns (solution, pos_err, ori_err) or (None, ...)."""
    lowers, uppers = _joint_limits(chain_data)
    seeds = []
    if prev_solution is not None:
        seeds.append(prev_solution)
    seeds.append([0.0] * 6)
    for _ in range(n_random):
        seeds.append(list(np.random.uniform(lowers, uppers)))
    for s in seeds:
        result, pe, oe = solve_ik_6d(chain_data, target_pos, target_quat, s, max_iters)
        if result is not None:
            return result, pe, oe
    return None, pe, oe


# ── Assembly trajectory computation ──────────────────────────────────────────

def compute_trajectory(insert_pos, asm_cfg):
    """Compute all poses along the assembly trajectory from the INSERT position.

    Returns list of (label, position, quaternion) tuples.
    """
    waypoints = []

    # Assembly init orientation: EE x → +Z
    R_init = Rotation.from_euler('y', -asm_cfg.get('init_pitch_deg', 90.0), degrees=True)

    # ── LIFT: straight +Z ──────────────────────────────────────────────────
    lift_mm = asm_cfg['lift_distance_mm']
    lift_pos = insert_pos.copy()
    lift_pos[2] += lift_mm / 1000.0
    waypoints.append(('LIFT_END', lift_pos.copy(), R_init.as_quat()))

    # ── P ARC: pitch 90° ───────────────────────────────────────────────────
    p_radius = asm_cfg['p_arc_radius_mm'] / 1000.0
    p_angle = asm_cfg.get('p_arc_angle_deg', 90.0)
    p_center_ee = np.array(asm_cfg['p_arc_center_dir_ee'], dtype=float)
    p_axis_ee = np.array(asm_cfg['p_arc_axis_ee'], dtype=float)

    p_center_world = R_init.apply(p_center_ee)
    p_axis_world = R_init.apply(p_axis_ee)
    p_pivot = lift_pos + p_center_world * p_radius
    p_offset = lift_pos - p_pivot

    n_p_steps = max(int(abs(p_angle) / 5), 1)
    R_curr = R_init
    pos_curr = lift_pos.copy()
    for i in range(1, n_p_steps + 1):
        frac = i / n_p_steps
        theta = np.radians(p_angle * frac)
        R_arc = Rotation.from_rotvec(p_axis_world * theta)
        pos_curr = p_pivot + R_arc.apply(p_offset)
        R_curr = R_arc * R_init
        if i == n_p_steps:
            waypoints.append(('P_ARC_END', pos_curr.copy(), R_curr.as_quat()))

    # ── Q ARC: sweep range ─────────────────────────────────────────────────
    q_radius = asm_cfg['q_arc_radius_mm'] / 1000.0
    q_center_ee = np.array(asm_cfg['q_arc_center_dir_ee'], dtype=float)
    q_axis_ee = np.array(asm_cfg['q_arc_axis_ee'], dtype=float)

    R_post_p = R_curr
    pos_post_p = pos_curr.copy()

    q_center_world = R_post_p.apply(q_center_ee)
    q_axis_world = R_post_p.apply(q_axis_ee)
    q_pivot = pos_post_p + q_center_world * q_radius
    q_offset = pos_post_p - q_pivot

    # Sweep from -max to +max in 5° steps
    q_max = asm_cfg.get('q_sweep_range_deg', 120.0)
    q_angles = np.arange(-q_max, q_max + 1, 5.0)
    for q_deg in q_angles:
        theta = np.radians(q_deg)
        R_q = Rotation.from_rotvec(q_axis_world * theta)
        q_pos = q_pivot + R_q.apply(q_offset)
        q_rot = R_q * R_post_p
        waypoints.append((f'Q={q_deg:+.0f}', q_pos.copy(), q_rot.as_quat()))

    return waypoints


# ── Main ─────────────────────────────────────────────────────────────────────

def check_reachability(chain_data, insert_pos, asm_cfg, verbose=False):
    """Check IK for all waypoints. Returns (results, summary_str)."""
    waypoints = compute_trajectory(insert_pos, asm_cfg)
    results = []
    prev_sol = None

    for label, pos, quat in waypoints:
        sol, pe, oe = try_ik_multi_seed(chain_data, pos, quat,
                                         prev_solution=prev_sol,
                                         n_random=20, max_iters=500)
        ok = sol is not None
        if ok:
            prev_sol = sol
        results.append({
            'label': label, 'pos': pos, 'quat': quat,
            'ok': ok, 'pos_err': pe, 'ori_err': oe,
            'joints': sol,
        })
        if verbose:
            status = 'OK' if ok else 'FAIL'
            print(f'  {label:>12s}  pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}]  '
                  f'{status}  pe={pe*1000:.1f}mm  oe={np.degrees(oe):.1f}deg')

    # Summary: find Q arc range
    q_results = [r for r in results if r['label'].startswith('Q=')]
    ok_angles = [float(r['label'].split('=')[1]) for r in q_results if r['ok']]
    fail_angles = [float(r['label'].split('=')[1]) for r in q_results if not r['ok']]

    if ok_angles:
        q_min, q_max = min(ok_angles), max(ok_angles)
        summary = f'Q arc reachable: [{q_min:+.0f}, {q_max:+.0f}] deg'
    else:
        summary = 'Q arc: NO reachable angles!'

    if fail_angles:
        summary += f'  |  FAIL at: {", ".join(f"{a:+.0f}" for a in sorted(fail_angles))} deg'

    return results, summary


def main():
    p = argparse.ArgumentParser(description='Assembly reachability checker')
    p.add_argument('--urdf', default='robotic_arm_v4_urdf/urdf/robotic_arm_v4_urdf.urdf')
    p.add_argument('--config', default='arm_vision/config/assembly_params.yaml')
    p.add_argument('--x', type=float, default=0.50, help='Insert position X (m)')
    p.add_argument('--y', type=float, default=0.00, help='Insert position Y (m)')
    p.add_argument('--z', type=float, default=0.35, help='Insert position Z (m)')
    p.add_argument('--q-range', type=float, default=120.0,
                   help='Q arc sweep range ± degrees (default: 120)')
    p.add_argument('--sweep', action='store_true',
                   help='Scan a grid of starting positions')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args()

    chain_data = _build_chain_data(args.urdf)
    print(f'Loaded URDF: {len(chain_data)} joints')

    with open(args.config, 'r') as f:
        asm_cfg = yaml.safe_load(f)['assembly']
    asm_cfg['q_sweep_range_deg'] = args.q_range

    if args.sweep:
        # Scan grid of starting positions
        print(f'\nSweeping insert positions (Q arc ±{args.q_range:.0f}°):')
        print(f'{"X":>6s} {"Y":>6s} {"Z":>6s}  {"Q arc range":>20s}  {"Details"}')
        print('-' * 70)
        for x in np.arange(0.30, 0.65, 0.05):
            for y in np.arange(-0.15, 0.20, 0.05):
                for z in np.arange(0.15, 0.50, 0.05):
                    pos = np.array([x, y, z])
                    results, summary = check_reachability(
                        chain_data, pos, asm_cfg, verbose=False)
                    # Only print if at least LIFT_END is reachable
                    lift_ok = any(r['ok'] for r in results if r['label'] == 'LIFT_END')
                    if lift_ok:
                        print(f'{x:6.2f} {y:6.2f} {z:6.2f}  {summary}')
    else:
        insert_pos = np.array([args.x, args.y, args.z])
        print(f'\nInsert position: [{args.x}, {args.y}, {args.z}]')
        print(f'Q arc sweep: ±{args.q_range}°\n')
        results, summary = check_reachability(
            chain_data, insert_pos, asm_cfg, verbose=True)
        print(f'\n{summary}')

        # Visual bar chart of Q arc
        q_results = [r for r in results if r['label'].startswith('Q=')]
        if q_results:
            print(f'\nQ arc reachability map:')
            for r in q_results:
                angle = float(r['label'].split('=')[1])
                bar = 'O' if r['ok'] else 'X'
                print(f'  {angle:+6.0f}° {bar}', end='')
                if not r['ok']:
                    print(f'  (pe={r["pos_err"]*1000:.1f}mm oe={np.degrees(r["ori_err"]):.1f}°)', end='')
                print()


if __name__ == '__main__':
    main()
