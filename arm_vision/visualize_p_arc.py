"""Standalone P-arc visualizer — no ROS required.

Reproduces the exact ArcInterpolator math from assembly_fsm.py and renders
the arc in 3D with matplotlib.  Edit the parameter block at the top to tweak
the geometry until it matches real-world motion.

Run:
    python arm_vision/visualize_p_arc.py
"""

import sys
# Fix: the system mpl_toolkits shadows the pip-installed one, breaking 3D.
# Reorder mpl_toolkits.__path__ so the pip version is found first.
import mpl_toolkits as _mpl_tk
_pip_tk = [p for p in _mpl_tk.__path__ if 'local' in p]
_sys_tk = [p for p in _mpl_tk.__path__ if 'local' not in p]
_mpl_tk.__path__ = _pip_tk + _sys_tk

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# ── Assembly params (mirror assembly_params.yaml) ────────────────────────────
# At pitch=90°: EE x → world +Z (up), EE z → world −X (forward)
# Pivot and handle both sit on the EE −x axis (= world −Z, straight down):
#   EE tip
#   │  ← 60 mm (P_ARC_RADIUS_MM) → pivot      (EE arc radius)
#   │  ← 108 mm (HANDLE_OFFSET)   → handle     (grip point)
# handle is 48 mm below pivot  →  handle path radius = 48 mm
P_ARC_RADIUS_MM     = 60.0
P_ARC_ANGLE_DEG     = 90.0
P_ARC_CENTER_DIR_EE = np.array([-1.0, 0.0, 0.0])   # EE −x: pivot straight below EE
P_ARC_AXIS_EE       = np.array([0.0, 1.0, 0.0])    # EE y = world Y (pitch rotation)

HANDLE_OFFSET_EE_MM = np.array([-108.0, 0.0, 0.0])  # mm — grip is 108 mm below EE (world Z)

# Starting EE pose
EE_START_POS_M  = np.array([0.50, 0.00, 0.50])
INIT_PITCH_DEG  = 90.0   # EE x → world +Z
INIT_YAW_DEG    = 0.0    # approach along world +X

N_FRAMES        = 8      # number of EE-frame snapshots shown
AXIS_LEN        = 0.04   # length of EE-frame axis arrows (m)
# ─────────────────────────────────────────────────────────────────────────────


def build_start_rotation() -> Rotation:
    """Mirror the FSM's initial rotation: R_yaw * R_pitch (line 323-325 in assembly_fsm.py)."""
    R_yaw   = Rotation.from_euler('z',  INIT_YAW_DEG,   degrees=True)
    R_pitch = Rotation.from_euler('y', -INIT_PITCH_DEG, degrees=True)
    return R_yaw * R_pitch


def simulate_arc(R_start: Rotation):
    """Reproduce ArcInterpolator.__init__ + tick() for all thetas."""
    # _make_arc in assembly_fsm.py
    center_world = R_start.apply(P_ARC_CENTER_DIR_EE)
    axis_world   = R_start.apply(P_ARC_AXIS_EE)
    radius_m     = P_ARC_RADIUS_MM / 1000.0
    pivot        = EE_START_POS_M + center_world * radius_m
    offset_0     = EE_START_POS_M - pivot   # radius vector at t=0  (ArcInterpolator._offset_0)

    thetas       = np.linspace(0.0, np.radians(P_ARC_ANGLE_DEG), 200)
    ee_pos_list  = []
    ee_rot_list  = []
    for theta in thetas:
        arc_rot = Rotation.from_rotvec(axis_world * theta)   # ArcInterpolator.tick
        pos     = pivot + arc_rot.apply(offset_0)
        rot     = arc_rot * R_start
        ee_pos_list.append(pos)
        ee_rot_list.append(rot)

    handle_offset_m = HANDLE_OFFSET_EE_MM / 1000.0
    handle_pos_list = [
        pos + rot.apply(handle_offset_m)
        for pos, rot in zip(ee_pos_list, ee_rot_list)
    ]

    return pivot, axis_world, offset_0, ee_pos_list, ee_rot_list, handle_pos_list


def draw_frame_triads(ax, ee_pos_list, ee_rot_list, n_frames):
    """Draw RGB xyz triads at n_frames equally-spaced points along the arc."""
    indices   = np.linspace(0, len(ee_pos_list) - 1, n_frames, dtype=int)
    colors    = ['red', 'green', 'blue']   # x, y, z
    for idx in indices:
        pos = ee_pos_list[idx]
        rot = ee_rot_list[idx]
        mat = rot.as_matrix()                # columns are x, y, z in world frame
        for axis_i, color in enumerate(colors):
            tip = pos + mat[:, axis_i] * AXIS_LEN
            ax.plot([pos[0], tip[0]], [pos[1], tip[1]], [pos[2], tip[2]],
                    color=color, linewidth=1.2)


def draw_energy_unit_sticks(ax, ee_pos_list, ee_rot_list, n_frames):
    """Draw EE-tip → handle line at n_frames snapshots."""
    indices         = np.linspace(0, len(ee_pos_list) - 1, n_frames, dtype=int)
    handle_offset_m = HANDLE_OFFSET_EE_MM / 1000.0
    for idx in indices:
        pos    = ee_pos_list[idx]
        rot    = ee_rot_list[idx]
        handle = pos + rot.apply(handle_offset_m)
        ax.plot([pos[0], handle[0]], [pos[1], handle[1]], [pos[2], handle[2]],
                color='gray', linewidth=1.0, linestyle='--', alpha=0.7)


def draw_world_axes(ax):
    """Small world-frame reference triad at the origin corner."""
    origin = np.array([0.0, 0.0, 0.0])
    L = 0.05
    for vec, color, label in [
        (np.array([L, 0, 0]), 'red',   '+X'),
        (np.array([0, L, 0]), 'green', '+Y'),
        (np.array([0, 0, L]), 'blue',  '+Z'),
    ]:
        ax.quiver(*origin, *vec, color=color, linewidth=1.5)
        ax.text(*(origin + vec * 1.3), label, color=color, fontsize=7)


def main():
    R_start = build_start_rotation()
    (pivot, axis_world,
     offset_0, ee_pos_list,
     ee_rot_list, handle_pos_list) = simulate_arc(R_start)

    ee_pos_arr     = np.array(ee_pos_list)
    handle_pos_arr = np.array(handle_pos_list)

    # ── Print diagnostics ────────────────────────────────────────────────────
    print("=== P-arc diagnostics ===")
    print(f"  Pivot (world):        {pivot}")
    print(f"  Axis  (world):        {axis_world}")
    print(f"  Radius (from offset): {np.linalg.norm(offset_0)*1000:.1f} mm")
    print(f"  EE start:             {ee_pos_arr[0]}")
    print(f"  EE end:               {ee_pos_arr[-1]}")
    print(f"  Handle start:         {handle_pos_arr[0]}")
    print(f"  Handle end:           {handle_pos_arr[-1]}")
    R_end = ee_rot_list[-1]
    print(f"  EE start euler (xyz): {R_start.as_euler('xyz', degrees=True)}")
    print(f"  EE end   euler (xyz): {R_end.as_euler('xyz', degrees=True)}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # EE tip path (blue)
    ax.plot(ee_pos_arr[:, 0], ee_pos_arr[:, 1], ee_pos_arr[:, 2],
            'b-', linewidth=2, label='EE tip path')

    # Handle path (orange)
    ax.plot(handle_pos_arr[:, 0], handle_pos_arr[:, 1], handle_pos_arr[:, 2],
            color='orange', linewidth=2, label='Handle path')

    # Pivot (red ×)
    ax.scatter(*pivot, color='red', s=80, marker='x', zorder=5, label='Pivot')

    # Axis of rotation arrow from pivot
    arrow_len = P_ARC_RADIUS_MM / 1000.0 * 0.6
    ax.quiver(*pivot, *(axis_world * arrow_len),
              color='purple', linewidth=1.5, label='Rotation axis')

    # EE-frame triads at N_FRAMES snapshots
    draw_frame_triads(ax, ee_pos_list, ee_rot_list, N_FRAMES)

    # Energy-unit sticks
    draw_energy_unit_sticks(ax, ee_pos_list, ee_rot_list, N_FRAMES)

    # World axes
    draw_world_axes(ax)

    # Start / end markers
    ax.scatter(*ee_pos_arr[0],  color='blue',   s=60, zorder=5)
    ax.scatter(*ee_pos_arr[-1], color='navy',   s=60, marker='^', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(
        f'P-arc  r={P_ARC_RADIUS_MM:.0f} mm  angle={P_ARC_ANGLE_DEG:.0f}°\n'
        f'center_dir_ee={P_ARC_CENTER_DIR_EE}  axis_ee={P_ARC_AXIS_EE}'
    )
    ax.legend(loc='upper left', fontsize=8)

    # Equal aspect ratio
    all_pts = np.vstack([ee_pos_arr, handle_pos_arr, pivot[None]])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    span  = (maxs - mins).max() / 2.0 + 0.05
    mid   = (mins + maxs) / 2.0
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
