"""
AprilTag world-map calibration — multi-tag relative pose method.

Usage:
  python main.py calibrate-cube [--device 0] [--tag-family tag25h9]
                                [--tag-size 0.032]
                                [--camera-cal data/camera_calibration.yaml]
                                [--output data/cube_config.yaml]

  --tag-size is the physical side length of the AprilTag's outer black border
  in metres (NOT the cube side, NOT including white quiet zone).
  For a 3.2 cm black square: --tag-size 0.032

Procedure:
  Tilt/rotate the cube to expose 2+ tag faces at once and press SPACE.
  Repeat from varied angles until all tags are connected in the graph.
  Press ENTER when enough captures are collected.

Output (data/cube_config.yaml):
  tag_size:         AprilTag black-square side in metres
  tag_family:       tag25h9
  reference_tag_id: 0
  tags:
    - id:               tag id
      position:         [x, y, z]  measured tag centre in cube frame (metres)
      orientation_xyzw: [x, y, z, w]  measured tag orientation in cube frame

Cube frame definition:
  Origin  — 20 mm inside tag 0's face centre (along tag 0's -Z axis).
  Axes    — identical to tag 0's axes (X right, Y up, Z outward when facing tag 0).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .camera_cal import load as load_camera_cal

# Distance from tag-0 face to the defined cube-frame origin (half the 4 cm side).
_CUBE_DEPTH = 0.020   # metres


# ── Pose graph utilities ──────────────────────────────────────────────────────

def _average_rotations(Rs: list[np.ndarray]) -> np.ndarray:
    """Project the element-wise mean of rotation matrices back onto SO(3)."""
    R_mean = np.mean(Rs, axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _build_pose_map(
    rel_obs: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]],
    root_id: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    BFS from root_id over averaged pairwise relative transforms.

    Each output entry (R, t): R maps tag frame → root frame; t is the tag
    origin expressed in the root tag's coordinate frame.
    """
    avg: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (a, b), obs in rel_obs.items():
        R_avg = _average_rotations([o[0] for o in obs])
        t_avg = np.mean([o[1] for o in obs], axis=0)
        avg[(a, b)] = (R_avg, t_avg)
        avg[(b, a)] = (R_avg.T, -(R_avg.T @ t_avg))

    all_ids = {i for pair in avg for i in pair}
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {
        root_id: (np.eye(3), np.zeros(3))
    }
    queue = [root_id]
    while queue:
        cur = queue.pop(0)
        R_cur, t_cur = poses[cur]
        for other in all_ids - set(poses):
            if (cur, other) in avg:
                R_rel, t_rel = avg[(cur, other)]
                poses[other] = (R_cur @ R_rel, R_cur @ t_rel + t_cur)
                queue.append(other)
    return poses


# ── Main calibration routine ──────────────────────────────────────────────────

def run(
    device: int = 0,
    tag_family: str = 'tag25h9',
    tag_size: float = 0.032,
    camera_cal: str = 'data/camera_calibration.yaml',
    output: str = 'data/cube_config.yaml',
    min_rel_captures: int = 8,
) -> dict:
    """Interactive multi-tag world-map calibration.  Returns the cube config dict."""
    try:
        from pupil_apriltags import Detector
    except ImportError:
        raise ImportError("Install pupil-apriltags:  pip install pupil-apriltags")

    K, dist = load_camera_cal(camera_cal)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    detector = Detector(
        families=tag_family,
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
    )

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera device {device}')

    print('\n=== AprilTag World-Map Calibration ===')
    print(f'  Tag size (black square): {tag_size * 100:.1f} cm')
    print(f'  Cube-frame origin: {_CUBE_DEPTH * 1000:.0f} mm inside tag 0 along -Z')
    print()
    print('  Tilt the cube so that 2+ tag faces are visible at once.')
    print('  SPACE → capture frame  (need ≥2 tags visible)')
    print(f'  Collect ≥{min_rel_captures} frames from varied angles, then press ENTER.')
    print('  Q / ESC → abort\n')

    # rel_obs[(id_A, id_B)] accumulates (R_B_in_A, t_B_in_A) per observation
    rel_obs: dict[tuple[int, int], list] = defaultdict(list)
    rel_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size,
        )

        disp = frame.copy()
        for d in dets:
            cv2.polylines(disp, [d.corners.astype(int).reshape(-1, 1, 2)],
                          True, (0, 220, 0), 2)
            cv2.putText(disp, f'id={d.tag_id}',
                        tuple(d.corners[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        n_vis = len(dets)
        clr = (0, 220, 220) if n_vis >= 2 else (0, 100, 200)
        cv2.putText(disp,
                    f'Visible: {n_vis} tag(s)   Captured: {rel_count}/{min_rel_captures}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
        if rel_count >= min_rel_captures:
            cv2.putText(disp, 'ENTER → save calibration',
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

        cv2.imshow('AprilTag Calibration', disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError('Calibration aborted.')

        if key == 13 and rel_count >= min_rel_captures:   # Enter
            break

        if key == ord(' '):
            if n_vis < 2:
                print(f'  [WARN] Only {n_vis} tag(s) visible — need ≥2.')
                continue

            det_map = {d.tag_id: d for d in dets}
            ids = list(det_map.keys())
            pairs = 0
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    R_a = det_map[a].pose_R;  t_a = det_map[a].pose_t.reshape(3)
                    R_b = det_map[b].pose_R;  t_b = det_map[b].pose_t.reshape(3)
                    rel_obs[(a, b)].append((R_a.T @ R_b, R_a.T @ (t_b - t_a)))
                    pairs += 1
            rel_count += 1
            print(f'  Frame {rel_count}: tags {ids}  ({pairs} pair(s) recorded)')

    cap.release()
    cv2.destroyAllWindows()

    # ── Build pose map ────────────────────────────────────────────────────────
    all_ids: set[int] = set()
    for a, b in rel_obs:
        all_ids |= {a, b}
    print(f'\n  Tags observed: {sorted(all_ids)}')

    # Use tag 0 as root (cube-frame reference); fall back to most-connected tag
    obs_count: dict[int, int] = defaultdict(int)
    for (a, b), lst in rel_obs.items():
        obs_count[a] += len(lst)
        obs_count[b] += len(lst)

    if 0 in all_ids:
        root_id = 0
    else:
        root_id = max(obs_count, key=obs_count.__getitem__)
        print(f'  [WARN] Tag id=0 not observed; using id={root_id} as reference.')

    poses = _build_pose_map(dict(rel_obs), root_id)

    unreachable = all_ids - set(poses)
    if unreachable:
        print(f'  [WARN] Tags {sorted(unreachable)} are not reachable from root — '
              f'capture more frames that connect them.')

    # ── Express all poses in cube frame ───────────────────────────────────────
    # Cube frame: same orientation as root (tag 0), origin _CUBE_DEPTH along -Z
    # In root frame: tag 0 is at (I, [0,0,0]), its +Z = [0,0,1].
    # => cube origin in root frame = [0, 0, -_CUBE_DEPTH]
    cube_origin = np.array([0.0, 0.0, -_CUBE_DEPTH])

    tags_out = []
    for tag_id, (R_tag, t_tag) in sorted(poses.items()):
        pos    = (t_tag - cube_origin).tolist()
        q_xyzw = Rotation.from_matrix(R_tag).as_quat().tolist()
        tags_out.append({
            'id':               int(tag_id),
            'position':         pos,
            'orientation_xyzw': q_xyzw,
        })

    print('\n  Tag positions in cube frame (metres):')
    for t in tags_out:
        p = t['position']
        print(f"    id={t['id']}  pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")

    result = {
        'tag_size':         float(tag_size),
        'tag_family':       tag_family,
        'reference_tag_id': int(root_id),
        'tags':             tags_out,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)

    print(f'  Cube config saved to: {out_path}')
    return result


def load(path: str = 'data/cube_config.yaml') -> dict:
    """Load cube config from YAML.  Returns the raw dict."""
    with open(path) as f:
        return yaml.safe_load(f)
