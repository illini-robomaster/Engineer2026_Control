"""
5-face AprilTag cube calibration.

Usage:
  python main.py calibrate-cube [--device 0] [--tag-family tag36h11]
                                [--side-length 0.06]
                                [--camera-cal data/camera_calibration.yaml]
                                [--output data/cube_config.yaml]

Procedure:
  The calibration identifies the tag IDs on each face and measures the cube's
  half-size from the detected tag poses.

  For each of the 5 faces:
    1. Rotate the cube so that face is pointing toward the camera.
    2. Hold still and press SPACE to capture.
    3. The detected tag ID and measured half-size are recorded.
    4. You will be prompted to assign the face label (+X/-X/+Y/-Y/+Z).

  After all 5 faces, the face rotation quaternions are computed geometrically
  and a cube_config.yaml compatible with the arm_vision detector is saved.

Output (data/cube_config.yaml):
  side_length: measured average (metres)
  tag_family: tag36h11
  faces:
    - id:               tag id
      face:             face label (+X/-X/+Y/-Y/+Z)
      position:         [x, y, z]  tag centre in cube frame (metres)
      orientation_xyzw: [x, y, z, w]  tag orientation in cube frame
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .camera_cal import load as load_camera_cal

# ── Face geometry ─────────────────────────────────────────────────────────────
# For each face label: the outward unit normal and the "up" direction of the tag
# both expressed in the cube's coordinate frame.
# Cube frame convention: X=forward, Y=left, Z=up.
#
# Tag convention (pupil-apriltags / apriltag_ros):
#   tag +Z = outward normal (toward camera when facing camera)
#   tag +Y = upward in the image when tag is upright
#   tag +X = rightward in the image when tag is upright

_FACE_NORMALS: dict[str, tuple[list, list]] = {
    '+X': ([1,  0,  0], [0,  0,  1]),  # normal=+X, tag-up=cube-Z
    '-X': ([-1, 0,  0], [0,  0,  1]),
    '+Y': ([0,  1,  0], [0,  0,  1]),
    '-Y': ([0, -1,  0], [0,  0,  1]),
    '+Z': ([0,  0,  1], [0,  1,  0]),  # top face: tag-up=cube-Y
}
FACE_LABELS = list(_FACE_NORMALS.keys())


def _face_quaternion(face_label: str) -> list[float]:
    """
    Compute the quaternion [xyzw] representing the tag frame's orientation
    inside the cube frame for the given face.

    The rotation matrix R has columns = [tag_x, tag_y, tag_z] expressed in
    the cube frame, where tag_z = face normal, tag_y = face "up".
    """
    normal, up = _FACE_NORMALS[face_label]
    z_col = np.array(normal, dtype=float)
    y_col = np.array(up,     dtype=float)
    x_col = np.cross(y_col, z_col)
    x_col /= np.linalg.norm(x_col)
    R = np.column_stack([x_col, y_col, z_col])
    q = Rotation.from_matrix(R).as_quat()   # scipy returns [x, y, z, w]
    return q.tolist()


# ── Main calibration routine ──────────────────────────────────────────────────

def run(
    device: int = 0,
    tag_family: str = 'tag36h11',
    side_length: float = 0.06,
    camera_cal: str = 'data/camera_calibration.yaml',
    output: str = 'data/cube_config.yaml',
) -> dict:
    """
    Interactive 5-face cube calibration.  Returns the cube config dict.
    """
    try:
        from pupil_apriltags import Detector
    except ImportError:
        raise ImportError("Install pupil-apriltags:  pip install pupil-apriltags")

    K, dist = load_camera_cal(camera_cal)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    tag_size = side_length   # tag occupies the full face side for pose scaling

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

    print('\n=== 5-Face Cube Calibration ===')
    print(f'  Tag family : {tag_family}')
    print(f'  Side length: {side_length*100:.1f} cm (approx)')
    print()
    print('  For each of the 5 faces:')
    print('    → Point that face toward the camera')
    print('    → Hold still and press SPACE to capture')
    print('    → Enter the face label when prompted (+X/-X/+Y/-Y/+Z)')
    print()
    print('  Q / ESC → abort\n')

    captures: list[dict] = []   # {tag_id, half_size, face_label (filled later)}

    remaining_labels = list(FACE_LABELS)

    while len(captures) < 5:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=tag_size,
        )

        display = frame.copy()
        for d in detections:
            corners = d.corners.astype(int)
            cv2.polylines(display, [corners.reshape(-1, 1, 2)], True, (0, 220, 0), 2)
            cv2.putText(display, f'id={d.tag_id}',
                        tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 220, 0), 2)

        cv2.putText(display, f'Captured: {len(captures)}/5',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
        if remaining_labels:
            cv2.putText(display, f'Next face: {remaining_labels[0]}',
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 0), 2)
        cv2.imshow('Cube Calibration', display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError('Calibration aborted by user.')

        if key == ord(' '):
            if not detections:
                print('  [WARN] No tag detected — move the cube closer or improve lighting.')
                continue
            if len(detections) > 1:
                print(f'  [WARN] {len(detections)} tags visible — show only one face at a time.')
                continue

            d = detections[0]
            # pose_R maps tag frame to camera frame; tag +Z = outward normal
            # Cube centre is half_size behind the tag along tag -Z in camera frame
            R_tag = d.pose_R                        # 3×3, tag→cam
            t_tag = d.pose_t.reshape(3)             # tag origin in cam frame
            tag_z_in_cam = R_tag[:, 2]              # tag's +Z axis in cam frame
            cube_center_cam = t_tag - side_length / 2.0 * tag_z_in_cam
            measured_half = float(np.linalg.norm(t_tag - cube_center_cam))

            print(f'  Detected tag id={d.tag_id}  '
                  f'tag-to-cam dist={np.linalg.norm(t_tag)*100:.1f} cm  '
                  f'half_size≈{measured_half*100:.1f} cm')

            # Ask user for face label
            print(f'  Remaining labels: {remaining_labels}')
            while True:
                label = input('  Enter face label for this tag: ').strip()
                if label in FACE_LABELS:
                    break
                print(f'  Invalid — choose from {FACE_LABELS}')

            if label not in remaining_labels:
                print(f'  [WARN] Face {label} already captured; overwriting.')
                captures = [c for c in captures if c['face_label'] != label]
            else:
                remaining_labels.remove(label)

            captures.append({
                'tag_id':    int(d.tag_id),
                'half_size': measured_half,
                'face_label': label,
            })
            print(f'  Saved: id={d.tag_id} → face {label}  '
                  f'({len(captures)}/5 done)\n')

    cap.release()
    cv2.destroyAllWindows()

    # ── Build output ──────────────────────────────────────────────────────────
    avg_half_size = float(np.mean([c['half_size'] for c in captures]))
    measured_side = avg_half_size * 2.0
    print(f'Average measured side length: {measured_side*100:.2f} cm  '
          f'(input: {side_length*100:.1f} cm)')

    faces = []
    for c in captures:
        label    = c['face_label']
        normal   = np.array(_FACE_NORMALS[label][0], dtype=float)
        position = (normal * avg_half_size).tolist()
        q_xyzw   = _face_quaternion(label)
        faces.append({
            'id':               c['tag_id'],
            'face':             label,
            'position':         position,
            'orientation_xyzw': q_xyzw,
        })
    faces.sort(key=lambda f: FACE_LABELS.index(f['face']))

    result = {
        'side_length': measured_side,
        'tag_family':  tag_family,
        'faces':       faces,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)

    print(f'Cube config saved to: {out_path}')
    return result


def load(path: str = 'data/cube_config.yaml') -> dict:
    """Load cube config from YAML.  Returns the raw dict."""
    with open(path) as f:
        return yaml.safe_load(f)
