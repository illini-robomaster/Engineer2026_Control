"""
Camera calibration using an OpenCV checkerboard pattern.

Usage:
  python main.py calibrate-camera [--device 0] [--cols 9] [--rows 6]
                                  [--square-size 0.025] [--output data/camera_calibration.yaml]

Procedure:
  1. Print a checkerboard (default 9×6 inner corners, 25 mm squares).
  2. Hold it flat and show it to the camera from multiple angles.
  3. Press SPACE to capture a frame, ESC or Q to finish and calibrate.
  4. At least 10 captures are recommended for a good calibration.

Output (data/camera_calibration.yaml):
  camera_matrix: 3×3 intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
  dist_coeffs:   distortion coefficients [k1, k2, p1, p2, k3]
  rms_error:     reprojection RMS error in pixels
  image_size:    [width, height]
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def run(
    device: int = 0,
    cols: int = 9,
    rows: int = 6,
    square_size: float = 0.025,
    output: str = 'data/camera_calibration.yaml',
) -> dict:
    """
    Interactive camera calibration.  Returns the calibration dict.

    Parameters
    ----------
    device      : OpenCV camera index (or path to a video file for testing).
    cols        : Number of inner corners along the wide side of the board.
    rows        : Number of inner corners along the short side.
    square_size : Physical side length of one square in metres.
    output      : Path to write the resulting YAML.
    """
    board_size = (cols, rows)
    # 3-D corner positions in the board's own frame (Z=0 plane)
    obj_pts_template = np.zeros((cols * rows, 3), dtype=np.float32)
    obj_pts_template[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    )

    obj_points: list[np.ndarray] = []   # 3-D points for each captured frame
    img_points: list[np.ndarray] = []   # 2-D points for each captured frame
    frame_size: tuple[int, int] | None = None

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera device {device}')

    print('\n=== Camera Calibration ===')
    print(f'  Board: {cols}×{rows} inner corners, {square_size*1000:.0f} mm squares')
    print('  SPACE  → capture frame')
    print('  C      → clear all captures')
    print('  Q / ESC → finish and calibrate')
    print()

    last_capture_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print('[WARN] Cannot read frame.')
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size, None)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, board_size, corners, found)
            cv2.putText(display, 'Board found — SPACE to capture',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        else:
            cv2.putText(display, 'No board detected',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2)

        cv2.putText(display, f'Captures: {len(obj_points)}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 0), 2)
        cv2.imshow('Camera Calibration', display)

        key = cv2.waitKey(1) & 0xFF
        now = time.monotonic()

        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            break

        if key == ord('c') or key == ord('C'):
            obj_points.clear()
            img_points.clear()
            print('  Cleared all captures.')

        if key == ord(' ') and found and (now - last_capture_time) > 0.5:
            # Refine corners to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(obj_pts_template.copy())
            img_points.append(corners_refined)
            frame_size = (gray.shape[1], gray.shape[0])
            last_capture_time = now
            print(f'  Captured frame #{len(obj_points)}')

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_points) < 5:
        raise RuntimeError(
            f'Only {len(obj_points)} captures — need at least 5 to calibrate.')

    print(f'\nCalibrating with {len(obj_points)} frames…')
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, frame_size, None, None)

    result = {
        'camera_matrix': K.tolist(),
        'dist_coeffs':   dist.tolist(),
        'rms_error':     float(rms),
        'image_size':    list(frame_size),
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)

    print(f'RMS reprojection error: {rms:.4f} px')
    print(f'Calibration saved to: {out_path}')
    return result


def load(path: str = 'data/camera_calibration.yaml') -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera calibration from YAML.

    Returns
    -------
    K    : 3×3 camera matrix
    dist : distortion coefficients array
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    K    = np.array(data['camera_matrix'], dtype=np.float64)
    dist = np.array(data['dist_coeffs'],   dtype=np.float64)
    return K, dist
