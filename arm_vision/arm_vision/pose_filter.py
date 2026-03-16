"""
Kalman-filter-based pose smoother for the vision teleop pipeline.

Provides continuous, smooth EE targets even when AprilTag detection drops
frames.  Uses a constant-velocity model for position and simple holdover
for orientation.

State vector: [x, y, z, vx, vy, vz]  (position + velocity in camera frame)

When a detection arrives  → Kalman update  (fuses measurement)
When detection is missing → Kalman predict (coasts on velocity estimate)
When coast time exceeds max_coast_s → returns None (arm should halt)

Additionally enforces a max EE velocity so detection glitches cannot
produce sudden jumps.

Orientation filtering:
  - Angular rate gate: clamps the per-frame orientation change to max_ori_rate
    so detection glitches (AprilTag ambiguity flips, tag appear/disappear) are
    smoothly absorbed instead of snapping the arm.
  - NLERP EMA: blends the accepted measurement toward the current orientation
    for additional smoothing.
"""

from __future__ import annotations

import numpy as np


class PoseFilter:
    """Kalman filter for cube position with orientation smoothing.

    Parameters
    ----------
    process_noise : float
        Process noise standard deviation (m).  Higher → trusts measurements
        more, lower → trusts the velocity model more.
    meas_noise : float
        Measurement noise standard deviation (m).  Typically 2–5 mm for
        AprilTag pose estimation at 0.3–0.5 m range.
    max_vel : float
        Maximum allowed EE velocity (m/s).  Caps the Kalman velocity
        estimate to prevent jumps from detection glitches.
    max_coast_s : float
        Maximum time (s) to coast without a measurement before returning
        None (which causes the arm to halt via detection_timeout_s).
    max_ori_rate : float
        Maximum allowed orientation change rate (rad/s).  When the angle
        between consecutive detected quaternions exceeds this rate, the
        step is clamped — the filter advances only by the allowed amount.
        Prevents AprilTag ambiguity flips from jerking the arm.
    ori_ema_alpha : float
        NLERP exponential moving average factor for orientation.
        0.0 = infinite smoothing (never updates), 1.0 = no smoothing.
        Applied AFTER the rate gate, so normal motion is smoothed gently
        and glitches are doubly attenuated.
    """

    def __init__(
        self,
        process_noise: float = 0.02,
        meas_noise: float = 0.005,
        max_vel: float = 0.5,
        max_coast_s: float = 0.5,
        reversal_boost: float = 8.0,
        max_ori_rate: float = 5.0,
        ori_ema_alpha: float = 0.5,
    ):
        self._q_std = process_noise
        self._r_std = meas_noise
        self._max_vel = max_vel
        self._max_coast_s = max_coast_s
        self._reversal_boost = reversal_boost
        self._max_ori_rate = max_ori_rate
        self._ori_alpha = ori_ema_alpha

        # Measurement matrix: observe [x, y, z] from state [x, y, z, vx, vy, vz]
        self._H = np.zeros((3, 6))
        self._H[:3, :3] = np.eye(3)
        self._R = np.eye(3) * meas_noise ** 2

        # State
        self._x: np.ndarray = np.zeros(6)           # [x, y, z, vx, vy, vz]
        self._P: np.ndarray = np.eye(6) * 1.0       # covariance
        self._last_quat: np.ndarray | None = None    # orientation holdover
        self._initialized = False
        self._coast_time = 0.0
        self._prev_meas: np.ndarray | None = None    # for direction reversal detection
        self._last_q_boost: float = 1.0               # exposed for diagnostics
        self._ori_dt_accum: float = 0.0               # time since last orientation update

    def step(
        self,
        cube_pos: np.ndarray | None,
        cube_quat: np.ndarray | None,
        dt: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Process one frame.

        Parameters
        ----------
        cube_pos  : detected position [x,y,z] in camera frame, or None if no detection.
        cube_quat : detected quaternion [x,y,z,w] in camera frame, or None.
        dt        : time since last call (seconds).

        Returns
        -------
        (filtered_pos, filtered_quat) — both in camera frame.
        Returns (None, None) if not yet initialised or coast time exceeded.
        """
        self._ori_dt_accum += dt

        if cube_pos is not None:
            if not self._initialized:
                self._x[:3] = cube_pos
                self._x[3:] = 0.0
                self._P = np.eye(6) * 0.01
                self._last_quat = cube_quat.copy() if cube_quat is not None else None
                self._ori_dt_accum = 0.0
                self._initialized = True
                self._coast_time = 0.0
                self._prev_meas = cube_pos.copy()
                return cube_pos.copy(), self._last_quat
            else:
                # Detect direction change: if the measurement-implied velocity
                # differs significantly from the filter's velocity (reversal OR
                # orthogonal switch like up→right), boost process noise so the
                # filter snaps to the new trajectory instead of coasting in the
                # old direction.
                q_boost = 1.0
                if self._prev_meas is not None and dt > 1e-6:
                    meas_vel = (cube_pos - self._prev_meas) / dt
                    filt_vel = self._x[3:6]
                    filt_speed = float(np.linalg.norm(filt_vel))
                    meas_speed = float(np.linalg.norm(meas_vel))
                    if filt_speed > 0.01 and meas_speed > 0.01:
                        cos_sim = float(np.dot(meas_vel, filt_vel)) / (meas_speed * filt_speed)
                        if cos_sim < 0.5:
                            # Direction changed by > 60° — boost
                            q_boost = self._reversal_boost
                self._prev_meas = cube_pos.copy()
                self._last_q_boost = q_boost

                self._predict(dt, q_boost=q_boost)
                self._update(cube_pos)
                if cube_quat is not None:
                    self._last_quat = self._filter_orientation(cube_quat)
                self._coast_time = 0.0
        else:
            if not self._initialized:
                return None, None
            self._predict(dt)
            self._coast_time += dt
            if self._coast_time > self._max_coast_s:
                return None, None

        return self._clamped_position(), self._last_quat

    @property
    def velocity(self) -> np.ndarray:
        """Current Kalman velocity estimate [vx, vy, vz] (m/s)."""
        return self._x[3:6].copy()

    @property
    def last_q_boost(self) -> float:
        """Process noise boost applied in the most recent step (1.0 = no boost)."""
        return self._last_q_boost

    def reset(self):
        """Reset the filter state (e.g. after re-homing)."""
        self._initialized = False
        self._coast_time = 0.0
        self._last_quat = None
        self._ori_dt_accum = 0.0

    # ── Orientation filtering ─────────────────────────────────────────────────

    def _filter_orientation(self, new_quat: np.ndarray) -> np.ndarray:
        """Angular rate gate + NLERP EMA for orientation smoothing.

        1. Antipodal check — ensure shortest rotation path (q ≡ -q).
        2. Rate gate — if the angular change since the last accepted frame
           exceeds max_ori_rate, clamp: advance only by the allowed amount
           toward the measurement (no hard rejection, so the filter still
           tracks real motion).
        3. NLERP EMA — blend the (possibly clamped) measurement with the
           current orientation for additional smoothing.
        """
        if self._last_quat is None:
            self._ori_dt_accum = 0.0
            return new_quat.copy()

        q = new_quat.copy()

        # Antipodal check — pick the hemisphere closer to _last_quat
        if np.dot(self._last_quat, q) < 0:
            q = -q

        # Angular distance between current filtered and new measurement
        dot = np.clip(float(np.dot(self._last_quat, q)), -1.0, 1.0)
        angle = 2.0 * np.arccos(dot)   # radians

        ori_dt = max(self._ori_dt_accum, 1e-6)
        self._ori_dt_accum = 0.0

        # Rate gate: clamp angular step to max_ori_rate * dt
        if angle > 1e-6:
            max_angle = self._max_ori_rate * ori_dt
            if angle > max_angle:
                # Clamp: advance only the allowed fraction toward measurement
                t = max_angle / angle
                q = self._nlerp(self._last_quat, q, t)

        # NLERP EMA blend
        return self._nlerp(self._last_quat, q, self._ori_alpha)

    @staticmethod
    def _nlerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        """Normalized linear interpolation between two unit quaternions."""
        result = (1.0 - alpha) * q0 + alpha * q1
        n = float(np.linalg.norm(result))
        if n > 1e-6:
            return result / n
        return q0.copy()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _predict(self, dt: float, q_boost: float = 1.0):
        """Constant-velocity prediction step.

        q_boost : multiplier on process noise.  Set > 1 on direction reversals
                  so the filter trusts the model less and snaps to new measurements.
        """
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt

        # Process noise: acceleration uncertainty
        G = np.zeros((6, 3))
        G[:3, :] = np.eye(3) * 0.5 * dt * dt
        G[3:, :] = np.eye(3) * dt
        Q = G @ G.T * (self._q_std * q_boost) ** 2

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def _update(self, z: np.ndarray):
        """Kalman measurement update."""
        y = z - self._H @ self._x                              # innovation
        S = self._H @ self._P @ self._H.T + self._R            # innovation covariance
        K = self._P @ self._H.T @ np.linalg.solve(S, np.eye(3))  # Kalman gain
        self._x = self._x + K @ y
        self._P = (np.eye(6) - K @ self._H) @ self._P

    def _clamped_position(self) -> np.ndarray:
        """Return position with velocity clamped to max_vel."""
        vel = self._x[3:6]
        speed = float(np.linalg.norm(vel))
        if speed > self._max_vel:
            self._x[3:6] = vel * (self._max_vel / speed)
        return self._x[:3].copy()
