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
"""

from __future__ import annotations

import numpy as np


class PoseFilter:
    """Kalman filter for cube position with orientation holdover.

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
    """

    def __init__(
        self,
        process_noise: float = 0.02,
        meas_noise: float = 0.005,
        max_vel: float = 0.5,
        max_coast_s: float = 0.5,
        reversal_boost: float = 8.0,
    ):
        self._q_std = process_noise
        self._r_std = meas_noise
        self._max_vel = max_vel
        self._max_coast_s = max_coast_s
        self._reversal_boost = reversal_boost

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
        if cube_pos is not None:
            if not self._initialized:
                self._x[:3] = cube_pos
                self._x[3:] = 0.0
                self._P = np.eye(6) * 0.01
                self._last_quat = cube_quat.copy() if cube_quat is not None else None
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
                    self._last_quat = cube_quat.copy()
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
