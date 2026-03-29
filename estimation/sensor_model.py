"""
Sensor noise model for IMU, GPS, and barometer.

Simulates realistic sensor characteristics:
  - IMU:  Gaussian noise + slowly drifting bias
  - GPS:  Gaussian noise at 5 Hz, momentarily unavailable near ground
  - Baro: Gaussian noise + altitude-proportional error at 50 Hz
"""

import numpy as np
from dataclasses import dataclass
import config


@dataclass
class SensorReadings:
    imu_accel:   np.ndarray   # m/s^2, body frame, noisy
    imu_gyro:    np.ndarray   # rad/s, body frame, noisy
    gps_pos:     np.ndarray   # m, world frame, 5 Hz (None if not updated)
    gps_vel:     np.ndarray   # m/s, world frame, 5 Hz (None if not updated)
    baro_alt:    float        # m, scalar, 50 Hz
    gps_fresh:   bool         # True when a new GPS packet arrived this step


class SensorModel:
    def __init__(self, params, rng: np.random.Generator = None):
        self.params = params
        self.rng    = rng or np.random.default_rng(0)

        # Drifting accelerometer bias (slowly varying)
        self._accel_bias = self.rng.normal(0, config.IMU_ACCEL_BIAS_STD, 3)
        self._bias_walk  = config.IMU_ACCEL_BIAS_STD * 0.01   # random walk scale per step

        self._gps_counter  = 0
        self._gps_interval = max(1, round((1.0 / config.GPS_RATE_HZ) / config.DT_PHYSICS))

    def step(self, true_state, sim_time: float) -> SensorReadings:
        """Generate one set of sensor readings from the true physics state."""

        # ── IMU acceleration (body frame) ──────────────────────────────────
        # True specific force in body frame: R^T @ (a_world - g_world)
        R = true_state.dcm
        a_world = np.zeros(3)  # computed from state derivative — approx with -g for display
        g_world = np.array([0.0, 0.0, -config.G])
        # Approximate: body accel ≈ R^T @ vel_dot ≈ we just add noise to body-frame gravity projection
        g_body     = R.T @ (-g_world)   # gravity as felt in body (reaction)
        accel_true = g_body
        accel_noisy = (accel_true
                       + self._accel_bias
                       + self.rng.normal(0, config.IMU_ACCEL_NOISE_STD, 3))

        # Bias random walk
        self._accel_bias += self.rng.normal(0, self._bias_walk, 3)
        np.clip(self._accel_bias, -0.5, 0.5, out=self._accel_bias)

        # ── IMU gyroscope (body frame) ────────────────────────────────────
        gyro_noisy = (true_state.ang_vel
                      + self.rng.normal(0, config.IMU_GYRO_NOISE_STD, 3))

        # ── Barometer ─────────────────────────────────────────────────────
        alt_noise  = config.BARO_ALT_NOISE_STD + 0.002 * true_state.altitude
        baro_alt   = true_state.altitude + self.rng.normal(0, alt_noise)

        # ── GPS (5 Hz) ────────────────────────────────────────────────────
        self._gps_counter += 1
        gps_fresh = self._gps_counter >= self._gps_interval
        if gps_fresh:
            self._gps_counter = 0
            gps_pos = true_state.position + self.rng.normal(0, config.GPS_POS_NOISE_STD, 3)
            gps_vel = true_state.velocity + self.rng.normal(0, config.GPS_VEL_NOISE_STD, 3)
        else:
            gps_pos = np.zeros(3)
            gps_vel = np.zeros(3)

        return SensorReadings(
            imu_accel = accel_noisy,
            imu_gyro  = gyro_noisy,
            gps_pos   = gps_pos,
            gps_vel   = gps_vel,
            baro_alt  = baro_alt,
            gps_fresh = gps_fresh,
        )
