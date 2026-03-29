"""
Extended Kalman Filter for lander state estimation.

EKF state (13-dim):
  [0:3]  position   (m, ENU)
  [3:6]  velocity   (m/s)
  [6:10] quaternion (w, x, y, z)
  [10:13] accel_bias (m/s^2, body frame)

Prediction uses IMU accelerometer + gyroscope as control inputs.
Updates from GPS (position + velocity) and barometer (altitude).
Jacobians computed numerically for correctness.
"""

import numpy as np
from physics.quaternion import quat_normalize, omega_matrix
import config


def _quat_to_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def _numerical_jacobian_vel_quat(q: np.ndarray, a_body: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """d(R(q) @ a_body)/dq, shape (3,4), computed numerically."""
    J = np.zeros((3, 4))
    R0 = _quat_to_dcm(q)
    f0 = R0 @ a_body
    for i in range(4):
        dq = np.zeros(4)
        dq[i] = eps
        Rp = _quat_to_dcm(quat_normalize(q + dq))
        J[:, i] = (Rp @ a_body - f0) / eps
    return J


class ExtendedKalmanFilter:
    def __init__(self, initial_state, params):
        """
        initial_state: PhysicsState
        params:        VehicleParams
        """
        self.params = params
        sv = initial_state.vec

        # EKF state x̂ (13-dim)
        self._x = np.zeros(13)
        self._x[0:3]   = sv[0:3]      # position
        self._x[3:6]   = sv[3:6]      # velocity
        self._x[6:10]  = sv[6:10]     # quaternion
        self._x[10:13] = np.zeros(3)  # accel bias

        # Covariance P (13×13)
        self._P = np.diag([
            5.0, 5.0, 5.0,       # position (m²)
            1.0, 1.0, 1.0,       # velocity
            0.01, 0.01, 0.01, 0.01,  # quaternion
            0.02, 0.02, 0.02,    # accel bias
        ])

        # Process noise Q
        q_p = config.EKF_PROCESS_NOISE_POS
        q_v = config.EKF_PROCESS_NOISE_VEL
        q_q = config.EKF_PROCESS_NOISE_QUAT
        q_b = config.EKF_PROCESS_NOISE_BIAS
        self._Q = np.diag([
            q_p, q_p, q_p,
            q_v, q_v, q_v,
            q_q, q_q, q_q, q_q,
            q_b, q_b, q_b,
        ])

        # Measurement noise
        gp  = config.GPS_POS_NOISE_STD
        gv  = config.GPS_VEL_NOISE_STD
        ba  = config.BARO_ALT_NOISE_STD
        self._R_gps  = np.diag([gp**2, gp**2, (gp*2)**2,
                                 gv**2, gv**2, (gv*2)**2])
        self._R_baro = np.array([[ba**2]])

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def position(self) -> np.ndarray:
        return self._x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._x[3:6].copy()

    @property
    def quaternion(self) -> np.ndarray:
        return quat_normalize(self._x[6:10])

    @property
    def covariance_trace(self) -> float:
        return float(np.trace(self._P))

    def get_state_vec(self) -> np.ndarray:
        x = self._x.copy()
        x[6:10] = quat_normalize(x[6:10])
        return x

    # ── Predict (IMU integration) ─────────────────────────────────────────────

    def predict(self, imu_accel: np.ndarray, imu_gyro: np.ndarray, dt: float):
        x = self._x
        q = quat_normalize(x[6:10])

        # De-bias accelerometer
        a_corrected = imu_accel - x[10:13]

        # Rotate to world frame and add gravity
        R = _quat_to_dcm(q)
        a_world = R @ a_corrected + np.array([0.0, 0.0, -config.G])

        # Integrate position and velocity
        x[0:3] += x[3:6] * dt
        x[3:6] += a_world * dt

        # Integrate quaternion
        dq = omega_matrix(imu_gyro) @ q * dt
        x[6:10] = quat_normalize(q + dq)

        # Bias random walk: no deterministic term (modelled via Q)

        # State transition Jacobian F
        F = np.eye(13)
        F[0:3, 3:6]   = np.eye(3) * dt
        F[3:6, 6:10]  = _numerical_jacobian_vel_quat(q, a_corrected) * dt
        F[3:6, 10:13] = -R * dt
        Omega = omega_matrix(imu_gyro)
        F[6:10, 6:10] = np.eye(4) + Omega * dt

        self._x = x
        self._P = F @ self._P @ F.T + self._Q

    # ── Update steps ──────────────────────────────────────────────────────────

    def update_gps(self, gps_pos: np.ndarray, gps_vel: np.ndarray):
        z = np.concatenate([gps_pos, gps_vel])
        h = np.concatenate([self._x[0:3], self._x[3:6]])
        H = np.zeros((6, 13))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        self._update(z, h, H, self._R_gps)

    def update_baro(self, baro_alt: float):
        z = np.array([baro_alt])
        h = np.array([self._x[2]])
        H = np.zeros((1, 13))
        H[0, 2] = 1.0
        self._update(z, h, H, self._R_baro)

    def _update(self, z: np.ndarray, h: np.ndarray, H: np.ndarray, R: np.ndarray):
        inn = z - h
        S   = H @ self._P @ H.T + R
        K   = self._P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0])).T
        self._x += K @ inn
        I_KH    = np.eye(13) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T  # Joseph form
        self._x[6:10] = quat_normalize(self._x[6:10])
