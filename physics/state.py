"""
Physics state vector definition.

State: 14-element float64 array
  [0:3]  position     (x, y, z) m, ENU frame, z=up
  [3:6]  velocity     (vx, vy, vz) m/s
  [6:10] quaternion   (w, x, y, z) body-to-world
  [10:13] ang_vel     (p, q, r) rad/s, body frame
  [13]   fuel_mass    kg
"""

import numpy as np
from physics.quaternion import quat_to_dcm, quat_to_euler, euler_to_quat


STATE_DIM = 14


class PhysicsState:
    __slots__ = ('_v',)

    def __init__(self, vec: np.ndarray):
        assert vec.shape == (STATE_DIM,)
        self._v = vec.copy()

    @classmethod
    def zeros(cls) -> 'PhysicsState':
        v = np.zeros(STATE_DIM)
        v[6] = 1.0  # quaternion w=1 (identity rotation)
        return cls(v)

    # ── Named views ──────────────────────────────────────────────────────────

    @property
    def vec(self) -> np.ndarray:
        return self._v

    @property
    def position(self) -> np.ndarray:
        return self._v[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self._v[3:6]

    @property
    def quaternion(self) -> np.ndarray:
        return self._v[6:10]

    @property
    def ang_vel(self) -> np.ndarray:
        return self._v[10:13]

    @property
    def fuel_mass(self) -> float:
        return float(self._v[13])

    @property
    def altitude(self) -> float:
        return float(self._v[2])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self._v[3:6]))

    @property
    def euler(self) -> np.ndarray:
        return quat_to_euler(self._v[6:10])

    @property
    def dcm(self) -> np.ndarray:
        return quat_to_dcm(self._v[6:10])

    def copy(self) -> 'PhysicsState':
        return PhysicsState(self._v.copy())

    def to_dict(self) -> dict:
        e = self.euler
        return {
            'x': self._v[0], 'y': self._v[1], 'z': self._v[2],
            'vx': self._v[3], 'vy': self._v[4], 'vz': self._v[5],
            'qw': self._v[6], 'qx': self._v[7], 'qy': self._v[8], 'qz': self._v[9],
            'p': self._v[10], 'q': self._v[11], 'r': self._v[12],
            'fuel': self._v[13],
            'roll': e[0], 'pitch': e[1], 'yaw': e[2],
            'altitude': self.altitude, 'speed': self.speed,
        }


def state_from_config() -> PhysicsState:
    """Build initial state from config defaults."""
    import config
    v = np.zeros(STATE_DIM)
    v[0] = config.INIT_X
    v[1] = config.INIT_Y
    v[2] = config.INIT_ALTITUDE
    v[3] = config.INIT_VX
    v[4] = config.INIT_VY
    v[5] = config.INIT_VZ
    q = euler_to_quat(0.0, config.INIT_PITCH, 0.0)
    v[6:10] = q
    v[13] = config.MASS_FUEL_INIT
    return PhysicsState(v)
