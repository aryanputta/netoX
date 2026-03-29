"""Unit-quaternion math. Convention: q = [w, x, y, z]."""

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Direction cosine matrix: rotates vector from body to world frame."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """ZYX Euler angles (roll, pitch, yaw) in radians."""
    w, x, y, z = q
    # roll (x-axis)
    sinr = 2.0 * (w*x + y*z)
    cosr = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(sinr, cosr)
    # pitch (y-axis)
    sinp = 2.0 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z-axis)
    siny = 2.0 * (w*z + x*y)
    cosy = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles → unit quaternion."""
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    return quat_normalize(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]))


def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """4×4 Omega matrix for quaternion kinematics: dq/dt = 0.5 * Omega(ω) @ q."""
    p, q, r = omega
    return 0.5 * np.array([
        [ 0,  -p,  -q,  -r],
        [ p,   0,   r,  -q],
        [ q,  -r,   0,   p],
        [ r,   q,  -p,   0],
    ])
