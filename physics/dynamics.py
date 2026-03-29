"""
6-DOF equations of motion for a rocket-style lander.

Forces (world frame):
  F_gravity  = [0, 0, -m*g]
  F_thrust   = R @ [0, 0, throttle * T_max]    body +z through engine
  F_drag     = -0.5 * rho * Cd * A * |v| * v   quadratic drag

Torques (body frame):
  tau_rcs    = [tx, ty, tz]                     RCS thrusters
  tau_gyro   = -omega × (I @ omega)             already in Euler's eq

Fuel burn:
  dm/dt = -throttle * T_max * burn_rate         (= throttle * T_max / (Isp*g))

Quaternion kinematics:
  dq/dt = 0.5 * Omega(omega) @ q
"""

import numpy as np
from physics.quaternion import quat_normalize, omega_matrix
from physics.state import STATE_DIM
import config


def derivatives(
    sv: np.ndarray,
    control: np.ndarray,
    params,
    wind_world: np.ndarray = None,
    engine_effectiveness: float = 1.0,
) -> np.ndarray:
    """
    Compute d(state)/dt.

    Args:
        sv:                   state vector (14,)
        control:              [throttle, tau_x, tau_y, tau_z]
        params:               VehicleParams
        wind_world:           wind velocity in world frame (m/s), optional
        engine_effectiveness: 0–1 multiplier for engine degradation failure mode

    Returns:
        dsv/dt  shape (14,)
    """
    pos   = sv[0:3]
    vel   = sv[3:6]
    q     = sv[6:10]
    omega = sv[10:13]
    fuel  = sv[13]

    throttle = float(np.clip(control[0], 0.0, 1.0))
    tau_rcs  = np.clip(control[1:4], -1.0, 1.0) * params.max_rcs_torque

    mass = params.mass_at(fuel)

    # Direction cosine matrix: body → world
    R = np.array([
        [1 - 2*(q[2]**2 + q[3]**2),     2*(q[1]*q[2] - q[0]*q[3]),     2*(q[1]*q[3] + q[0]*q[2])],
        [    2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2),     2*(q[2]*q[3] - q[0]*q[1])],
        [    2*(q[1]*q[3] - q[0]*q[2]),     2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)],
    ])

    # ── Thrust in world frame ──────────────────────────────────────────────
    thrust_body  = np.array([0.0, 0.0, throttle * params.max_thrust * engine_effectiveness])
    thrust_world = R @ thrust_body

    # ── Drag in world frame ────────────────────────────────────────────────
    vel_rel = vel.copy()
    if wind_world is not None:
        vel_rel = vel - wind_world
    speed_rel = np.linalg.norm(vel_rel)
    if speed_rel > 1e-3:
        drag_world = -0.5 * config.RHO_AIR * params.cd_body * params.reference_area \
                     * speed_rel * vel_rel
    else:
        drag_world = np.zeros(3)

    # ── Gravity ───────────────────────────────────────────────────────────
    gravity_world = np.array([0.0, 0.0, -config.G * mass])

    # ── Translational EOM ─────────────────────────────────────────────────
    dpos = vel
    dvel = (thrust_world + drag_world + gravity_world) / mass

    # ── Quaternion kinematics ─────────────────────────────────────────────
    dq = omega_matrix(omega) @ q   # 0.5 already inside omega_matrix

    # ── Rotational EOM (Euler's equations in body frame) ──────────────────
    I   = params.inertia_tensor
    Iw  = I @ omega
    domega = params.inertia_inv @ (tau_rcs - np.cross(omega, Iw))

    # ── Fuel burn ─────────────────────────────────────────────────────────
    dfuel = -throttle * params.max_thrust * params.fuel_burn_rate
    if fuel <= 0.0:
        dfuel = 0.0  # tank empty

    dsv = np.empty(STATE_DIM)
    dsv[0:3]  = dpos
    dsv[3:6]  = dvel
    dsv[6:10] = dq
    dsv[10:13] = domega
    dsv[13]   = dfuel
    return dsv


def rk4_step(
    sv: np.ndarray,
    control: np.ndarray,
    params,
    dt: float,
    wind_world: np.ndarray = None,
    engine_effectiveness: float = 1.0,
) -> np.ndarray:
    """
    Classic RK4 integrator. Quaternion is re-normalised after accumulation
    to prevent drift accumulation from the non-linear quaternion kinematics.
    """
    def f(s):
        return derivatives(s, control, params, wind_world, engine_effectiveness)

    k1 = f(sv)
    k2 = f(sv + 0.5 * dt * k1)
    k3 = f(sv + 0.5 * dt * k2)
    k4 = f(sv +       dt * k3)

    sv_next = sv + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Normalise quaternion
    sv_next[6:10] = quat_normalize(sv_next[6:10])

    # Clamp fuel
    sv_next[13] = max(0.0, sv_next[13])

    # Ground contact: stop at z=0
    if sv_next[2] <= 0.0:
        sv_next[2]  = 0.0
        sv_next[3:6] = np.zeros(3)
        sv_next[10:13] = np.zeros(3)

    return sv_next
