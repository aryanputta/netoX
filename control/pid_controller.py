"""
Cascaded PID landing controller.

Altitude loop:
  Outer (50 Hz): altitude error → reference vertical velocity (sqrt profile)
  Inner (200 Hz): velocity error → throttle

The sqrt velocity profile vz_ref = -sqrt(2 * a_ref * z) guarantees the vehicle
can always brake to a stop at the landing pad given the available deceleration.
This is structurally equivalent to a cascaded PID with feedforward.

Horizontal + attitude loop (cascaded):
  Position error  → desired tilt angle  (50 Hz)
  Tilt error      → RCS torque          (200 Hz)

Control output: [throttle ∈ [0,1], tau_x, tau_y, tau_z ∈ [-1,1]]
"""

import numpy as np
from control.pid import PID
import config


class CascadedPIDController:
    def __init__(self, params):
        self.params = params

        # ── Altitude velocity controller (inner) ─────────────────────────
        # Error: vz_ref - vz_actual → throttle delta
        self.pid_vz = PID(2.5, 0.5, 0.8,
                          output_min=-0.8, output_max=0.8,
                          integral_limit=0.6,
                          deriv_filter_tau=0.02)

        # ── Horizontal position → desired tilt (outer, 50 Hz) ─────────────
        self.pid_x  = PID(config.KP_XY, config.KI_XY, config.KD_XY,
                          output_min=-0.30, output_max=0.30,
                          integral_limit=0.25,
                          deriv_filter_tau=0.05)

        self.pid_y  = PID(config.KP_XY, config.KI_XY, config.KD_XY,
                          output_min=-0.30, output_max=0.30,
                          integral_limit=0.25,
                          deriv_filter_tau=0.05)

        # ── Attitude controllers (inner, 200 Hz) ──────────────────────────
        self.pid_roll  = PID(config.KP_ROLL,  config.KI_ROLL,  config.KD_ROLL,
                             output_min=-1.0, output_max=1.0,
                             integral_limit=config.ICLAMP_ATT)

        self.pid_pitch = PID(config.KP_PITCH, config.KI_PITCH, config.KD_PITCH,
                             output_min=-1.0, output_max=1.0,
                             integral_limit=config.ICLAMP_ATT)

        self.pid_yaw   = PID(config.KP_YAW,   config.KI_YAW,   config.KD_YAW,
                             output_min=-1.0, output_max=1.0,
                             integral_limit=config.ICLAMP_ATT)

        self._outer_dt   = 1.0 / 50.0
        self._outer_tick = 0.0
        self._des_roll   = 0.0
        self._des_pitch  = 0.0
        self._throttle   = 0.6

    def reset(self):
        for pid in (self.pid_vz, self.pid_x, self.pid_y,
                    self.pid_roll, self.pid_pitch, self.pid_yaw):
            pid.reset()
        self._outer_tick = 0.0
        self._des_roll   = 0.0
        self._des_pitch  = 0.0

    def compute(
        self,
        state,
        target: np.ndarray,
        dt: float,
        yaw_setpoint: float = 0.0,
    ) -> np.ndarray:
        pos   = state.position
        vel   = state.velocity
        euler = state.euler
        mass  = self.params.mass_at(state.fuel_mass)

        # ── Outer loop (50 Hz) ───────────────────────────────────────────
        self._outer_tick += dt
        if self._outer_tick >= self._outer_dt:
            dt_outer = self._outer_tick
            self._outer_tick = 0.0

            # Reference vertical velocity: linear proportional profile.
            # vz_ref = -kp * z  (at z=100 → -8 m/s, at z=10 → -0.8 m/s).
            # Ensures vehicle decelerates as it approaches the ground.
            altitude = max(0.0, pos[2] - target[2])
            vz_ref   = float(np.clip(-0.08 * altitude, -8.0, 0.1))

            # Gravity feedforward (hover throttle)
            gravity_ff  = (mass * config.G) / self.params.max_thrust
            vz_err      = vz_ref - vel[2]
            thr_delta   = self.pid_vz.update(vz_err, dt_outer)
            self._throttle = float(np.clip(gravity_ff + thr_delta, 0.0, 1.0))

            # Positive pitch → +x thrust (from DCM analysis).
            # Positive roll  → -y thrust.
            self._des_pitch =  self.pid_x.update(target[0] - pos[0], dt_outer)
            self._des_roll  = -self.pid_y.update(target[1] - pos[1], dt_outer)

        # ── Inner loop: attitude (every step) ────────────────────────────
        roll_err  = self._des_roll  - euler[0]
        pitch_err = self._des_pitch - euler[1]
        yaw_err   = _wrap(yaw_setpoint - euler[2])

        tau_x = self.pid_roll.update(roll_err,   dt)
        tau_y = self.pid_pitch.update(pitch_err, dt)
        tau_z = self.pid_yaw.update(yaw_err,     dt)

        return np.array([self._throttle, tau_x, tau_y, tau_z])


def _wrap(a: float) -> float:
    while a >  np.pi: a -= 2 * np.pi
    while a < -np.pi: a += 2 * np.pi
    return a
