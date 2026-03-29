"""
PID controller with anti-windup and derivative low-pass filter.

Anti-windup: conditional integration — integrator is frozen when the
output is saturated and the error would push it further into saturation.

Derivative filter: first-order low-pass on the derivative term to
suppress high-frequency sensor noise.
"""

import numpy as np


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -np.inf,
        output_max: float =  np.inf,
        integral_limit: float = np.inf,
        deriv_filter_tau: float = 0.01,  # s; 0 = no filter
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_min = output_min
        self.out_max = output_max
        self.ilim    = integral_limit
        self.tau     = deriv_filter_tau

        self._integral  = 0.0
        self._prev_err  = 0.0
        self._d_filt    = 0.0   # filtered derivative

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0
        self._d_filt   = 0.0

    def update(self, error: float, dt: float) -> float:
        # Derivative (with low-pass filter)
        d_raw = (error - self._prev_err) / dt if dt > 1e-9 else 0.0
        if self.tau > 1e-9:
            alpha = dt / (self.tau + dt)
            self._d_filt += alpha * (d_raw - self._d_filt)
        else:
            self._d_filt = d_raw

        raw_out = self.kp * error + self.ki * self._integral + self.kd * self._d_filt
        out     = float(np.clip(raw_out, self.out_min, self.out_max))

        # Conditional integration: don't wind up if saturated in same direction
        saturated = (out == self.out_min and error < 0) or (out == self.out_max and error > 0)
        if not saturated:
            self._integral = float(np.clip(self._integral + error * dt, -self.ilim, self.ilim))

        self._prev_err = error
        return out
