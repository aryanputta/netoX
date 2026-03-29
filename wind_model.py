"""
Stochastic wind disturbance model running as a background thread.

Models:
  - Steady mean wind in configurable direction
  - Turbulent gusts (Dryden model approximation via coloured noise)
  - Updates shared.wind at 20 Hz
"""

import threading
import time
import numpy as np
import config


class WindModel(threading.Thread):
    def __init__(self, shared, rng: np.random.Generator = None):
        super().__init__(name="WindModel", daemon=True)
        self.shared   = shared
        self.rng      = rng or np.random.default_rng(1)
        self._wind    = np.zeros(3)
        self._gust    = np.zeros(3)
        self._dt      = 0.05   # 20 Hz update

    def run(self):
        tau_gust = 2.0   # gust time constant (s)
        alpha    = self._dt / (tau_gust + self._dt)

        while not self.shared.shutdown.is_set():
            t0 = time.perf_counter()

            mean  = config.WIND_SPEED_MEAN
            std   = config.WIND_SPEED_STD
            direc = config.WIND_DIRECTION

            # Steady component
            steady = np.array([
                mean * np.cos(direc),
                mean * np.sin(direc),
                0.0,
            ])

            # First-order Markov gust (coloured noise)
            white = self.rng.normal(0, std * np.sqrt(2.0 / tau_gust * self._dt), 3)
            self._gust = (1 - alpha) * self._gust + alpha * white

            self._wind = steady + self._gust
            self.shared.update_wind(self._wind)

            elapsed = time.perf_counter() - t0
            sleep_t = self._dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
