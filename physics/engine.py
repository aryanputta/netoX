"""
Simulation engine: runs physics at fixed dt, publishes state to SharedState.
Runs in its own thread to decouple from networking and control.
"""

import time
import threading
import numpy as np

from physics.state import PhysicsState
from physics.dynamics import rk4_step
import config


class SimulationEngine(threading.Thread):
    def __init__(self, params, shared, dt: float = config.DT_PHYSICS):
        super().__init__(name="SimEngine", daemon=True)
        self.params  = params
        self.shared  = shared
        self.dt      = dt
        self._step   = 0
        self._sim_t  = 0.0

    def run(self):
        shared  = self.shared
        params  = self.params
        dt      = self.dt
        sv      = shared.read_physics().vec.copy()

        loop_period = dt  # real-time: wall time == sim time

        while not shared.shutdown.is_set():
            t0 = time.perf_counter()

            # Pull latest control from shared 
            ctrl = shared.read_control()
            wind = shared.read_wind()

            # Failure injection
            eng_eff = 1.0
            if config.ENGINE_DEGRADE_PCT > 0 and self._sim_t > config.ENGINE_DEGRADE_START:
                eng_eff = 1.0 - config.ENGINE_DEGRADE_PCT

            sv = rk4_step(sv, ctrl, params, dt, wind, eng_eff)

            state = PhysicsState(sv)
            shared.update_physics(state, self._sim_t)

            self._step  += 1
            self._sim_t += dt

            # Real-time pacing: sleep the remainder of dt
            elapsed = time.perf_counter() - t0
            sleep_t = loop_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
