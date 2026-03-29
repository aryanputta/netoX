"""
Control thread: reads physics state, computes control, writes back to shared.

Uses LoopTimer for deterministic 200 Hz timing with overrun tracking.
Logs all telemetry to TelemetryLogger for replay and post-analysis.
"""

import threading
import numpy as np

from control.pid_controller import CascadedPIDController
from control.rl_controller import RLController, compute_landing_reward
from loop_timer import LoopTimer
from logger import get_logger, TelemetryLogger
import config

log = get_logger(__name__)


class ControlThread(threading.Thread):
    def __init__(self, shared, params, metrics=None, session_name: str = None):
        super().__init__(name="Controller", daemon=True)
        self.shared   = shared
        self.params   = params
        self.metrics  = metrics
        self.pid      = CascadedPIDController(params)
        self.rl       = RLController(params)
        self._target  = np.array([config.TARGET_X, config.TARGET_Y, config.TARGET_Z])
        self._tlog    = TelemetryLogger(session_name)
        self._timer   = LoopTimer(rate_hz=1.0 / config.DT_PHYSICS)

    @property
    def telemetry_path(self):
        return self._tlog.path

    def run(self):
        shared  = self.shared
        dt      = config.DT_PHYSICS
        target  = self._target
        pid     = self.pid
        rl      = self.rl
        sim_t   = 0.0
        timer   = self._timer

        while not shared.shutdown.is_set():
            state     = shared.read_physics()
            ctrl_name = shared.read_controller_name()

            if ctrl_name == 'RL' and rl.is_trained:
                control = rl.compute(state, target, dt)
            else:
                control = pid.compute(state, target, dt)

            shared.update_control(control)

            # Telemetry log (every step → full resolution for replay)
            self._tlog.record(sim_t, state, control, ctrl_name)

            # Metrics
            if self.metrics:
                self.metrics.record(sim_t, state, control)

            # RL episode recording
            if ctrl_name == 'RL':
                reward = compute_landing_reward(
                    state, target, control, landed=(state.altitude < 0.2)
                )
                rl.record_step(state, control, reward, target)

            sim_t += dt

            # Deterministic timing
            overrun = timer.wait()
            if overrun > 5.0:
                log.debug(f"Control loop overrun: {overrun:.1f} ms at T+{sim_t:.2f}s")

        # Flush log on shutdown
        self._tlog.close()
        stats = timer.stats()
        log.info(
            f"Control thread stopped. "
            f"Steps logged: {self._tlog.record_count}  "
            f"Overruns: {stats['overruns']}  "
            f"Max overrun: {stats['max_overrun_ms']:.1f} ms"
        )
