"""
Thread-safe shared state bus connecting all system components.

All cross-thread data passes through this class.
Uses RLock for compound read/write and separate deques per metric stream.
"""

import threading
import time
import numpy as np
from collections import deque
from physics.state import PhysicsState, state_from_config
import config


class SharedState:
    def __init__(self):
        self._lock = threading.RLock()
        self.shutdown = threading.Event()

        # Physics state (flight computer writes, everything else reads)
        self._physics   = state_from_config()
        self._sim_time  = 0.0
        self._control   = np.array([0.6, 0.0, 0.0, 0.0])  # throttle, tx, ty, tz
        self._ctrl_name = 'PID'
        self._wind      = np.zeros(3)

        # Ground-station view (affected by network latency/loss)
        self._ground_pkt   = None
        self._ground_time  = 0.0   # wall time when last packet arrived

        # Network statistics
        self._latency_hist  = deque(maxlen=200)
        self._pkts_recv     = 0
        self._pkts_dropped  = 0
        self._ise_history   = {'PID': deque(maxlen=500), 'RL': deque(maxlen=500)}
        self._running_ise   = 0.0
        self._ise_last_ctrl = 'PID'

    # ── Physics ───────────────────────────────────────────────────────────────

    def update_physics(self, state: PhysicsState, sim_time: float):
        with self._lock:
            self._physics  = state
            self._sim_time = sim_time
            # Update running ISE
            pos_err = np.linalg.norm(state.position - np.array([
                config.TARGET_X, config.TARGET_Y, config.TARGET_Z
            ]))
            dt = config.DT_PHYSICS
            self._running_ise += pos_err**2 * dt
            self._ise_history[self._ise_last_ctrl].append(self._running_ise)

    def read_physics(self) -> PhysicsState:
        with self._lock:
            return self._physics

    def read_sim_time(self) -> float:
        with self._lock:
            return self._sim_time

    # ── Control ───────────────────────────────────────────────────────────────

    def update_control(self, control: np.ndarray):
        with self._lock:
            self._control = control.copy()

    def read_control(self) -> np.ndarray:
        with self._lock:
            return self._control.copy()

    def set_controller(self, name: str):
        with self._lock:
            self._ctrl_name     = name
            self._ise_last_ctrl = name
            self._running_ise   = 0.0  # reset ISE when switching

    def read_controller_name(self) -> str:
        with self._lock:
            return self._ctrl_name

    # ── Wind ──────────────────────────────────────────────────────────────────

    def update_wind(self, wind: np.ndarray):
        with self._lock:
            self._wind = wind.copy()

    def read_wind(self) -> np.ndarray:
        with self._lock:
            return self._wind.copy()

    # ── Networking ────────────────────────────────────────────────────────────

    def update_ground_telemetry(self, pkt, recv_wall: float, latency: float):
        with self._lock:
            self._ground_pkt  = pkt
            self._ground_time = recv_wall
            self._pkts_recv  += 1

    def record_net_latency(self, latency_ms: float):
        with self._lock:
            self._latency_hist.append(latency_ms)

    def record_net_drop(self):
        with self._lock:
            self._pkts_dropped += 1

    def read_net_stats(self) -> dict:
        with self._lock:
            lat_list = list(self._latency_hist)
            total    = self._pkts_recv + self._pkts_dropped
            loss_pct = 100.0 * self._pkts_dropped / total if total > 0 else 0.0
            mean_lat = float(np.mean(lat_list)) if lat_list else 0.0
            ise_hist = {
                'PID': list(self._ise_history['PID']),
                'RL':  list(self._ise_history['RL']),
            }
            return {
                'latency_hist':    lat_list,
                'mean_latency_ms': mean_lat,
                'loss_pct':        loss_pct,
                'pkts_recv':       self._pkts_recv,
                'pkts_dropped':    self._pkts_dropped,
                'ise_history':     ise_hist,
            }
