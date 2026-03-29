"""
Quantitative performance metrics for landing controller evaluation.

Metrics:
  ISE   — Integral Squared Error on position (lower = better)
  ITAE  — Integral Time-Absolute Error (penalises late errors)
  IAE   — Integral Absolute Error
  touchdown_speed     — m/s at ground contact (< 1.5 m/s = success)
  touchdown_pos_err   — lateral distance from target (m)
  fuel_used           — kg of propellant consumed
  settling_time       — s to stay within 5% of target
  max_tilt            — deg, max tilt angle during descent
  control_effort      — mean |action|² (efficiency proxy)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import config


@dataclass
class EpisodeReport:
    controller:         str
    duration:           float      # s
    # Tracking error integrals
    ISE:                float
    ITAE:               float
    IAE:                float
    # Settling
    settling_time:      float      # s  (NaN if never settled)
    max_tilt_deg:       float
    # Landing quality
    touchdown_speed:    float      # m/s
    touchdown_pos_err:  float      # m
    landed:             bool
    # Resources
    fuel_used:          float      # kg
    control_effort:     float      # mean ||action||²
    # Network
    mean_latency_ms:    float
    packet_loss_pct:    float
    packets_dropped:    int

    def success_rating(self) -> str:
        if not self.landed:                 return "CRASH"
        if self.touchdown_speed > 3.0:      return "HARD LANDING"
        if self.touchdown_pos_err > 5.0:    return "OFF-TARGET"
        if self.touchdown_speed < 1.0 and self.touchdown_pos_err < 1.5:
            return "PRECISION"
        return "NOMINAL"

    def summary_table(self) -> str:
        s = self.success_rating()
        lines = [
            f"  Controller:        {self.controller:<10}  [{s}]",
            f"  Duration:          {self.duration:.1f} s",
            f"  ISE (position):    {self.ISE:.2f} m²·s",
            f"  ITAE:              {self.ITAE:.2f} m·s²",
            f"  Settling time:     {self.settling_time:.2f} s" if not np.isnan(self.settling_time) else "  Settling time:     never",
            f"  Max tilt:          {self.max_tilt_deg:.1f} °",
            f"  Touchdown speed:   {self.touchdown_speed:.2f} m/s",
            f"  Touchdown error:   {self.touchdown_pos_err:.2f} m",
            f"  Fuel used:         {self.fuel_used:.3f} kg",
            f"  Control effort:    {self.control_effort:.4f}",
            f"  Net latency (μ):   {self.mean_latency_ms:.1f} ms",
            f"  Packet loss:       {self.packet_loss_pct:.1f} %",
        ]
        return "\n".join(lines)


class MetricsCollector:
    def __init__(self, target: np.ndarray = None):
        self.target = target if target is not None else np.array([
            config.TARGET_X, config.TARGET_Y, config.TARGET_Z
        ])
        self._t:       List[float]       = []
        self._pos_err: List[float]       = []
        self._tilts:   List[float]       = []
        self._actions: List[np.ndarray]  = []
        self._landed   = False
        self._td_speed = 0.0
        self._td_err   = 0.0
        self._fuel_start = config.MASS_FUEL_INIT
        self._fuel_end   = config.MASS_FUEL_INIT
        self._settling_t = np.nan
        self._in_tube    = False
        self._tube_entry = np.nan
        self._tube_count = 0
        self._tube_thresh = 5.0  # m (5% of 100 m initial altitude)

    def record(self, t: float, state, action: np.ndarray):
        pos_err = float(np.linalg.norm(state.position - self.target))
        tilt    = float(np.degrees(np.linalg.norm(state.euler[:2])))
        self._t.append(t)
        self._pos_err.append(pos_err)
        self._tilts.append(tilt)
        self._actions.append(action.copy())
        self._fuel_end = state.fuel_mass

        # Touchdown detection
        if state.altitude < 0.2 and not self._landed:
            self._landed  = True
            self._td_speed = state.speed
            self._td_err   = float(np.linalg.norm(state.position[:2] - self.target[:2]))

        # Settling: must stay inside tube for 2+ seconds
        if pos_err < self._tube_thresh:
            if not self._in_tube:
                self._in_tube    = True
                self._tube_entry = t
            self._tube_count += 1
            if self._tube_count > int(2.0 / config.DT_PHYSICS) and np.isnan(self._settling_t):
                self._settling_t = self._tube_entry
        else:
            self._in_tube    = False
            self._tube_count = 0

    def finalize(
        self,
        controller_name: str,
        net_latency_ms:  float = 0.0,
        net_loss_pct:    float = 0.0,
        pkts_dropped:    int   = 0,
    ) -> EpisodeReport:
        t  = np.array(self._t)
        pe = np.array(self._pos_err)
        if len(t) < 2:
            dt = config.DT_PHYSICS
        else:
            dt = float(np.mean(np.diff(t)))

        ISE  = float(np.trapz(pe**2, t))
        IAE  = float(np.trapz(pe,    t))
        ITAE = float(np.trapz(t * pe, t))

        actions = np.array(self._actions) if self._actions else np.zeros((1, 4))
        effort  = float(np.mean(np.sum(actions**2, axis=1)))
        dur     = float(t[-1] - t[0]) if len(t) > 1 else 0.0

        return EpisodeReport(
            controller        = controller_name,
            duration          = dur,
            ISE               = ISE,
            ITAE              = ITAE,
            IAE               = IAE,
            settling_time     = self._settling_t,
            max_tilt_deg      = float(np.max(self._tilts)) if self._tilts else 0.0,
            touchdown_speed   = self._td_speed,
            touchdown_pos_err = self._td_err,
            landed            = self._landed,
            fuel_used         = max(0.0, self._fuel_start - self._fuel_end),
            control_effort    = effort,
            mean_latency_ms   = net_latency_ms,
            packet_loss_pct   = net_loss_pct,
            packets_dropped   = pkts_dropped,
        )

    def reset(self):
        self.__init__(self.target)


def print_comparison(reports: List[EpisodeReport]):
    print("\n" + "═" * 60)
    print("  CONTROLLER COMPARISON")
    print("═" * 60)
    for r in reports:
        print()
        print(r.summary_table())
    print()
    if len(reports) >= 2:
        a, b = reports[0], reports[1]
        print("─" * 60)
        print(f"  ISE improvement  ({b.controller} vs {a.controller}): "
              f"{100*(a.ISE - b.ISE)/max(a.ISE,1e-6):+.1f}%")
        print(f"  Fuel saved:      {a.fuel_used - b.fuel_used:+.3f} kg")
        print(f"  Touch. speed Δ:  {a.touchdown_speed - b.touchdown_speed:+.2f} m/s")
    print("═" * 60)
