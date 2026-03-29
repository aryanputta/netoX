"""
Deterministic loop timer for real-time threads.

Uses monotonic clock and sleep-with-drift-compensation to maintain
a precise loop rate even when individual iterations run long.

Usage:
    timer = LoopTimer(rate_hz=200.0)
    while running:
        do_work()
        overrun_ms = timer.wait()   # sleeps remainder of period
        if overrun_ms > 5.0:
            log.warning(f"Control loop overrun: {overrun_ms:.1f} ms")
"""

import time


class LoopTimer:
    def __init__(self, rate_hz: float):
        self.period  = 1.0 / rate_hz
        self._next   = time.perf_counter()
        self.overruns = 0
        self.max_overrun_ms = 0.0

    def wait(self) -> float:
        """
        Sleep until the next scheduled tick.
        Returns overrun in ms (0 if on time, positive if late).
        """
        self._next += self.period
        now = time.perf_counter()
        delay = self._next - now
        overrun = 0.0

        if delay > 0:
            time.sleep(delay)
        else:
            # We're already late — snap next tick to now to avoid drift cascade
            overrun = -delay * 1000.0
            self._next = time.perf_counter()
            self.overruns += 1
            if overrun > self.max_overrun_ms:
                self.max_overrun_ms = overrun

        return overrun

    def reset(self):
        self._next   = time.perf_counter()
        self.overruns = 0
        self.max_overrun_ms = 0.0

    def stats(self) -> dict:
        return {
            'period_ms':        self.period * 1000.0,
            'overruns':         self.overruns,
            'max_overrun_ms':   self.max_overrun_ms,
        }
