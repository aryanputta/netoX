"""
Structured telemetry logger with rotating file output and console handler.

Writes two streams:
  1. Structured JSON-lines telemetry log (all state fields, timestamped)
  2. Human-readable console + rotating text log

Usage:
    from logger import get_logger, TelemetryLogger
    log = get_logger(__name__)
    tlog = TelemetryLogger("flight_001")
    tlog.record(sim_t, state, control, ctrl_name)
"""

import logging
import logging.handlers
import json
import time
import threading
import numpy as np
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Standard logger with console + rotating file handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    # Console (INFO and above)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file (DEBUG and above, 5 MB × 3 files)
    fh = logging.handlers.RotatingFileHandler(
        LOG_DIR / 'netobot.log', maxBytes=5*1024*1024, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class TelemetryLogger:
    """
    Writes one JSON-lines record per physics step to a timestamped log file.
    Thread-safe; writes happen in a background thread to avoid blocking the
    200 Hz control loop.
    """

    def __init__(self, session_name: str | None = None):
        ts = time.strftime('%Y%m%d_%H%M%S')
        name = session_name or f"session_{ts}"
        self._path = LOG_DIR / f"telemetry_{name}.jsonl"
        self._queue: list[str] = []
        self._lock  = threading.Lock()
        self._flush_event = threading.Event()
        self._shutdown    = threading.Event()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        self._count = 0

    def record(self, sim_t: float, state, control: np.ndarray, ctrl_name: str):
        """Enqueue one telemetry record (non-blocking)."""
        e = state.euler
        rec = {
            't':     round(sim_t, 4),
            'x':     round(float(state.position[0]), 4),
            'y':     round(float(state.position[1]), 4),
            'z':     round(float(state.position[2]), 4),
            'vx':    round(float(state.velocity[0]), 4),
            'vy':    round(float(state.velocity[1]), 4),
            'vz':    round(float(state.velocity[2]), 4),
            'roll':  round(float(np.degrees(e[0])), 3),
            'pitch': round(float(np.degrees(e[1])), 3),
            'yaw':   round(float(np.degrees(e[2])), 3),
            'p':     round(float(state.ang_vel[0]), 4),
            'q':     round(float(state.ang_vel[1]), 4),
            'r':     round(float(state.ang_vel[2]), 4),
            'fuel':  round(float(state.fuel_mass), 4),
            'thr':   round(float(control[0]), 4),
            'tx':    round(float(control[1]), 4),
            'ty':    round(float(control[2]), 4),
            'tz':    round(float(control[3]), 4),
            'ctrl':  ctrl_name,
        }
        line = json.dumps(rec, separators=(',', ':'))
        with self._lock:
            self._queue.append(line)
            self._count += 1
        if self._count % 200 == 0:   # flush every 1 s at 200 Hz
            self._flush_event.set()

    def flush(self):
        """Force immediate flush."""
        self._flush_event.set()
        time.sleep(0.01)

    def close(self):
        self._shutdown.set()
        self._flush_event.set()
        self._writer.join(timeout=2.0)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def record_count(self) -> int:
        return self._count

    def _write_loop(self):
        with open(self._path, 'w') as f:
            while not self._shutdown.is_set():
                self._flush_event.wait(timeout=1.0)
                self._flush_event.clear()
                with self._lock:
                    lines, self._queue = self._queue, []
                if lines:
                    f.write('\n'.join(lines) + '\n')
                    f.flush()
