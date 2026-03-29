"""
Replay system for post-flight analysis.

Reads a telemetry JSONL log and replays it through:
  - The visualization dashboard (at configurable speed)
  - The metrics engine (to re-compute KPIs from a saved flight)
  - Optional: re-runs control decisions to compare PID vs RL on the same trajectory

Usage:
    python replay.py logs/telemetry_session_20260328_120000.jsonl [--speed 2.0]
    python replay.py logs/telemetry_*.jsonl --compare-controllers
    python replay.py logs/telemetry_*.jsonl --metrics-only
"""

import argparse
import json
import time
import sys
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Iterator

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from physics.state import PhysicsState
from physics.quaternion import euler_to_quat
from evaluation.metrics import MetricsCollector, print_comparison
import config


@dataclass
class ReplayFrame:
    t:       float
    x:       float
    y:       float
    z:       float
    vx:      float
    vy:      float
    vz:      float
    roll:    float   # deg
    pitch:   float   # deg
    yaw:     float   # deg
    p:       float   # rad/s
    q_rate:  float
    r_rate:  float
    fuel:    float
    thr:     float
    tx:      float
    ty:      float
    tz:      float
    ctrl:    str


def load_log(path: str) -> List[ReplayFrame]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frames.append(ReplayFrame(
                t=d['t'],   x=d['x'],   y=d['y'],   z=d['z'],
                vx=d['vx'], vy=d['vy'], vz=d['vz'],
                roll=d['roll'], pitch=d['pitch'], yaw=d['yaw'],
                p=d['p'],   q_rate=d['q'], r_rate=d['r'],
                fuel=d['fuel'],
                thr=d['thr'], tx=d['tx'], ty=d['ty'], tz=d['tz'],
                ctrl=d['ctrl'],
            ))
    return frames


def frame_to_physics_state(f: ReplayFrame) -> PhysicsState:
    q = euler_to_quat(np.radians(f.roll), np.radians(f.pitch), np.radians(f.yaw))
    sv = np.array([
        f.x, f.y, f.z,
        f.vx, f.vy, f.vz,
        *q,
        f.p, f.q_rate, f.r_rate,
        f.fuel,
    ], dtype=float)
    return PhysicsState(sv)


class ReplayPlayer:
    """Streams frames at a controllable playback rate."""

    def __init__(self, frames: List[ReplayFrame], speed: float = 1.0):
        self.frames  = frames
        self.speed   = speed
        self._idx    = 0
        self._paused = False

    def __iter__(self) -> Iterator[ReplayFrame]:
        if not self.frames:
            return
        t_prev = self.frames[0].t
        for frame in self.frames:
            if self._paused:
                time.sleep(0.05)
                continue
            dt = (frame.t - t_prev) / self.speed
            if dt > 0:
                time.sleep(max(0.0, dt))
            t_prev = frame.t
            self._idx += 1
            yield frame

    def pause(self):   self._paused = True
    def resume(self):  self._paused = False

    @property
    def progress(self) -> float:
        if not self.frames:
            return 1.0
        return self._idx / len(self.frames)


class ReplayDashboard:
    """Lightweight dashboard for replay (no networking, simpler layout)."""

    def __init__(self, frames: List[ReplayFrame], title: str, speed: float = 1.0):
        self.frames = frames
        self.speed  = speed
        self.title  = title
        N = len(frames)

        # Pre-extract arrays for fast plotting
        self.t     = np.array([f.t   for f in frames])
        self.x     = np.array([f.x   for f in frames])
        self.y     = np.array([f.y   for f in frames])
        self.z     = np.array([f.z   for f in frames])
        self.speed_arr = np.array([
            np.sqrt(f.vx**2 + f.vy**2 + f.vz**2) for f in frames
        ])
        self.roll  = np.array([f.roll  for f in frames])
        self.pitch = np.array([f.pitch for f in frames])
        self.thr   = np.array([f.thr   for f in frames])
        self.fuel  = np.array([f.fuel  for f in frames])
        self._cursor = 0

    def run(self):
        matplotlib.rcParams.update({
            'axes.facecolor': '#0d1117', 'figure.facecolor': '#0d1117',
            'axes.edgecolor': '#30363d', 'text.color': '#e6edf3',
            'axes.labelcolor': '#e6edf3', 'xtick.color': '#8b949e',
            'ytick.color': '#8b949e', 'grid.color': '#21262d',
            'grid.linestyle': '--', 'grid.alpha': 0.6,
            'font.family': 'monospace',
        })

        fig = plt.figure(figsize=(14, 8), facecolor='#0d1117')
        fig.canvas.manager.set_window_title(f'REPLAY — {self.title}')

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               left=0.07, right=0.97, top=0.90, bottom=0.10,
                               wspace=0.4, hspace=0.5)

        ax3d  = fig.add_subplot(gs[:, 0], projection='3d')
        ax_alt = fig.add_subplot(gs[0, 1])
        ax_att = fig.add_subplot(gs[0, 2])
        ax_thr = fig.add_subplot(gs[1, 1])
        ax_spd = ax_alt.twinx()
        ax_fuel = ax_thr.twinx()

        for ax, ttl in [
            (ax_alt, 'ALTITUDE & SPEED'),
            (ax_att, 'EULER ANGLES'),
            (ax_thr, 'THROTTLE & FUEL'),
        ]:
            ax.set_title(ttl, fontsize=7, color='#58a6ff', pad=4)
            ax.grid(True)
            ax.tick_params(labelsize=6)

        # Full trajectory on 3D plot (static background)
        ax3d.set_facecolor('#0d1117')
        ax3d.set_title('3D TRAJECTORY (REPLAY)', fontsize=7, color='#58a6ff')
        ax3d.plot(self.x, self.y, self.z, color='#30363d', lw=1, alpha=0.5, label='full')
        ax3d.scatter([0], [0], [0], c='#ff4444', s=80, marker='X', zorder=10, label='Target')

        cursor_pt = ax3d.scatter([self.x[0]], [self.y[0]], [self.z[0]],
                                  c='#f0883e', s=80, zorder=11)

        progress_txt = fig.text(0.5, 0.94, '', ha='center', fontsize=9,
                                color='#e6edf3', fontfamily='monospace')

        n_pts = len(self.frames)
        step  = max(1, n_pts // 500)   # plot at most ~500 update points

        def _update(i):
            c = min(i * step, n_pts - 1)
            self._cursor = c

            # Advance 3D cursor
            cursor_pt._offsets3d = ([self.x[c]], [self.y[c]], [self.z[c]])

            # Time-series up to cursor
            ax_alt.cla(); ax_spd.cla()
            ax_alt.set_title('ALTITUDE & SPEED', fontsize=7, color='#58a6ff', pad=4)
            ax_alt.plot(self.t[:c], self.z[:c],         color='#58a6ff', lw=1)
            ax_spd.plot(self.t[:c], self.speed_arr[:c], color='#f0883e', lw=1, alpha=0.8)
            ax_alt.set_ylabel('Alt (m)', fontsize=7, color='#58a6ff')
            ax_spd.set_ylabel('Speed (m/s)', fontsize=7, color='#f0883e')
            ax_alt.grid(True); ax_alt.tick_params(labelsize=6)

            ax_att.cla()
            ax_att.set_title('EULER ANGLES', fontsize=7, color='#58a6ff', pad=4)
            ax_att.plot(self.t[:c], self.roll[:c],  color='#ff6e6e', lw=1, label='Roll')
            ax_att.plot(self.t[:c], self.pitch[:c], color='#6eff6e', lw=1, label='Pitch')
            ax_att.legend(fontsize=6); ax_att.grid(True); ax_att.tick_params(labelsize=6)

            ax_thr.cla(); ax_fuel.cla()
            ax_thr.set_title('THROTTLE & FUEL', fontsize=7, color='#58a6ff', pad=4)
            ax_thr.plot(self.t[:c],  self.thr[:c],  color='#58a6ff', lw=1)
            ax_fuel.plot(self.t[:c], self.fuel[:c], color='#f0883e', lw=1, alpha=0.8)
            ax_thr.set_ylim(0, 1.05)
            ax_thr.set_ylabel('Throttle', fontsize=7); ax_thr.grid(True)
            ax_thr.tick_params(labelsize=6); ax_fuel.tick_params(labelsize=6, colors='#f0883e')

            f = self.frames[c]
            progress_txt.set_text(
                f"REPLAY {self.title}  T+{f.t:.1f}s  "
                f"Alt={f.z:.1f}m  Speed={np.sqrt(f.vx**2+f.vy**2+f.vz**2):.2f}m/s  "
                f"Fuel={f.fuel:.2f}kg  [{f.ctrl}]  "
                f"{100*c//n_pts}%"
            )

        from matplotlib.animation import FuncAnimation
        n_frames = (n_pts + step - 1) // step
        interval = max(20, int(step * (self.t[-1] / n_pts) * 1000 / self.speed))
        anim = FuncAnimation(fig, _update, frames=n_frames,
                              interval=interval, blit=False, cache_frame_data=False)
        plt.show()


def compute_metrics_from_log(frames: List[ReplayFrame], ctrl_name: str = None) -> None:
    """Re-compute and print performance metrics from a replay log."""
    target = np.array([config.TARGET_X, config.TARGET_Y, config.TARGET_Z])
    mc = MetricsCollector(target)
    ctrl = ctrl_name or (frames[0].ctrl if frames else 'UNKNOWN')
    for f in frames:
        state  = frame_to_physics_state(f)
        action = np.array([f.thr, f.tx, f.ty, f.tz])
        mc.record(f.t, state, action)
    report = mc.finalize(ctrl)
    print(report.summary_table())
    print(f"\n  Result: {report.success_rating()}")


def main():
    p = argparse.ArgumentParser(description='NetoBot Replay System')
    p.add_argument('log', nargs='+',  help='JSONL log file(s)')
    p.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    p.add_argument('--metrics-only', action='store_true', help='Print metrics, no visualization')
    p.add_argument('--compare-controllers', action='store_true',
                   help='Compare metrics across multiple log files')
    args = p.parse_args()

    log_files = [Path(f) for f in args.log]
    for lf in log_files:
        if not lf.exists():
            print(f"ERROR: log file not found: {lf}", file=sys.stderr)
            sys.exit(1)

    if args.compare_controllers:
        reports = []
        for lf in log_files:
            frames = load_log(str(lf))
            target = np.array([config.TARGET_X, config.TARGET_Y, config.TARGET_Z])
            mc = MetricsCollector(target)
            ctrl = frames[0].ctrl if frames else 'UNKNOWN'
            for f in frames:
                mc.record(f.t, frame_to_physics_state(f), np.array([f.thr, f.tx, f.ty, f.tz]))
            reports.append(mc.finalize(f'{ctrl} ({lf.stem})'))
        print_comparison(reports)
        return

    for lf in log_files:
        print(f"\nLoading {lf}…")
        frames = load_log(str(lf))
        print(f"  {len(frames)} frames  ({frames[0].t:.1f}s → {frames[-1].t:.1f}s)")

        if args.metrics_only:
            compute_metrics_from_log(frames)
        else:
            dash = ReplayDashboard(frames, lf.stem, speed=args.speed)
            dash.run()


if __name__ == '__main__':
    main()
