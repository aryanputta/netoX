"""
Real-time mission control dashboard.

Layout (2×3 grid):
  [3D Trajectory]  [Altitude + Speed]  [Roll/Pitch/Yaw]
  [Motor Throttle] [Network Stats]     [Metrics Panel]

Interactive controls:
  P  — switch to PID controller
  R  — switch to RL controller
  F  — toggle engine degradation failure
  W  — toggle wind gust
  Q  — quit

The dashboard is designed to run on the main thread (required by matplotlib
on macOS). All data is read from the SharedState via thread-safe snapshots.
"""

import threading
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from collections import deque

import config


# ── Vehicle 2D projection (side-view silhouette points) ──────────────────────
def _vehicle_silhouette(scale: float = 1.0):
    """Returns (x_body, z_body) pairs for a simple rocket silhouette."""
    r, h = config.BODY_RADIUS * scale, config.BODY_HEIGHT * scale
    leg  = 0.3 * scale
    nose = 0.15 * scale
    xs = np.array([-r, -r, -r*0.3, 0, r*0.3, r, r,  r+leg, r+leg*0.5, -r-leg*0.5, -r-leg, -r])
    zs = np.array([ 0,  h,   h+nose, h+nose*1.5, h+nose, h, 0, -leg*0.5,  0,         0,     -leg*0.5, 0])
    return xs, zs


class Dashboard:
    def __init__(self, shared, params):
        self.shared = shared
        self.params = params

        # History buffers for plotting (separate from SharedState to avoid lock contention)
        self._N      = 1000
        self.t_hist  = deque(maxlen=self._N)
        self.x_hist  = deque(maxlen=self._N)
        self.y_hist  = deque(maxlen=self._N)
        self.z_hist  = deque(maxlen=self._N)
        self.vx_hist = deque(maxlen=self._N)
        self.vy_hist = deque(maxlen=self._N)
        self.vz_hist = deque(maxlen=self._N)
        self.roll_h  = deque(maxlen=self._N)
        self.pitch_h = deque(maxlen=self._N)
        self.yaw_h   = deque(maxlen=self._N)
        self.thr_h   = deque(maxlen=self._N)
        self.fuel_h  = deque(maxlen=self._N)
        self.lat_h   = deque(maxlen=200)
        self.loss_h  = deque(maxlen=200)
        self.ise_pid = deque(maxlen=500)
        self.ise_rl  = deque(maxlen=500)

        # EKF estimate history (ground-station view — affected by network latency)
        self.ex_hist = deque(maxlen=self._N)
        self.ey_hist = deque(maxlen=self._N)
        self.ez_hist = deque(maxlen=self._N)

        self._frame = 0
        self._running_ise = {'PID': 0.0, 'RL': 0.0}
        self._ise_t = 0.0

        self._setup_figure()

    # ── Figure setup ──────────────────────────────────────────────────────────

    def _setup_figure(self):
        matplotlib.rcParams.update({
            'font.family': 'monospace',
            'axes.facecolor': '#0d1117',
            'figure.facecolor': '#0d1117',
            'axes.edgecolor': '#30363d',
            'axes.labelcolor': '#e6edf3',
            'xtick.color': '#8b949e',
            'ytick.color': '#8b949e',
            'text.color': '#e6edf3',
            'grid.color': '#21262d',
            'grid.linestyle': '--',
            'grid.alpha': 0.6,
        })

        self.fig = plt.figure(figsize=(18, 10), facecolor='#0d1117')
        self.fig.canvas.manager.set_window_title('NetoBot Mission Control')

        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               left=0.05, right=0.97, top=0.93, bottom=0.08,
                               wspace=0.35, hspace=0.45)

        # [0,0] 3D Trajectory
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax3d.set_facecolor('#0d1117')

        # [0,1] Altitude + Speed
        self.ax_alt  = self.fig.add_subplot(gs[0, 1])
        self.ax_spd  = self.ax_alt.twinx()

        # [0,2] Euler angles
        self.ax_att = self.fig.add_subplot(gs[0, 2])

        # [1,0] Throttle + fuel
        self.ax_thr = self.fig.add_subplot(gs[1, 0])
        self.ax_fuel = self.ax_thr.twinx()

        # [1,1] Network stats
        self.ax_net = self.fig.add_subplot(gs[1, 1])

        # [1,2] ISE comparison
        self.ax_ise = self.fig.add_subplot(gs[1, 2])

        self._style_axes()
        self._draw_static_elements()
        self._add_status_bar()
        self._connect_keys()

    def _style_axes(self):
        titles = {
            self.ax3d:  '3D TRAJECTORY',
            self.ax_alt: 'ALTITUDE  (blue)  &  SPEED  (orange)',
            self.ax_att: 'EULER ANGLES',
            self.ax_thr: 'THROTTLE  (blue)  &  FUEL  (orange)',
            self.ax_net: 'NETWORK TELEMETRY',
            self.ax_ise: 'RUNNING ISE: PID vs RL',
        }
        for ax, title in titles.items():
            if ax != self.ax3d:
                ax.set_title(title, fontsize=7, color='#58a6ff', pad=4)
                ax.grid(True)

        self.ax_alt.set_ylabel('Altitude (m)', fontsize=7)
        self.ax_spd.set_ylabel('Speed (m/s)',  fontsize=7, color='#f0883e')
        self.ax_att.set_ylabel('Angle (deg)',  fontsize=7)
        self.ax_thr.set_ylabel('Throttle',     fontsize=7)
        self.ax_fuel.set_ylabel('Fuel (kg)',   fontsize=7, color='#f0883e')
        self.ax_net.set_ylabel('Latency (ms)', fontsize=7)
        self.ax_ise.set_ylabel('ISE (m²·s)',  fontsize=7)

        self.ax3d.set_xlabel('X (m)', fontsize=7)
        self.ax3d.set_ylabel('Y (m)', fontsize=7)
        self.ax3d.set_zlabel('Z (m)', fontsize=7)
        self.ax3d.set_title('3D TRAJECTORY', fontsize=7, color='#58a6ff')

    def _draw_static_elements(self):
        # Target marker on 3D plot
        self.ax3d.scatter([0], [0], [0], c='#ff4444', s=100, marker='X', zorder=10, label='Target')
        self.ax3d.scatter([config.INIT_X], [config.INIT_Y], [config.INIT_ALTITUDE],
                          c='#44ff88', s=60, marker='o', zorder=10, label='Start')

        # Ground plane
        xx, yy = np.meshgrid(np.linspace(-30, 30, 5), np.linspace(-30, 30, 5))
        self.ax3d.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='#3fb950')
        self.ax3d.legend(loc='upper right', fontsize=6)

    def _add_status_bar(self):
        self._status_text = self.fig.text(
            0.5, 0.97, 'INITIALISING…',
            ha='center', va='top', fontsize=9, color='#e6edf3',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='#161b22', ec='#30363d'),
        )
        self._ctrl_text = self.fig.text(
            0.01, 0.97, '', ha='left', va='top', fontsize=8, color='#3fb950',
            fontfamily='monospace',
        )
        self._key_help = self.fig.text(
            0.99, 0.97,
            '[P]PID  [R]RL  [F]Fail  [W]Wind  [Q]Quit',
            ha='right', va='top', fontsize=7, color='#8b949e',
            fontfamily='monospace',
        )

    def _connect_keys(self):
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        shared = self.shared
        key = event.key.lower() if event.key else ''
        if key == 'p':
            shared.set_controller('PID')
        elif key == 'r':
            shared.set_controller('RL')
        elif key == 'f':
            # Toggle engine degradation failure
            import config as cfg
            cfg.ENGINE_DEGRADE_PCT = 0.0 if cfg.ENGINE_DEGRADE_PCT > 0 else 0.30
            cfg.ENGINE_DEGRADE_START = shared.read_sim_time()
        elif key == 'w':
            import config as cfg
            cfg.WIND_SPEED_MEAN = 0.0 if cfg.WIND_SPEED_MEAN > 0 else 8.0
        elif key == 'q':
            shared.shutdown.set()
            plt.close('all')

    # ── Animation ─────────────────────────────────────────────────────────────

    def _fetch_snapshot(self):
        """Grab one consistent snapshot from shared state (minimal lock time)."""
        shared = self.shared
        return (
            shared.read_physics(),
            shared.read_sim_time(),
            shared.read_control(),
            shared.read_net_stats(),
            shared.read_controller_name(),
        )

    def _update_histories(self, state, sim_t, control):
        e = state.euler
        self.t_hist.append(sim_t)
        self.x_hist.append(state.position[0])
        self.y_hist.append(state.position[1])
        self.z_hist.append(state.position[2])
        self.vx_hist.append(state.velocity[0])
        self.vy_hist.append(state.velocity[1])
        self.vz_hist.append(state.velocity[2])
        self.roll_h.append(np.degrees(e[0]))
        self.pitch_h.append(np.degrees(e[1]))
        self.yaw_h.append(np.degrees(e[2]))
        self.thr_h.append(control[0])
        self.fuel_h.append(state.fuel_mass)

    def _animate(self, frame):
        try:
            state, sim_t, control, net_stats, ctrl_name = self._fetch_snapshot()
        except Exception:
            return

        self._update_histories(state, sim_t, control)
        self._frame += 1

        t    = list(self.t_hist)
        xh   = list(self.x_hist)
        yh   = list(self.y_hist)
        zh   = list(self.z_hist)
        spd  = [np.sqrt(vx**2 + vy**2 + vz**2)
                for vx, vy, vz in zip(self.vx_hist, self.vy_hist, self.vz_hist)]

        # ── 3D trajectory ─────────────────────────────────────────────────
        ax = self.ax3d
        ax.cla()
        ax.set_facecolor('#0d1117')
        ax.set_xlabel('X (m)', fontsize=6); ax.set_ylabel('Y (m)', fontsize=6)
        ax.set_zlabel('Z m)', fontsize=6)
        ax.set_title('3D TRAJECTORY', fontsize=7, color='#58a6ff')
        if len(xh) > 1:
            ax.plot(xh, yh, zh, color='#58a6ff', lw=1.5, label='True')
        ax.scatter([state.position[0]], [state.position[1]], [state.position[2]],
                   c='#f0883e', s=80, zorder=10)
        ax.scatter([0], [0], [0], c='#ff4444', s=100, marker='X')
        # Vehicle attitude arrows (body axes)
        R = state.dcm
        sc = 8.0
        p  = state.position
        for vec, col in zip(R.T, ['#ff6e6e', '#6eff6e', '#6e6eff']):
            ax.quiver(*p, *(vec * sc), color=col, lw=1.5, arrow_length_ratio=0.3)
        ax.set_xlim(-30, 30); ax.set_ylim(-30, 30); ax.set_zlim(0, config.INIT_ALTITUDE + 10)
        ax.tick_params(labelsize=5)

        # ── Altitude & speed ──────────────────────────────────────────────
        self.ax_alt.cla(); self.ax_spd.cla()
        self.ax_alt.set_facecolor('#0d1117')
        self.ax_alt.set_title('ALTITUDE & SPEED', fontsize=7, color='#58a6ff', pad=4)
        if t:
            self.ax_alt.plot(t, zh,  color='#58a6ff', lw=1.2, label='Alt')
            self.ax_spd.plot(t, spd, color='#f0883e', lw=1.0, alpha=0.8, label='Speed')
            self.ax_alt.axhline(0, color='#3fb950', lw=0.8, ls='--')
        self.ax_alt.set_ylabel('Altitude (m)', fontsize=7, color='#58a6ff')
        self.ax_spd.set_ylabel('Speed (m/s)', fontsize=7, color='#f0883e')
        self.ax_alt.grid(True, color='#21262d', ls='--', alpha=0.6)
        self.ax_alt.tick_params(labelsize=6)
        self.ax_spd.tick_params(labelsize=6, colors='#f0883e')

        # ── Euler angles ──────────────────────────────────────────────────
        self.ax_att.cla()
        self.ax_att.set_facecolor('#0d1117')
        self.ax_att.set_title('EULER ANGLES', fontsize=7, color='#58a6ff', pad=4)
        if t:
            self.ax_att.plot(t, list(self.roll_h),  color='#ff6e6e', lw=1, label='Roll')
            self.ax_att.plot(t, list(self.pitch_h), color='#6eff6e', lw=1, label='Pitch')
            self.ax_att.plot(t, list(self.yaw_h),   color='#6e6eff', lw=1, label='Yaw')
        self.ax_att.set_ylabel('deg', fontsize=7)
        self.ax_att.legend(fontsize=6, loc='upper right')
        self.ax_att.grid(True, color='#21262d', ls='--', alpha=0.6)
        self.ax_att.tick_params(labelsize=6)

        # ── Throttle & fuel ───────────────────────────────────────────────
        self.ax_thr.cla(); self.ax_fuel.cla()
        self.ax_thr.set_facecolor('#0d1117')
        self.ax_thr.set_title('THROTTLE & FUEL', fontsize=7, color='#58a6ff', pad=4)
        if t:
            self.ax_thr.plot(t, list(self.thr_h),  color='#58a6ff', lw=1.2)
            self.ax_fuel.plot(t, list(self.fuel_h), color='#f0883e', lw=1, alpha=0.8)
        self.ax_thr.set_ylim(0, 1.05)
        self.ax_thr.set_ylabel('Throttle', fontsize=7, color='#58a6ff')
        self.ax_fuel.set_ylabel('Fuel (kg)', fontsize=7, color='#f0883e')
        self.ax_thr.grid(True, color='#21262d', ls='--', alpha=0.6)
        self.ax_thr.tick_params(labelsize=6)
        self.ax_fuel.tick_params(labelsize=6, colors='#f0883e')

        # ── Network stats ─────────────────────────────────────────────────
        self.ax_net.cla()
        self.ax_net.set_facecolor('#0d1117')
        self.ax_net.set_title('NETWORK TELEMETRY', fontsize=7, color='#58a6ff', pad=4)
        if net_stats:
            lat_hist = list(net_stats.get('latency_hist', []))
            if lat_hist:
                self.ax_net.plot(lat_hist, color='#d29922', lw=1, label='RTT (ms)')
                self.ax_net.axhline(np.mean(lat_hist), color='#ff6e6e', lw=0.8,
                                    ls='--', label=f'μ={np.mean(lat_hist):.1f}ms')
                self.ax_net.legend(fontsize=6)
            # Annotate stats
            loss = net_stats.get('loss_pct', 0)
            pkts = net_stats.get('pkts_recv', 0)
            self.ax_net.text(0.02, 0.9, f'Loss: {loss:.1f}%  Rx: {pkts}',
                             transform=self.ax_net.transAxes, fontsize=7,
                             color='#8b949e')
        self.ax_net.set_ylabel('Latency (ms)', fontsize=7)
        self.ax_net.grid(True, color='#21262d', ls='--', alpha=0.6)
        self.ax_net.tick_params(labelsize=6)

        # ── ISE comparison ────────────────────────────────────────────────
        self.ax_ise.cla()
        self.ax_ise.set_facecolor('#0d1117')
        self.ax_ise.set_title('RUNNING ISE: PID vs RL', fontsize=7, color='#58a6ff', pad=4)
        ise_data = net_stats.get('ise_history', {}) if net_stats else {}
        pid_ise = list(ise_data.get('PID', []))
        rl_ise  = list(ise_data.get('RL',  []))
        if pid_ise:
            self.ax_ise.plot(pid_ise, color='#58a6ff', lw=1, label='PID')
        if rl_ise:
            self.ax_ise.plot(rl_ise, color='#3fb950', lw=1, label='RL')
        self.ax_ise.legend(fontsize=6)
        self.ax_ise.set_ylabel('ISE (m²·s)', fontsize=7)
        self.ax_ise.grid(True, color='#21262d', ls='--', alpha=0.6)
        self.ax_ise.tick_params(labelsize=6)

        # ── Status bar ────────────────────────────────────────────────────
        fuel_pct = 100 * state.fuel_mass / config.MASS_FUEL_INIT
        lat_ms   = net_stats.get('mean_latency_ms', 0) if net_stats else 0
        status = (
            f"T+{sim_t:6.1f}s  │  "
            f"Alt: {state.altitude:6.1f}m  │  "
            f"Speed: {state.speed:5.2f}m/s  │  "
            f"Fuel: {fuel_pct:5.1f}%  │  "
            f"Lat: {lat_ms:5.1f}ms  │  "
            f"{'LANDED' if state.altitude < 0.2 else 'DESCENT'}"
        )
        self._status_text.set_text(status)

        ctrl_color = '#3fb950' if ctrl_name == 'PID' else '#f78166'
        self._ctrl_text.set_text(f'[{ctrl_name}]')
        self._ctrl_text.set_color(ctrl_color)

        # Failure indicators
        if config.ENGINE_DEGRADE_PCT > 0:
            self.fig.patch.set_facecolor('#1a0a0a')
        else:
            self.fig.patch.set_facecolor('#0d1117')

    def run_blocking(self):
        """Start animation — blocks until window is closed. Call on main thread."""
        self._anim = FuncAnimation(
            self.fig,
            self._animate,
            interval=int(config.DT_VIZ * 1000),
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
