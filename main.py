"""
NetoBot Autonomous Landing System — main orchestrator.

Starts all threads (physics, control, wind, networking) then hands off to
the visualization dashboard on the main thread (required by matplotlib/macOS).

Usage:
    python main.py [--controller pid|rl] [--train] [--no-viz] [--no-network]
                   [--latency MEAN_MS] [--loss PCT] [--fail] [--wind SPEED]

Keyboard shortcuts (when dashboard is open):
    P  switch to PID controller
    R  switch to RL controller
    F  toggle engine degradation failure
    W  toggle wind gust
    Q  quit
"""

import argparse
import sys
import time
import threading
import numpy as np

from vehicle import load_from_cad
from physics.engine import SimulationEngine
from shared_state import SharedState
from control_thread import ControlThread
from wind_model import WindModel
from networking.telemetry_server import TelemetryServer
from networking.ground_station import GroundStationClient
from visualization.dashboard import Dashboard
from evaluation.metrics import MetricsCollector, print_comparison
import config


def parse_args():
    p = argparse.ArgumentParser(description='NetoBot Autonomous Landing System')
    p.add_argument('--controller', choices=['pid', 'rl'], default='pid',
                   help='Starting controller (default: pid)')
    p.add_argument('--train',      action='store_true',
                   help='Run RL training before simulation')
    p.add_argument('--no-viz',     action='store_true',
                   help='Headless mode (no dashboard)')
    p.add_argument('--no-network', action='store_true',
                   help='Skip networking layer')
    p.add_argument('--latency',    type=float, default=config.NET_LATENCY_MEAN_MS,
                   help='Network latency mean (ms)')
    p.add_argument('--loss',       type=float, default=config.NET_LOSS_RATE * 100,
                   help='Packet loss rate (%%)')
    p.add_argument('--fail',       action='store_true',
                   help='Enable engine degradation failure mode')
    p.add_argument('--wind',       type=float, default=config.WIND_SPEED_MEAN,
                   help='Wind speed (m/s)')
    p.add_argument('--compare',    action='store_true',
                   help='Run back-to-back PID and RL episodes then compare metrics')
    return p.parse_args()


def run_headless(shared, duration=60.0):
    """Headless mode: print telemetry to console at 2 Hz."""
    t_start = time.monotonic()
    while not shared.shutdown.is_set():
        state  = shared.read_physics()
        sim_t  = shared.read_sim_time()
        ctrl   = shared.read_controller_name()
        e      = state.euler

        print(
            f"\r  T+{sim_t:6.1f}s  "
            f"Alt={state.altitude:7.2f}m  "
            f"Speed={state.speed:5.2f}m/s  "
            f"Fuel={state.fuel_mass:.2f}kg  "
            f"Roll={np.degrees(e[0]):6.1f}°  "
            f"[{ctrl}]   ",
            end='', flush=True
        )

        if state.altitude < 0.1:
            print(f"\n  LANDED at T+{sim_t:.1f}s  speed={state.speed:.2f}m/s")
            break
        if sim_t > duration:
            print(f"\n  Timeout at T+{sim_t:.1f}s")
            break

        time.sleep(0.5)

    shared.shutdown.set()


def main():
    args = parse_args()

    # Apply CLI overrides
    config.NET_LATENCY_MEAN_MS = args.latency
    config.NET_LOSS_RATE       = args.loss / 100.0
    config.WIND_SPEED_MEAN     = args.wind
    if args.fail:
        config.ENGINE_DEGRADE_PCT   = 0.30
        config.ENGINE_DEGRADE_START = 5.0

    # ── Load CAD parameters ───────────────────────────────────────────────
    params = load_from_cad('cad/lander_cad.json')
    print(params.summary())
    print(f"\n  Packet sizes: {__import__('networking.packets', fromlist=['packet_sizes']).packet_sizes()}")
    print()

    # ── Optional RL training ──────────────────────────────────────────────
    if args.train:
        print("  Running RL training… (use train_rl.py for full training)")
        from train_rl import main as train_main
        sys.argv = ['train_rl.py', '--episodes', '100', '--rl-episodes', '20']
        train_main()

    # ── Shared state ──────────────────────────────────────────────────────
    shared = SharedState()
    shared.set_controller(args.controller.upper())

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = MetricsCollector()

    # ── Comparison mode ───────────────────────────────────────────────────
    if args.compare:
        _run_comparison(params, args)
        return

    # ── Threads ───────────────────────────────────────────────────────────
    threads = []

    sim  = SimulationEngine(params, shared)
    ctrl = ControlThread(shared, params, metrics)
    wind = WindModel(shared)
    threads.extend([sim, ctrl, wind])

    if not args.no_network:
        telem = TelemetryServer(shared)
        gs    = GroundStationClient(
            shared,
            latency_mean_ms  = args.latency,
            latency_std_ms   = config.NET_LATENCY_STD_MS,
            packet_loss_rate = args.loss / 100.0,
        )
        threads.extend([telem, gs])

    for t in threads:
        t.start()

    print(f"  All threads started. Controller: [{args.controller.upper()}]")
    print("  Dashboard: P=PID  R=RL  F=Fail  W=Wind  Q=Quit\n")

    # ── Visualization (main thread) or headless ───────────────────────────
    try:
        if args.no_viz:
            run_headless(shared)
        else:
            dash = Dashboard(shared, params)
            dash.run_blocking()
    except KeyboardInterrupt:
        pass
    finally:
        shared.shutdown.set()
        for t in threads:
            t.join(timeout=2.0)

    # ── Final metrics ──────────────────────────────────────────────────────
    net = shared.read_net_stats()
    report = metrics.finalize(
        args.controller.upper(),
        net_latency_ms = net.get('mean_latency_ms', 0),
        net_loss_pct   = net.get('loss_pct', 0),
        pkts_dropped   = net.get('pkts_dropped', 0),
    )
    print("\n" + report.summary_table())
    print(f"\n  Result: {report.success_rating()}")


def _run_comparison(params, args):
    """Run PID then RL back-to-back offline, print comparison."""
    from train_rl import run_episode_fast, evaluate
    from control.pid_controller import CascadedPIDController
    from control.rl_controller import RLController

    print("  Running side-by-side comparison (offline fast simulation)…\n")

    pid = CascadedPIDController(params)
    rl  = RLController(params)

    # Load pre-trained RL if available
    import pathlib
    if pathlib.Path('rl_policy.npz').exists():
        rl.load('rl_policy')
        print("  Loaded pre-trained RL model from rl_policy.npz")
    else:
        print("  No rl_policy.npz found — training from scratch (100 BC episodes)…")
        from train_rl import main as train_main
        sys.argv = ['train_rl.py', '--episodes', '100', '--no-reinforce']
        train_main()
        rl.load('rl_policy')

    r_pid = evaluate(params, pid, 'PID',    n_episodes=10)
    r_rl  = evaluate(params, rl,  'RL-BC',  n_episodes=10)

    print("\n  ┌─────────────────────────────────────────────────┐")
    print("  │           COMPARISON (10-episode average)        │")
    print("  ├────────────┬─────────────┬─────────────┬────────┤")
    print("  │ Controller │ Touch Speed │ Touch Error │  Fuel  │")
    print("  ├────────────┼─────────────┼─────────────┼────────┤")
    for r in [r_pid, r_rl]:
        print(f"  │ {r['controller']:<10} │ {r['touch_speed']:8.3f} m/s │ "
              f"{r['touch_err']:8.3f} m  │ {r['fuel_used']:.3f}kg│")
    print("  └────────────┴─────────────┴─────────────┴────────┘")

    # Engineering tradeoff analysis
    print("\n  ENGINEERING TRADEOFFS:")
    print(f"   Touch speed  Δ: {r_rl['touch_speed'] - r_pid['touch_speed']:+.3f} m/s "
          f"({'RL better' if r_rl['touch_speed'] < r_pid['touch_speed'] else 'PID better'})")
    print(f"   Position err Δ: {r_rl['touch_err'] - r_pid['touch_err']:+.3f} m "
          f"({'RL better' if r_rl['touch_err'] < r_pid['touch_err'] else 'PID better'})")
    print(f"   Fuel saved:     {r_pid['fuel_used'] - r_rl['fuel_used']:+.3f} kg "
          f"({'RL more efficient' if r_rl['fuel_used'] < r_pid['fuel_used'] else 'PID more efficient'})")

    print("\n  Network latency impact:")
    print(f"   At μ={args.latency:.0f}ms latency, ground-station control commands")
    print(f"   arrive ~{args.latency/1000:.3f}s stale → degraded closed-loop bandwidth")
    print(f"   Effective control bandwidth ≈ {1000/(2*args.latency+10):.1f} Hz "
          f"(Nyquist limited by round-trip)")

    print("\n  CAD parameter sensitivity:")
    print(f"   Vehicle mass: {params.total_mass:.1f} kg  →  hover throttle "
          f"≈ {params.total_mass*9.81/params.max_thrust*100:.1f}%")
    print(f"   If CG offset = {params.cg_offset[2]:.3f}m  → attitude coupling with throttle")
    print(f"   Ixx={params.Ixx:.4f} kg·m²  → roll bandwidth "
          f"≈ {(params.max_rcs_torque/params.Ixx)**0.5/(2*3.14159):.1f} Hz (natural)")


if __name__ == '__main__':
    main()
