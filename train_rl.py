"""
Offline RL training script.

Phase 1: Collect PID demonstrations (fast simulation, no networking/viz).
Phase 2: Behavioral cloning from demonstrations.
Phase 3: Optional REINFORCE fine-tuning.

Saves trained model to 'rl_policy.npz'.

Usage:
    python train_rl.py [--episodes N] [--no-reinforce]
"""

import argparse
import sys
import numpy as np
from collections import deque

from vehicle import load_from_cad
from physics.state import state_from_config, PhysicsState
from physics.dynamics import rk4_step
from physics.quaternion import euler_to_quat
from control.pid_controller import CascadedPIDController
from control.rl_controller import RLController, compute_landing_reward, _encode_state
from evaluation.metrics import MetricsCollector, print_comparison
import config


def run_episode_fast(params, controller, max_t=60.0, noise=True, rng=None):
    """
    Fast episode runner: works with raw vectors to avoid per-step Python overhead.
    Returns (states, actions, rewards) arrays for BC training.
    """
    if rng is None:
        rng = np.random.default_rng()

    sv = np.zeros(14)
    sv[0] = config.INIT_X + (rng.uniform(-5, 5) if noise else 0)
    sv[1] = config.INIT_Y + (rng.uniform(-5, 5) if noise else 0)
    sv[2] = config.INIT_ALTITUDE + (rng.uniform(-20, 20) if noise else 0)
    sv[3] = config.INIT_VX + (rng.uniform(-2, 2) if noise else 0)
    sv[4] = config.INIT_VY + (rng.uniform(-2, 2) if noise else 0)
    sv[5] = config.INIT_VZ + (rng.uniform(-1, 1) if noise else 0)
    init_pitch = config.INIT_PITCH + (rng.uniform(-0.1, 0.1) if noise else 0)
    sv[6:10] = euler_to_quat(0.0, init_pitch, 0.0)
    sv[13]   = config.MASS_FUEL_INIT

    target  = np.array([config.TARGET_X, config.TARGET_Y, config.TARGET_Z])
    dt      = config.DT_PHYSICS

    # Pre-allocate output arrays (worst case max steps)
    max_steps = int(max_t / dt)
    states_buf  = np.empty((max_steps, config.RL_STATE_DIM))
    actions_buf = np.empty((max_steps, config.RL_ACTION_DIM))
    rewards_buf = np.empty(max_steps)
    n = 0

    if hasattr(controller, 'reset'):
        controller.reset()

    # Use a lighter-weight wrapper that avoids re-allocating arrays each step
    class _LightState:
        __slots__ = ('_v',)
        def __init__(self, v): self._v = v
        @property
        def position(self):     return self._v[0:3]
        @property
        def velocity(self):     return self._v[3:6]
        @property
        def quaternion(self):   return self._v[6:10]
        @property
        def ang_vel(self):      return self._v[10:13]
        @property
        def fuel_mass(self):    return float(self._v[13])
        @property
        def altitude(self):     return float(self._v[2])
        @property
        def speed(self):        return float(np.linalg.norm(self._v[3:6]))
        @property
        def euler(self):
            from physics.quaternion import quat_to_euler
            return quat_to_euler(self._v[6:10])
        @property
        def dcm(self):
            from physics.quaternion import quat_to_dcm
            return quat_to_dcm(self._v[6:10])

    # Training uses forward Euler + 4x larger dt for speed.
    # The RL policy only needs approximate dynamics to learn the right actions.
    train_dt = dt * 4   # 0.02s steps (50Hz) — ~16x faster than RK4 @ 200Hz
    from physics.dynamics import derivatives

    state = _LightState(sv)
    for i in range(max_steps):
        action = controller.compute(state, target, dt)   # controller uses nominal dt
        states_buf[i]  = _encode_state(state, target)
        actions_buf[i] = action
        landed = float(sv[2]) < 0.1
        rewards_buf[i] = compute_landing_reward(state, target, action, landed)
        n += 1

        if landed:
            break

        # Forward Euler with larger step (4× faster than RK4 @ 0.005s)
        dsv = derivatives(sv, action, params)
        sv  = sv + train_dt * dsv
        from physics.quaternion import quat_normalize
        sv[6:10] = quat_normalize(sv[6:10])
        sv[13]   = max(0.0, sv[13])
        if sv[2] < 0.0:
            sv[2] = 0.0; sv[3:6] = 0; sv[10:13] = 0
        state._v = sv

    return states_buf[:n], actions_buf[:n], rewards_buf[:n]


def evaluate(params, controller, name: str, n_episodes: int = 5) -> dict:
    """Evaluate controller over multiple episodes. Returns mean metrics."""
    rng = np.random.default_rng(99)
    speeds, errs, fuels = [], [], []
    for ep in range(n_episodes):
        sv = np.zeros(14)
        sv[0] = config.INIT_X; sv[1] = config.INIT_Y; sv[2] = config.INIT_ALTITUDE
        sv[3] = config.INIT_VX; sv[4] = config.INIT_VY; sv[5] = config.INIT_VZ
        sv[6:10] = euler_to_quat(0.0, config.INIT_PITCH, 0.0)
        sv[13]   = config.MASS_FUEL_INIT
        state    = PhysicsState(sv)
        target   = np.array([config.TARGET_X, config.TARGET_Y, config.TARGET_Z])
        if hasattr(controller, 'reset'):
            controller.reset()
        for _ in range(int(60.0 / config.DT_PHYSICS)):
            action = controller.compute(state, target, config.DT_PHYSICS)
            sv_next = rk4_step(state.vec, action, params, config.DT_PHYSICS)
            state   = PhysicsState(sv_next)
            if state.altitude < 0.1:
                break
        speeds.append(state.speed)
        errs.append(float(np.linalg.norm(state.position[:2] - target[:2])))
        fuels.append(config.MASS_FUEL_INIT - state.fuel_mass)

    return {
        'controller':    name,
        'touch_speed':   np.mean(speeds),
        'touch_err':     np.mean(errs),
        'fuel_used':     np.mean(fuels),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL landing controller")
    parser.add_argument('--episodes',     type=int, default=30,  help='BC collection episodes')
    parser.add_argument('--rl-episodes',  type=int, default=10,  help='REINFORCE episodes')
    parser.add_argument('--no-reinforce', action='store_true')
    parser.add_argument('--output',       default='rl_policy', help='Output model path')
    args = parser.parse_args()

    print("=" * 60)
    print("  NETOBOT — Autonomous Landing RL Training")
    print("=" * 60)

    params = load_from_cad('cad/lander_cad.json')
    print(params.summary())

    pid = CascadedPIDController(params)
    rl  = RLController(params)

    rng = np.random.default_rng(42)

    # ── Phase 1: Collect PID demonstrations ──────────────────────────────
    print(f"\n[Phase 1] Collecting {args.episodes} PID demonstration episodes…")
    all_states  = []
    all_actions = []
    for ep in range(args.episodes):
        pid.reset()
        s, a, _ = run_episode_fast(params, pid, noise=True, rng=rng)
        all_states.append(s)
        all_actions.append(a)
        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep+1}/{args.episodes} episodes "
                  f"({sum(len(x) for x in all_states):,} steps total)")

    states_bc  = np.vstack(all_states)
    actions_bc = np.vstack(all_actions)
    print(f"  Dataset: {len(states_bc):,} state-action pairs")

    # ── Phase 2: Behavioral Cloning ───────────────────────────────────────
    print(f"\n[Phase 2] Behavioral Cloning ({config.RL_BC_EPOCHS} epochs)…")
    losses = rl.train_bc(states_bc, actions_bc, verbose=True)
    print(f"  Final BC loss: {losses[-1]:.6f}")

    # ── Phase 3: REINFORCE fine-tuning ────────────────────────────────────
    if not args.no_reinforce:
        print(f"\n[Phase 3] REINFORCE fine-tuning ({args.rl_episodes} episodes)…")
        returns = []
        for ep in range(args.rl_episodes):
            rl._reset_episode()
            sv = np.zeros(14)
            sv[0] = config.INIT_X; sv[1] = config.INIT_Y; sv[2] = config.INIT_ALTITUDE
            sv[3] = config.INIT_VX; sv[4] = config.INIT_VY; sv[5] = config.INIT_VZ
            sv[6:10] = euler_to_quat(0.0, config.INIT_PITCH, 0.0)
            sv[13]   = config.MASS_FUEL_INIT
            state    = PhysicsState(sv)
            target   = np.array([config.TARGET_X, config.TARGET_Y, config.TARGET_Z])

            for step in range(int(60.0 / config.DT_PHYSICS)):
                action  = rl.compute(state, target, config.DT_PHYSICS)
                landed  = state.altitude < 0.1
                reward  = compute_landing_reward(state, target, action, landed)
                rl.record_step(state, action, reward, target)
                sv_next = rk4_step(state.vec, action, params, config.DT_PHYSICS)
                state   = PhysicsState(sv_next)
                if landed:
                    break

            mean_ret = rl.finish_episode()
            returns.append(mean_ret)
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1:3d}/{args.rl_episodes}  "
                      f"mean_return={mean_ret:8.2f}  "
                      f"recent_avg={np.mean(returns[-10:]):8.2f}")

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n[Evaluation] Comparing PID vs RL over 5 episodes…")
    pid.reset()
    r_pid = evaluate(params, pid, 'PID',   n_episodes=5)
    r_rl  = evaluate(params, rl,  'RL-BC', n_episodes=5)

    print("\n  Controller     Touch Speed   Touch Error   Fuel Used")
    print("  " + "-" * 54)
    for r in [r_pid, r_rl]:
        print(f"  {r['controller']:<14} {r['touch_speed']:6.2f} m/s    "
              f"{r['touch_err']:6.2f} m       {r['fuel_used']:.3f} kg")

    # ── Save model ────────────────────────────────────────────────────────
    rl.save(args.output)
    print(f"\n  Model saved → {args.output}.npz")
    print("=" * 60)


if __name__ == '__main__':
    main()
