"""
RL-based landing controller.

Training pipeline:
  1. Behavioral Cloning (BC): supervised learning from PID demonstrations
     - Fast convergence, stable initial policy
  2. REINFORCE online fine-tuning: policy gradient using episode rewards
     - Improves performance in edge cases and failure modes

Inference: < 0.1 ms per call (pure NumPy 14→128→64→4 MLP)

Control output: [throttle, tau_x_norm, tau_y_norm, tau_z_norm]
  Same interface as CascadedPIDController.compute().
"""

import threading
import numpy as np
from collections import deque
from typing import Optional, List, Tuple

from control.neural_net import NumpyMLP
import config


_STATE_NORMS = np.array([
    # pos error (3)     vel (3)          euler (3)        omega (3)      fuel_frac alt_norm
    50.0, 50.0, 100.0,  10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 100.0
])


def _encode_state(state, target: np.ndarray) -> np.ndarray:
    """Build normalised 14-dim input vector from PhysicsState + target."""
    pos_err   = target - state.position
    fuel_frac = state.fuel_mass / config.MASS_FUEL_INIT
    raw = np.concatenate([
        pos_err,
        state.velocity,
        state.euler,
        state.ang_vel,
        [fuel_frac, state.altitude],
    ])
    return raw / _STATE_NORMS


def _decode_action(out: np.ndarray, params) -> np.ndarray:
    """Map raw network output → [throttle, tau_x, tau_y, tau_z]."""
    throttle = float(np.clip(1.0 / (1.0 + np.exp(-out[0])), 0.0, 1.0))  # sigmoid
    torques  = np.tanh(out[1:4])                                          # tanh → [-1,1]
    return np.array([throttle, torques[0], torques[1], torques[2]])


class RLController:
    """
    Wraps NumpyMLP as a drop-in replacement for CascadedPIDController.
    Thread-safe weight swap for online training.
    """

    def __init__(self, params):
        self.params  = params
        self.net     = NumpyMLP([config.RL_STATE_DIM,
                                 config.RL_HIDDEN_1,
                                 config.RL_HIDDEN_2,
                                 config.RL_ACTION_DIM])
        self._lock   = threading.Lock()

        # REINFORCE trajectory buffer
        self._episode_states:  List[np.ndarray] = []
        self._episode_actions: List[np.ndarray] = []
        self._episode_rewards: List[float]       = []
        self._episode_log_probs: List[float]     = []

        self.is_trained = False

    # ── Inference ─────────────────────────────────────────────────────────────

    def compute(self, state, target: np.ndarray, dt: float, yaw_setpoint: float = 0.0) -> np.ndarray:
        s = _encode_state(state, target)
        with self._lock:
            raw = self.net.forward(s)
        return _decode_action(raw, self.params)

    # ── Behavioral Cloning ────────────────────────────────────────────────────

    def train_bc(
        self,
        states: np.ndarray,   # (N, state_dim)
        actions: np.ndarray,  # (N, action_dim) — PID outputs
        epochs: int = config.RL_BC_EPOCHS,
        lr: float = config.RL_LR,
        verbose: bool = True,
    ):
        """
        Supervised learning from PID demonstrations.
        actions are the raw PID control vectors.
        """
        N = states.shape[0]
        batch = config.RL_BATCH_SIZE
        losses = []

        for epoch in range(epochs):
            idx  = np.random.permutation(N)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch):
                end    = min(start + batch, N)
                sb     = states[idx[start:end]]
                ab     = actions[idx[start:end]]

                # Forward
                pred   = self.net.forward(sb)

                # MSE loss on throttle; separate scale for torques
                diff   = pred - ab
                diff[:, 0] *= 2.0   # upweight throttle accuracy
                loss   = (diff**2).mean()
                epoch_loss += loss
                n_batches  += 1

                # Backward: grad of MSE = 2*(pred-target)/N
                grad = 2.0 * diff / diff.shape[0]
                grad[:, 0] *= 2.0
                dW, db = self.net.backward(grad)
                with self._lock:
                    self.net.apply_gradients(dW, db, lr)

            losses.append(epoch_loss / n_batches)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"  BC epoch {epoch+1:3d}/{epochs}  loss={losses[-1]:.6f}")

        self.is_trained = True
        return losses

    # ── REINFORCE online fine-tuning ──────────────────────────────────────────

    def record_step(self, state, action: np.ndarray, reward: float, target: np.ndarray):
        """Record one (s, a, r) transition during an episode."""
        s = _encode_state(state, target)
        self._episode_states.append(s)
        self._episode_actions.append(action.copy())
        self._episode_rewards.append(reward)

    def finish_episode(self, lr: float = config.RL_LR * 0.1) -> float:
        """
        Compute discounted returns and apply one REINFORCE gradient step.
        Returns mean episode return.
        """
        if len(self._episode_rewards) < 2:
            self._reset_episode()
            return 0.0

        rewards = np.array(self._episode_rewards)
        returns = _discounted_returns(rewards, config.RL_GAMMA)

        # Normalise returns for variance reduction
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        states  = np.array(self._episode_states)    # (T, state_dim)
        actions = np.array(self._episode_actions)   # (T, action_dim)

        pred    = self.net.forward(states)           # (T, action_dim)

        # Policy gradient via MSE proxy:
        # Treat as supervised with targets = action weighted by return.
        # Negative return → push away; positive → push toward.
        targets = pred - returns[:, None] * (pred - actions)

        grad = 2.0 * (pred - targets) / len(states)
        dW, db = self.net.backward(grad)
        with self._lock:
            self.net.apply_gradients(dW, db, lr)

        mean_ret = float(returns.mean())
        self._reset_episode()
        return mean_ret

    def _reset_episode(self):
        self._episode_states.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()

    def save(self, path: str):
        with self._lock:
            self.net.save(path)

    def load(self, path: str):
        with self._lock:
            data = np.load(path + '.npz')
            for i in range(self.net.n_layers):
                self.net.W[i] = data[f'W{i}']
                self.net.b[i] = data[f'b{i}']
        self.is_trained = True


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    T = len(rewards)
    G = np.zeros(T)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        G[t]    = running
    return G


def compute_landing_reward(state, target: np.ndarray, action: np.ndarray, landed: bool) -> float:
    """Reward function used during both BC data collection annotation and REINFORCE."""
    w = config.RL_REWARD_WEIGHTS
    pos_err  = np.linalg.norm(state.position - target)
    vel_norm = np.linalg.norm(state.velocity)
    att_norm = np.linalg.norm(state.euler[:2])   # roll + pitch (not yaw)
    omega_n  = np.linalg.norm(state.ang_vel)
    effort   = np.linalg.norm(action)

    r = (
        -w['pos']    * pos_err
        -w['vel']    * vel_norm
        -w['att']    * att_norm
        -w['omega']  * omega_n
        -w['effort'] * effort
        +w['alive']
    )

    if landed and pos_err < 2.0 and vel_norm < 1.0:
        r += w['landing']

    return float(r)
