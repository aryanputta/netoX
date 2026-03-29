"""
Feed-forward neural network in pure NumPy.
Used for the RL/behavioral-cloning controller.

Architecture: Linear → Tanh → Linear → Tanh → Linear
Gradients computed analytically for backpropagation.
Adam optimizer implemented in-place.
"""

import numpy as np
from typing import List, Tuple


def _tanh(x):    return np.tanh(x)
def _dtanh(x):   return 1.0 - np.tanh(x)**2
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyMLP:
    """
    Multi-layer perceptron. Supports variable depth.

    Layer layout:
      input → [Linear+Tanh] × n_hidden_layers → [Linear] → output
    The output activation is applied externally (by the controller).
    """

    def __init__(self, layer_sizes: List[int], seed: int = 42):
        rng = np.random.default_rng(seed)
        self.sizes  = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        # He initialisation for tanh hidden layers
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self.n_layers):
            fan_in = layer_sizes[i]
            scale  = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0, scale, (layer_sizes[i], layer_sizes[i+1])))
            self.b.append(np.zeros(layer_sizes[i+1]))

        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self._t = 0   # adam step counter

        # Cache for backward
        self._cache: List[Tuple[np.ndarray, np.ndarray]] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass. Caches pre-activations for backward.
        x: (batch, input_dim)  or (input_dim,)
        """
        batched = x.ndim == 2
        if not batched:
            x = x[np.newaxis, :]

        self._cache = []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            self._cache.append((h, z))
            if i < self.n_layers - 1:
                h = _tanh(z)
            else:
                h = z  # output layer: no activation (applied by caller)

        return h.squeeze(0) if not batched else h

    def backward(self, grad_out: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass using cached activations.
        grad_out: gradient w.r.t. output  (batch, out_dim) or (out_dim,)
        Returns: (dW_list, db_list) — same shapes as self.W, self.b
        """
        if grad_out.ndim == 1:
            grad_out = grad_out[np.newaxis, :]

        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        delta = grad_out
        for i in reversed(range(self.n_layers)):
            h_in, z = self._cache[i]
            if i < self.n_layers - 1:
                delta = delta * _dtanh(z)

            dW[i] = h_in.T @ delta / grad_out.shape[0]
            db[i] = delta.mean(axis=0)

            if i > 0:
                delta = delta @ self.W[i].T

        return dW, db

    def apply_gradients(self, dW: List[np.ndarray], db: List[np.ndarray], lr: float):
        """Adam update."""
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = self._t

        for i in range(self.n_layers):
            # Weights
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * dW[i]**2
            mhat = self.mW[i] / (1 - beta1**t)
            vhat = self.vW[i] / (1 - beta2**t)
            self.W[i] -= lr * mhat / (np.sqrt(vhat) + eps)

            # Biases
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * db[i]**2
            mhat = self.mb[i] / (1 - beta1**t)
            vhat = self.vb[i] / (1 - beta2**t)
            self.b[i] -= lr * mhat / (np.sqrt(vhat) + eps)

    def get_flat_params(self) -> np.ndarray:
        parts = []
        for W, b in zip(self.W, self.b):
            parts.extend([W.ravel(), b.ravel()])
        return np.concatenate(parts)

    def set_flat_params(self, theta: np.ndarray):
        offset = 0
        for i in range(self.n_layers):
            nw = self.W[i].size
            nb = self.b[i].size
            self.W[i] = theta[offset:offset+nw].reshape(self.W[i].shape)
            offset += nw
            self.b[i] = theta[offset:offset+nb]
            offset += nb

    def save(self, path: str):
        np.savez(path, **{f'W{i}': w for i, w in enumerate(self.W)},
                       **{f'b{i}': b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path: str, layer_sizes: List[int]) -> 'NumpyMLP':
        net = cls(layer_sizes)
        data = np.load(path + '.npz')
        net.W = [data[f'W{i}'] for i in range(net.n_layers)]
        net.b = [data[f'b{i}'] for i in range(net.n_layers)]
        return net
