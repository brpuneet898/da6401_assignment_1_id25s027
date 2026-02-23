"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

from __future__ import annotations
import numpy as np
from .activations import get_activation


class NeuralLayer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        weight_init: str = "xavier",
        seed: int = 42,
    ):
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.activation_name = (activation or "none").lower().strip()
        self.act, self.dact, self._cache_type = get_activation(self.activation_name)

        self.weight_init = (weight_init or "xavier").lower().strip()
        self.rng = np.random.default_rng(seed)

        self.W = self._init_weights(self.in_features, self.out_features, self.weight_init).astype(np.float32)
        self.b = np.zeros((1, self.out_features), dtype=np.float32)

        self.grad_W = np.zeros_like(self.W, dtype=np.float32)
        self.grad_b = np.zeros_like(self.b, dtype=np.float32)

        self._X = None
        self._Z = None
        self._A = None

    def _init_weights(self, fan_in: int, fan_out: int, method: str) -> np.ndarray:
        if method in ["xavier", "glorot"]:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return self.rng.uniform(-limit, limit, size=(fan_in, fan_out))
        if method in ["he", "kaiming"]:
            std = np.sqrt(2.0 / fan_in)
            return self.rng.normal(0.0, std, size=(fan_in, fan_out))
        if method in ["random", "normal"]:
            return self.rng.normal(0.0, 0.01, size=(fan_in, fan_out))
        if method in ["zeros", "zero"]:
            return np.zeros((fan_in, fan_out), dtype=np.float32)
        raise ValueError(f"The given weight_init: {method} is not supported. Please choose either xavier, he, random, or zeros.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        self._X = X
        Z = X @ self.W + self.b 
        self._Z = Z
        A = self.act(Z).astype(np.float32)
        self._A = A
        return A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        dA = dA.astype(np.float32)

        if self.activation_name not in ["none", "identity", "linear", ""]:
            if self._cache_type == "pre":
                dZ = dA * self.dact(self._Z)
            else:
                dZ = dA * self.dact(self._A)
        else:
            dZ = dA

        X = self._X
        n = X.shape[0]
        self.grad_W = (X.T @ dZ) / n
        self.grad_b = np.mean(dZ, axis=0, keepdims=True)

        dX = dZ @ self.W.T
        return dX.astype(np.float32)