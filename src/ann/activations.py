"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

from __future__ import annotations
import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def drelu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out

def dsigmoid_from_sigmoid(sig_x: np.ndarray) -> np.ndarray:
    return sig_x * (1.0 - sig_x)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x).astype(np.float32)

def dtanh_from_tanh(tanh_x: np.ndarray) -> np.ndarray:
    return (1.0 - tanh_x * tanh_x).astype(np.float32)

def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    expv = np.exp(shifted)
    return expv / np.sum(expv, axis=1, keepdims=True)

def get_activation(name: str):
    name = (name or "").lower().strip()
    if name == "relu":
        return relu, drelu, "pre"
    if name == "sigmoid":
        return sigmoid, dsigmoid_from_sigmoid, "post"
    if name == "tanh":
        return tanh, dtanh_from_tanh, "post"
    if name in ["identity", "linear", ""]:
        return (lambda x: x), (lambda _: np.ones_like(_, dtype=np.float32)), "pre"
    raise ValueError(f"The given activation: {name} function is not supported. Please choose either relu, sigmoid, tanh, or linear.")