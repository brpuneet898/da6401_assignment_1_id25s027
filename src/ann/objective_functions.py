"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

from __future__ import annotations
import numpy as np
from .activations import softmax

def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = y.astype(int).reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

def cross_entropy_loss_and_grad(logits: np.ndarray, y_true: np.ndarray, eps: float = 1e-12,):
    probs = softmax(logits)  
    n = logits.shape[0]
    if y_true.ndim == 1:
        y_idx = y_true.astype(int)
        correct = probs[np.arange(n), y_idx]
        loss = -np.mean(np.log(np.clip(correct, eps, 1.0)))
        dlogits = probs.copy()
        dlogits[np.arange(n), y_idx] -= 1.0
        dlogits /= n
        return float(loss), dlogits.astype(np.float32), probs
    y_oh = y_true.astype(np.float32)
    loss = -np.mean(np.sum(y_oh * np.log(np.clip(probs, eps, 1.0)), axis=1))
    dlogits = (probs - y_oh) / n
    return float(loss), dlogits.astype(np.float32), probs


def mse_loss_and_grad(logits: np.ndarray, y_true: np.ndarray,):
    probs = softmax(logits)
    n, c = probs.shape

    if y_true.ndim == 1:
        y_oh = _one_hot(y_true, c)
    else:
        y_oh = y_true.astype(np.float32)
    diff = probs - y_oh
    loss = float(np.mean(np.sum(diff * diff, axis=1)))
    dprobs = (2.0 * diff) / n
    dot = np.sum(dprobs * probs, axis=1, keepdims=True)
    dlogits = probs * (dprobs - dot)
    return loss, dlogits.astype(np.float32), probs


def get_loss(name: str):
    name = (name or "").lower().strip()
    if name in ["cross_entropy", "ce", "cross-entropy"]:
        return cross_entropy_loss_and_grad
    if name in ["mse", "mean_squared_error", "mean-squared-error"]:
        return mse_loss_and_grad
    raise ValueError(f"The given loss: {name} function is not supported. Please choose either cross_entropy or mse.")