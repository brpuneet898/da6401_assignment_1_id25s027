"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

def cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> float:
    return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis = 1))

def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y - y_pred) ** 2)
