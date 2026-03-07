"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

class ReLU():
    def __init__(self, z):
        self.z = z
        self.grad = 0.0

    def forward(self):
        self.a = np.maximum(0, self.z)
        return self.a
    
    def backward(self, d):
        self.grad += (self.z > 0).astype(float) 

def relu(z: np.ndarray) -> np.ndarray :
    a = np.maximum(0, z)
    return a

def relu_derivative(z: np.ndarray) -> np.ndarray :
    return (z > 0).astype(float)

def _neg_sigmoid(z):
    return 1 / (1 + np.exp(-z))

def _pos_sigmoid(z):
    e = np.exp(z)
    return e / (e + 1)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return stable_sigmoid(x)

def stable_sigmoid(x):
    positive_mask = x >= 0
    negative_mask = ~positive_mask
    result = np.empty_like(x)

    result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
    exp_x = np.exp(x[negative_mask])
    result[negative_mask] = exp_x = exp_x / (1 + exp_x)
    
    return result

def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    return a * (1 - a)

def tanh(z: np.ndarray) -> np.ndarray:
    e = np.exp(2 * z)
    a = (e - 1) / (e + 1)
    return a

def tanh_derivative(a: np.ndarray) -> np.ndarray:
    return 1 - a ** 2

def softmax(z: np.ndarray) -> np.ndarray:
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def softmax_derivative(a: np.ndarray) -> np.ndarray:
    s = a.reshape((-1, 1)) 
    return np.diagflat(a) - np.dot(s, s.T)

