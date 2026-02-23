"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

from __future__ import annotations
import numpy as np


class OptimizerBase:
    def step(self, layers):
        raise NotImplementedError


class SGD(OptimizerBase):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0):
        self.lr = float(lr)
        self.wd = float(weight_decay)

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "W"):
                continue
            gW = layer.grad_W + self.wd * layer.W
            gb = layer.grad_b
            layer.W -= self.lr * gW
            layer.b -= self.lr * gb


class Momentum(OptimizerBase):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr = float(lr)
        self.mu = float(momentum)
        self.wd = float(weight_decay)
        self.vW = {}
        self.vb = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            if not hasattr(layer, "W"):
                continue
            key = id(layer)
            if key not in self.vW:
                self.vW[key] = np.zeros_like(layer.W, dtype=np.float32)
                self.vb[key] = np.zeros_like(layer.b, dtype=np.float32)

            gW = layer.grad_W + self.wd * layer.W
            gb = layer.grad_b

            self.vW[key] = self.mu * self.vW[key] + gW
            self.vb[key] = self.mu * self.vb[key] + gb

            layer.W -= self.lr * self.vW[key]
            layer.b -= self.lr * self.vb[key]


class NAG(OptimizerBase):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr = float(lr)
        self.mu = float(momentum)
        self.wd = float(weight_decay)
        self.vW = {}
        self.vb = {}

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "W"):
                continue
            key = id(layer)
            if key not in self.vW:
                self.vW[key] = np.zeros_like(layer.W, dtype=np.float32)
                self.vb[key] = np.zeros_like(layer.b, dtype=np.float32)

            gW = layer.grad_W + self.wd * layer.W
            gb = layer.grad_b

            vW_prev = self.vW[key]
            vb_prev = self.vb[key]

            self.vW[key] = self.mu * self.vW[key] + gW
            self.vb[key] = self.mu * self.vb[key] + gb

            layer.W -= self.lr * (self.mu * vW_prev + (1.0 + self.mu) * gW)
            layer.b -= self.lr * (self.mu * vb_prev + (1.0 + self.mu) * gb)


class RMSProp(OptimizerBase):
    def __init__(self, lr: float = 1e-3, beta: float = 0.9, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.sW = {}
        self.sb = {}

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "W"):
                continue
            key = id(layer)
            if key not in self.sW:
                self.sW[key] = np.zeros_like(layer.W, dtype=np.float32)
                self.sb[key] = np.zeros_like(layer.b, dtype=np.float32)

            gW = layer.grad_W + self.wd * layer.W
            gb = layer.grad_b

            self.sW[key] = self.beta * self.sW[key] + (1.0 - self.beta) * (gW * gW)
            self.sb[key] = self.beta * self.sb[key] + (1.0 - self.beta) * (gb * gb)

            layer.W -= self.lr * gW / (np.sqrt(self.sW[key]) + self.eps)
            layer.b -= self.lr * gb / (np.sqrt(self.sb[key]) + self.eps)


class Adam(OptimizerBase):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.mW, self.vW, self.mb, self.vb = {}, {}, {}, {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, "W"):
                continue
            key = id(layer)
            if key not in self.mW:
                self.mW[key] = np.zeros_like(layer.W, dtype=np.float32)
                self.vW[key] = np.zeros_like(layer.W, dtype=np.float32)
                self.mb[key] = np.zeros_like(layer.b, dtype=np.float32)
                self.vb[key] = np.zeros_like(layer.b, dtype=np.float32)

            gW = layer.grad_W + self.wd * layer.W
            gb = layer.grad_b

            self.mW[key] = self.b1 * self.mW[key] + (1.0 - self.b1) * gW
            self.vW[key] = self.b2 * self.vW[key] + (1.0 - self.b2) * (gW * gW)
            self.mb[key] = self.b1 * self.mb[key] + (1.0 - self.b1) * gb
            self.vb[key] = self.b2 * self.vb[key] + (1.0 - self.b2) * (gb * gb)

            mW_hat = self.mW[key] / (1.0 - self.b1 ** self.t)
            vW_hat = self.vW[key] / (1.0 - self.b2 ** self.t)
            mb_hat = self.mb[key] / (1.0 - self.b1 ** self.t)
            vb_hat = self.vb[key] / (1.0 - self.b2 ** self.t)

            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class Nadam(OptimizerBase):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self.wd = float(weight_decay)
        self.mW, self.vW, self.mb, self.vb = {}, {}, {}, {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, "W"):
                continue
            key = id(layer)
            if key not in self.mW:
                self.mW[key] = np.zeros_like(layer.W, dtype=np.float32)
                self.vW[key] = np.zeros_like(layer.W, dtype=np.float32)
                self.mb[key] = np.zeros_like(layer.b, dtype=np.float32)
                self.vb[key] = np.zeros_like(layer.b, dtype=np.float32)

            gW = layer.grad_W + self.wd * layer.W
            gb = layer.grad_b

            self.mW[key] = self.b1 * self.mW[key] + (1.0 - self.b1) * gW
            self.vW[key] = self.b2 * self.vW[key] + (1.0 - self.b2) * (gW * gW)
            self.mb[key] = self.b1 * self.mb[key] + (1.0 - self.b1) * gb
            self.vb[key] = self.b2 * self.vb[key] + (1.0 - self.b2) * (gb * gb)

            mW_hat = self.mW[key] / (1.0 - self.b1 ** self.t)
            vW_hat = self.vW[key] / (1.0 - self.b2 ** self.t)
            mb_hat = self.mb[key] / (1.0 - self.b1 ** self.t)
            vb_hat = self.vb[key] / (1.0 - self.b2 ** self.t)

            mW_nest = self.b1 * mW_hat + (1.0 - self.b1) * gW / (1.0 - self.b1 ** self.t)
            mb_nest = self.b1 * mb_hat + (1.0 - self.b1) * gb / (1.0 - self.b1 ** self.t)

            layer.W -= self.lr * mW_nest / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_nest / (np.sqrt(vb_hat) + self.eps)


def get_optimizer(name: str, learning_rate: float, weight_decay: float = 0.0, **kwargs) -> OptimizerBase:
    name = (name or "").lower().strip()
    if name == "sgd":
        return SGD(lr=learning_rate, weight_decay=weight_decay)
    if name == "momentum":
        return Momentum(lr=learning_rate, momentum=float(kwargs.get("momentum", 0.9)), weight_decay=weight_decay)
    if name == "nag":
        return NAG(lr=learning_rate, momentum=float(kwargs.get("momentum", 0.9)), weight_decay=weight_decay)
    if name == "rmsprop":
        return RMSProp(lr=learning_rate, beta=float(kwargs.get("beta", 0.9)), eps=float(kwargs.get("eps", 1e-8)), weight_decay=weight_decay)
    if name == "adam":
        return Adam(lr=learning_rate, beta1=float(kwargs.get("beta1", 0.9)), beta2=float(kwargs.get("beta2", 0.999)), eps=float(kwargs.get("eps", 1e-8)), weight_decay=weight_decay)
    if name == "nadam":
        return Nadam(lr=learning_rate, beta1=float(kwargs.get("beta1", 0.9)), beta2=float(kwargs.get("beta2", 0.999)), eps=float(kwargs.get("eps", 1e-8)), weight_decay=weight_decay)
    raise ValueError(f"The given optimizer: {name} is not supported. Please choose either sgd, momentum, nag, rmsprop, adam, or nadam.")