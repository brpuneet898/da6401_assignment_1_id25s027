"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np


def sgd(weights, grads, lr):
    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    w -= lr * (w_grad)
    b -= lr * b_grad

    return w, b

def momentum(weights, grads, w_velocities, b_velocities, lr, momentum_gamma):
    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    w_velocities = momentum_gamma * w_velocities + lr * w_grad
    b_velocities = momentum_gamma * b_velocities + lr * b_grad

    w -= w_velocities
    b -= b_velocities

    return w, b, w_velocities, b_velocities


def nesterov(weights, grads, w_velocities, b_velocities, lr, momentum_gamma):
    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    w_velocities = momentum_gamma * w_velocities + lr * w_grad
    b_velocities = momentum_gamma * b_velocities + lr * b_grad

    w -= w_velocities
    b -= b_velocities

    return w, b, w_velocities, b_velocities


def rmsprop(weights, grads, v_weights, v_biases, lr, beta2, epsilon):
    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    v_weights = beta2 * v_weights + (1 - beta2) * w_grad ** 2
    v_biases  = beta2 * v_biases + (1- beta2) * b_grad ** 2

    w_update = lr * w_grad / (np.sqrt(v_weights) + epsilon)
    b_update = lr * b_grad / (np.sqrt(v_biases) + epsilon)

    w -= w_update
    b -= b_update

    return w, b, v_weights, v_biases

def adam(weights, grads, mw, vw, mb, vb, lr, b1, b2, ts, eps):
    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    mw = b1 * mw + (1 - b1) * w_grad
    mb = b1 * mb + (1 - b1) * b_grad

    mwhat = mw / (1 - b1**ts)
    mbhat = mb / (1 - b1**ts)

    vw = b2 * vw + (1 - b2) * (w_grad**2)
    vb = b2 * vb + (1 - b2) * (b_grad**2)

    vwhat = vw / (1 - b2**ts)
    vbhat = vb / (1 - b2**ts)

    w_update = lr * mwhat / (np.sqrt(vwhat) + eps)
    b_update = lr * mbhat / (np.sqrt(vbhat) + eps)

    w -= w_update
    b -= b_update

    return w, b, mw, vw, mb, vb
    
def nadam(weights, grads, mw, vw, mb, vb, lr, b1, b2, ts, eps):
    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    mw = b1 * mw + (1 - b1) * w_grad
    mb = b1 * mb + (1 - b1) * b_grad

    mwhat = mw / (1 - b1**ts)
    mbhat = mb / (1 - b1**ts)

    vw = b2 * vw + (1 - b2) * (w_grad**2)
    vb = b2 * vb + (1 - b2) * (b_grad**2)

    vwhat = vw / (1 - b2**ts)
    vbhat = vb / (1 - b2**ts)

    w_update = lr * (mwhat + (b1/(1 - b1**ts)) * w_grad) / (np.sqrt(vwhat) + eps)
    b_update = lr * (mbhat + (b1/(1 - b1**ts)) * b_grad )/ (np.sqrt(vbhat) + eps)

    w -= w_update
    b -= b_update

    return w, b, mw, vw, mb, vb