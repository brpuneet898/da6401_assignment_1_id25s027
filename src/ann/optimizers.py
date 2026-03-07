"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np


def sgd(weights, grads, lr):
    ## here we are defining sgd optimizer
    # as learnt in the class stochastic gradient descent takes 1 by e
    # it is bit slower but has more guarantee of convergence
    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    w -= lr * (w_grad)
    b -= lr * b_grad

    return w, b

def momentum(weights, grads, w_velocities, b_velocities, lr, momentum_gamma):
    ## from here we move from just stochasticity to the notion of momentum
    ## momentum based gradient descent have the notion of velocities

    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    w_velocities = momentum_gamma * w_velocities + lr * w_grad
    b_velocities = momentum_gamma * b_velocities + lr * b_grad

    w -= w_velocities
    b -= b_velocities

    return w, b, w_velocities, b_velocities


def nesterov(weights, grads, w_velocities, b_velocities, lr, momentum_gamma):
    ## it is one more step ahead than the momentum based , comes under the same family
    ## but litte bit better
    w, b = weights 
    w_grad, b_grad = grads
    w_grad, b_grad = w_grad.T, b_grad.T

    w_velocities = momentum_gamma * w_velocities + lr * w_grad
    b_velocities = momentum_gamma * b_velocities + lr * b_grad

    w -= w_velocities
    b -= b_velocities

    return w, b, w_velocities, b_velocities


def rmsprop(weights, grads, v_weights, v_biases, lr, beta2, epsilon):
    ## from here we move to the adaptive type of gradients 
    ## you can see the difference we will be using 2 betas
    ## and square of the gradietns. 
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
    ## proper adaptive based gradient descent, this is the most commonly used optimizer in the deep learning community
    ## most of the real-world applications these days start with adam these days
    ## now the learning rate starts to adapt based on the gradients and where it is standing
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
    ## this is something different like it encompasses the best of two worlds
    ## it takes in nesterov and also the adam
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