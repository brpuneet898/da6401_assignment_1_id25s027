"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

def cross_entropy(y: np.ndarray, y_pred: np.ndarray) -> float:
    # this is the function that defines cross entropy loss function
    # this will be particularly useful for the multi-class classification problems
    # currently we will be using datasets like MNIST digits and MNIST fashion 
    # they both are multiclass classfication task, so this will be highly useful
    # this will use softmax activation function in the output layer 
    # it will work on the logits and give out the probabilities.
    return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis = 1))

def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    # this is the function that defines MSE, this is particularly useful for the 
    # regression problems.
    return np.mean((y - y_pred) ** 2)
