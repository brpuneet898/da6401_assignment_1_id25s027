"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

class ReLU():
    # we are implementing the relu activation as a class, it stores the input during the forward pass so that the 
    # gradient can be computed during the backward propogation.
    def __init__(self, z):
        # initiating the relu object
        self.z = z
        self.grad = 0.0

    def forward(self):
        # forward pass of the relu function, here we compute the output, which is the max of 0 or input z.
        self.a = np.maximum(0, self.z)
        return self.a
    
    def backward(self, d):
        # backward pass of the relu function, here we compute the gradients of the relu wrt to z. 
        # if z is more than 0, the gradient is 1, else it will be considered as 0. 
        self.grad += (self.z > 0).astype(float) 

def relu(z: np.ndarray) -> np.ndarray :
    # this is the function of relu which takes in the array or input tensor and gives the activated result 
    a = np.maximum(0, z)
    return a

def relu_derivative(z: np.ndarray) -> np.ndarray :
    # here we are computing the derivative wrt relu function, if the z is more than 0, the first order derivative is 1
    # otherwise it is 0.
    return (z > 0).astype(float)

def _neg_sigmoid(z):
    # this will act as a helper function for the sigmoid function specially when the z is negative.
    return 1 / (1 + np.exp(-z))

def _pos_sigmoid(z):
    # this will act as a helper function for the sigmoid function specially when the z is positive.
    e = np.exp(z)
    return e / (e + 1)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # this is just a sigmoid wrapper which will help in numerically stable computations
    return stable_sigmoid(x)

def stable_sigmoid(x):
    # since we have already defined the helper functions for the sigmoid, we use them here to prevent any kind of 
    # overflow or underflow issues. 
    positive_mask = x >= 0
    negative_mask = ~positive_mask
    result = np.empty_like(x)

    result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
    exp_x = np.exp(x[negative_mask])
    result[negative_mask] = exp_x = exp_x / (1 + exp_x)
    
    return result

def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    # here we are define the derivative of the sigmoid function. 
    return a * (1 - a)

def tanh(z: np.ndarray) -> np.ndarray:
    # this is the hyperbolic tangent activation function 
    e = np.exp(2 * z)
    a = (e - 1) / (e + 1)
    return a

def tanh_derivative(a: np.ndarray) -> np.ndarray:
    # in here we define the derivative of the tanh function.
    return 1 - a ** 2

def softmax(z: np.ndarray) -> np.ndarray:
    # here we are defining the softmax function
    # it is applicable for multi-class distribution
    # it converts the logits coming from the last layer and converts it to the probability distribution
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / exps.sum(axis=1, keepdims=True)


def softmax_derivative(a: np.ndarray) -> np.ndarray:
    # here we are defining the derivative of the softmax function.
    # we actually calculate the jacobian matrix of the softmax output. It helps to simplify the gradient prorpogation.
    s = a.reshape((-1, 1)) 
    return np.diagflat(a) - np.dot(s, s.T)

