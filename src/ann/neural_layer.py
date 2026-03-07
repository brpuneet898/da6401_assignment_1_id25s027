"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np

try:
    from ann.activations import *
except:
    from activations import *


# we will defining these activation function in this file.

activation_functions = {
    "relu": [relu, relu_derivative],
    "sigmoid": [sigmoid, sigmoid_derivative],
    "softmax": [softmax, softmax_derivative],
    "tanh": [tanh, tanh_derivative]
}


class NeuralLayer():
    ## first we define the neural layer with its input, output, weights, biases, activation functions.
    def __init__(self, n_in: int, n_out: int, wi: str, activation: str, layer: str):
        self.layer = layer
        self.wi = wi
        self.weights = []
        self.biases = []
        self.n_in = n_in
        self.n_out = n_out
        self.activation_function = activation_functions[activation]
        self.activations = None

        self.logits = None

        self.weight_init()

        self.zero_grad()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        
        self.pre_activations = x

        z = x @ self.weights + self.biases 
        self.activations = self.activation_function[0](z)  

        return self.activations, z
    
    def __repr__(self):
        ## again a helper function that will be used in the printing stuff while doing wandb
        return f"NeuralLayer<layer:{self.layer}|n_in:{self.n_in}|n_out:{self.n_out}|activation:{self.activation_function[0].__name__}>"
    
    def forward(self, x: np.ndarray)-> np.ndarray:
        ## this is the simple forward pass
        return self(x)
    
    def backward(self, delta: np.ndarray)-> np.ndarray:
        ## here we will do the backpropagation, the gradietns wrt to both weights and biases
        self.grad_W =  delta.T @  self.pre_activations 
        self.grad_b = delta.sum(axis=0, keepdims=True).T

        assert self.grad_W.shape == (self.n_out, self.n_in)
        assert self.grad_b.shape == (self.n_out,1)
        
    def weight_init(self):
        ## initalizing the weights here with 0, random and xavier init. 
        np.random.seed(0)
        if self.wi in ["zero", "zeros"]:
            self.weights = np.zeros((self.n_in, self.n_out))

        elif self.wi == "random":
            self.weights = np.random.randn(self.n_in, self.n_out) * 0.01
            
        elif self.wi == "xavier":
            limit = np.sqrt(6/(self.n_in + self.n_out))
            low, high = -limit, limit
            size = (self.n_in, self.n_out)
            self.weights = np.random.uniform(low, high, size=size)
        else:
            raise ValueError("Unsupported initialization method. Choose from ['zero', 'zeros', 'random', 'xavier']")

        self.biases = np.zeros((1, self.n_out))

    def update_weights(self, weights: np.ndarray, biases: np.ndarray)->None:
        ## the function defined to updated the weights and biases after the backprop
        ## method .
        assert weights.shape == self.weights.shape
        assert biases.shape == self.biases.shape
        self.weights = weights
        self.biases = biases

    def zero_grad(self)->None:
        ## function to zeroing out the gradients after the one step is completed.
        self.grad_W = np.zeros_like(self.weights.T)
        self.grad_b = np.zeros_like(self.biases.T)

    
    def parameters(self)->tuple[np.ndarray]:
        return self.weights, self.biases
    
    def grads(self)->tuple[np.ndarray]:

        assert self.grad_W.shape == self.weights.T.shape
        assert self.grad_b.shape == self.biases.T.shape
        return self.grad_W, self.grad_b