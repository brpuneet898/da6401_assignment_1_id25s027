"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Callable, Dict

import numpy as np

from ann.neural_layer import NeuralLayer

from ann.optimizers import sgd, momentum, nesterov, rmsprop, adam, nadam
from ann.objective_functions import cross_entropy, mse

from ann.utils import get_dead_neurons

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

import matplotlib.pyplot as plt

from wandb import Image


class NeuralNetwork:
    def __init__(self, cli_args, **kwargs):
        self.hidden_activation = getattr(cli_args, "activation", "tanh") or "tanh"
        
        self.hidden_layers = None

        try:
            self.hidden_layers = cli_args.hidden_size      
        except Exception as e:
            pass

        try:
            self.hidden_layers = cli_args.hidden_layers
        except Exception as e:
            pass

        if not self.hidden_layers:
            self.hidden_layers =  [128,128,128,128,128]

        self.learning_rate     = getattr(cli_args, "learning_rate", 0.0001) or 0.0001
        self.optimizer         = getattr(cli_args, "optimizer", "nag") or "nag"
        self.loss_function     = getattr(cli_args, "loss", "cross_entropy") or "cross_entropy"
        self.weight_init       = getattr(cli_args, "weight_init", "xavier") or "xavier"
        self.weight_decay      = getattr(cli_args, "weight_decay", 0.0001) or 0.0001
        self.wandb_project     = getattr(cli_args, "wandb_project", "") or ""

        self.input_size        = kwargs.get("input_size", ) or 784
        self.output_size       = kwargs.get("output_size") or 10
        self.output_activation = kwargs.get("output_activation") or "softmax"
        

        assert self.hidden_activation in ["relu", "sigmoid", "softmax", "tanh"],                 f"Activation function {self.hidden_activation} must be any of ('relu', 'sigmoid', 'tanh', 'softmax')"
        assert self.weight_init       in ["zero", "zeros","random", "xavier"],                   "Weight initialization method must be one of ('zero', 'zeros', 'random', 'xavier')"
        assert self.optimizer         in ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], "Optimizer must be one of ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']"
        assert self.loss_function     in ["cross_entropy", "mse"],                               "Loss function must be one of ('cross_entropy', 'mse')"

        if self.loss_function == 'mse':
            self.output_activation = 'sigmoid'
        
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

        self.layers = []

        self.momentum_gamma = 0.9
        if self.optimizer in ["momentum", "nag"]:
            self.momentum_gamma = 0.9
            self.velocities_w = []
            self.velocities_b = []

        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8
        self.timestep = 1  
        if self.optimizer in ['rmsprop', 'adam', 'nadam']:
            self.m_weights = []
            self.v_weights = []
            self.m_biases  = []
            self.v_biases  = []


        for i in range(len(self.layer_sizes)-1):

            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i+1]
            if (i+1 == len(self.layer_sizes)-1):
                activation = self.output_activation
                layer = "output"
            else:
                activation = self.hidden_activation
                layer = "hidden"

            if i == 0:
                layer = "input"

            self.layers.append(NeuralLayer(n_in, n_out, wi=self.weight_init, activation=activation, layer=layer))

            if self.optimizer in ["momentum", "nag"]:
                self.velocities_w.append(np.zeros_like(self.layers[-1].weights))
                self.velocities_b.append(np.zeros_like(self.layers[-1].biases))
            elif self.optimizer in ['rmsprop', 'adam', 'nadam']:
                self.m_weights.append(np.zeros_like(self.layers[-1].weights))
                self.v_weights.append(np.zeros_like(self.layers[-1].weights))
                self.m_biases.append(np.zeros_like(self.layers[-1].biases))
                self.v_biases.append(np.zeros_like(self.layers[-1].biases))

    def forward(self, x):
        a = x

        for i in range(len(self.layers)):
            a, logits = self.layers[i].forward(a)

        return logits
    
    def backward(self, y_true, y_pred):      
        if self.loss_function == "cross_entropy" and self.output_activation == "softmax":
            delta = y_pred - y_true
        elif self.loss_function == "mse":
            delta = (y_pred - y_true) * (y_pred * (1 - y_pred))


        grad_W, grad_b = [], []

        self.layers[-1].grad_W =  delta.T @ self.layers[-2].activations
        self.layers[-1].grad_b =  delta.sum(axis=0, keepdims=True).T

        grad_W.append(self.layers[-1].grad_W)
        grad_b.append(self.layers[-1].grad_b)

        for i in reversed(range(len(self.layers)-1)):
            delta = (delta @ self.layers[i+1].weights.T) * self.layers[i].activation_function[1](self.layers[i].activations)


            self.layers[i].backward(delta)

            gw, gb = self.layers[i].grads()

            grad_W.append(gw)
            grad_b.append(gb)


        self.grad_W = np.empty(len(grad_W), dtype=object)
        self.grad_b = np.empty(len(grad_b), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W, grad_b)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb
        return self.grad_W, self.grad_b

    def update_weights(self):
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        eps = self.epsilon
        mg = self.momentum_gamma
        ts = self.timestep; self.timestep += 1
        i = 0

        for layer in self.layers:
            weights = layer.parameters()
            grads = layer.grads()

            if self.optimizer == "sgd":
                new_weights, new_biases = sgd(weights, grads, lr)

            elif self.optimizer == "momentum":
                new_weights, new_biases, self.velocities_w[i], self.velocities_b[i] = momentum(weights, grads, self.velocities_w[i], self.velocities_b[i], lr, mg)

            elif self.optimizer == "nag":
                new_weights, new_biases, self.velocities_w[i], self.velocities_b[i] = nesterov(weights, grads, self.velocities_w[i], self.velocities_b[i], lr, mg)
            
            elif self.optimizer == "rmsprop":
                new_weights, new_biases, self.v_weights[i], self.v_biases[i] = rmsprop(weights, grads, self.v_weights[i], self.v_biases[i], lr, b2, eps) # weights, grads, v_weights, v_biases, lr, beta2, epsilon
            
            
            elif self.optimizer == "adam":
                new_weights, new_biases, self.m_weights[i], self.v_weights[i], self.m_biases[i], self.v_biases[i] = adam(weights, grads, self.m_weights[i], self.v_weights[i], self.m_biases[i], self.v_biases[i], lr, b1, b2, ts, eps)
            
            elif self.optimizer == "nadam":
                new_weights, new_biases, self.m_weights[i], self.v_weights[i], self.m_biases[i], self.v_biases[i] = nadam(weights, grads, self.m_weights[i], self.v_weights[i], self.m_biases[i], self.v_biases[i], lr, b1, b2, ts, eps)
            
            
            if self.weight_decay > 0:
                new_weights -= lr * self.weight_decay * new_weights
                new_biases -= lr * self.weight_decay * new_biases

            layer.update_weights(new_weights, new_biases)
            i = i + 1

    
    def train(self, 
              x_train: np.ndarray, 
              y_train: np.ndarray, 
              epochs: int, 
              batch_size: int=8,
              x_val: np.ndarray=None,
              y_val: np.ndarray=None,
              wandb=None
            )->None:
        
        n_samples = x_train.shape[0]

        if batch_size is None:
            batch_size = n_samples

        validate = False
        if x_val is not None and y_val is not None:
            validate = True
        

        train_losses = []
        val_losses = []
        train_scores = []
        val_scores = []
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            grad_norms = []
            for start_idx in range(0, n_samples, batch_size):
                if self.optimizer == "nag":
                    i = 0
                    for layer in self.layers:
                        layer.weights -= self.momentum_gamma * self.velocities_w[i]
                        layer.biases -= self.momentum_gamma * self.velocities_b[i]
                        i = i + 1

                end_idx = start_idx + batch_size
            
                batch_indices = indices[start_idx:end_idx]

                x, y = x_train[batch_indices], y_train[batch_indices]
                logits = self.forward(x)

                y_pred = self.layers[-1].activation_function[0](logits)

                self.backward(y, y_pred)
                grad_norms.append(np.linalg.norm(self.layers[1].grad_W))
                self.update_weights()


            train_metrics = self.evaluate(x_train, y_train)
            train_loss=train_metrics["loss"]
            train_accuracy=train_metrics["accuracy"]
            train_losses.append(train_loss)
            train_scores.append(train_accuracy)            

            if validate:
                val_metrics = self.evaluate(x_val, y_val)
                val_loss = val_metrics["loss"]
                val_accuracy = val_metrics["accuracy"]
                val_losses.append(val_loss)
                val_scores.append(val_accuracy)

            if wandb:
                log = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_recall": train_metrics["recall"],
                    "train_f1": train_metrics["f1"],
                    "train_precision": train_metrics["precision"],
                }

                image = self.wandb_plot(grad_norms, title=f"Epoch {epoch+1}", xlabel="Iterations", ylabel="Grad Norm")
                
                wandb.log({f"gradient_image_epoch_{epoch+1}": image})

                if validate:
                    log["val_accuracy"] = val_accuracy
                    log["val_loss"] = val_loss
                    log["overfitting_gap"] = train_loss - val_loss
                    log["train_recall"]= val_metrics["recall"],
                    log["train_f1"]= val_metrics["f1"],
                    log["train_precision"]= val_metrics["precision"] 

                wandb.log(log)

            if epoch % 5 == 0 or epoch == epochs - 1:
                val = f"| Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}" if validate else ""
                print(
                    f"Epoch {epoch+1:3d}/{epochs} : "
                    f"Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy}",
                    val
                    )

        if wandb:
            final_metric = {
                "final_train_loss": train_losses[-1],
                "final_train_accuracy": train_scores[-1]
            }

            if validate:
                final_metric["final_val_loss"] = val_losses[-1]
                final_metric["best_val_loss"] = np.min(val_losses)
                final_metric["convergence_epoch"] = np.argmin(val_losses)
                final_metric["final_val_accuracy"] = val_scores[-1]
  
            wandb.log(final_metric)

        
    def predict(self, x_test: np.ndarray)->np.ndarray:
        logits = self.forward(x_test)

        preds = self.layers[-1].activation_function(logits)

        return preds.argmax(axis=1)

    def evaluate(self, X, y)->Dict[str,float]:     
        logits = self.forward(X)

        y_pred = self.layers[-1].activation_function[0](logits)

        loss_func = globals()[self.loss_function]
        loss = loss_func(y, y_pred)

        accuracy = self.accuracy_score(y, y_pred)

        y_true = np.argmax(y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        precision = precision_score(y_true, y_pred, average="micro")
        recall = recall_score(y_true,y_pred, average="micro")
        f1 = f1_score(y_true, y_pred, average="micro")
        return {"accuracy": float(accuracy), "recall": recall, "f1": f1, "precision": precision, "loss": float(loss), "logits": logits, "y_true": y_true, "y_pred": y_pred}


    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.weights.copy() 
            d[f"b{i}"] = layer.biases.copy() 
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key in weight_dict:
                layer.weights = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.biases = weight_dict[b_key].copy()
    

    def accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray)-> float:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        return np.mean(y_true==y_pred) * 100
    
    def wandb_imshow(self, x: np.ndarray, title=""):
        x[np.isnan(x)] = 0
        mn = np.min(x)
        mx = np.max(x)
        try:
            x=((x - mn) / (mx - mn)) * 255
        except Exception as e:
            pass
        if title:
            plt.title(title)

        image = plt.imshow(x, cmap="gray", aspect="auto")
        return image

    def wandb_plot(self, x, y=None, title="", xlabel = "", ylabel=""):
        if title:
            plt.title(title)
        if xlabel and ylabel:
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
        
        if y is not None:
            image = Image(plt.plot(x, y)[0])
        else:
            image = Image(plt.plot(x)[0])

        plt.clf()

        return image