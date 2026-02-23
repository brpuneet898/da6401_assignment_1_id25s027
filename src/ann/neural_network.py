"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from __future__ import annotations
import numpy as np

from .neural_layer import NeuralLayer
from .objective_functions import get_loss
from .optimizers import get_optimizer
from .activations import softmax


class NeuralNetwork:
    def __init__(self, cli_args):
        self.input_dim = int(getattr(cli_args, "input_dim", 784))
        self.num_classes = int(getattr(cli_args, "num_classes", 10))

        activation = getattr(cli_args, "activation", "relu")
        weight_init = getattr(cli_args, "weight_init", "xavier")

        hl = getattr(cli_args, "hidden_layers", None)
        nn = getattr(cli_args, "num_neurons", None)

        hidden_sizes = None
        if isinstance(hl, (list, tuple)) and len(hl) > 0 and all(isinstance(x, (int, np.integer)) for x in hl):
            hidden_sizes = [int(x) for x in hl]
        elif isinstance(hl, (int, np.integer)) and nn is not None:
            hidden_sizes = [int(nn)] * int(hl)
        elif hl is None and nn is not None:
            hidden_sizes = [int(nn)]  
        else:
            hidden_sizes = [128] 

        self.hidden_sizes = hidden_sizes

        self.loss_name = getattr(cli_args, "loss", "cross_entropy")
        self.loss_fn = get_loss(self.loss_name)

        lr = float(getattr(cli_args, "learning_rate", 1e-3))
        opt_name = getattr(cli_args, "optimizer", "sgd")
        wd = float(getattr(cli_args, "weight_decay", 0.0))
        self.optimizer = get_optimizer(opt_name, learning_rate=lr, weight_decay=wd)

        seed = int(getattr(cli_args, "seed", 42))
        sizes = [self.input_dim] + self.hidden_sizes + [self.num_classes]
        self.layers = []
        for i in range(len(sizes) - 1):
            in_f, out_f = sizes[i], sizes[i + 1]
            is_last = (i == len(sizes) - 2)
            act = "linear" if is_last else activation
            self.layers.append(
                NeuralLayer(
                    in_features=in_f,
                    out_features=out_f,
                    activation=act,
                    weight_init=weight_init,
                    seed=seed + i,
                )
            )

        self._last_logits = None  

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        self._last_logits = out
        return out

    def backward(self, y_true, y_pred_logits):
        loss, dlogits, _ = self.loss_fn(y_pred_logits, y_true)

        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)

        grad_w = [layer.grad_W for layer in self.layers]
        grad_b = [layer.grad_b for layer in self.layers]
        return loss, grad_w, grad_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    def predict(self, X):
        logits = self.forward(X)
        probs = softmax(logits)
        return np.argmax(probs, axis=1)

    def train(self, X_train, y_train, epochs, batch_size):
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(int) if y_train.ndim == 1 else y_train.astype(np.float32)

        n = X_train.shape[0]
        history = {"loss": [], "acc": []}

        for ep in range(int(epochs)):
            idx = np.arange(n)
            np.random.shuffle(idx)
            Xs = X_train[idx]
            ys = y_train[idx]

            total_loss = 0.0
            correct = 0
            seen = 0

            for start in range(0, n, int(batch_size)):
                xb = Xs[start:start + int(batch_size)]
                yb = ys[start:start + int(batch_size)]

                logits = self.forward(xb)
                loss, _, _ = self.backward(yb, logits)
                self.update_weights()

                total_loss += loss * xb.shape[0]
                preds = self.predict(xb)
                y_idx = yb if yb.ndim == 1 else np.argmax(yb, axis=1)
                correct += int(np.sum(preds == y_idx))
                seen += xb.shape[0]

            history["loss"].append(total_loss / max(seen, 1))
            history["acc"].append(correct / max(seen, 1))

        return history

    def evaluate(self, X, y):
        X = X.astype(np.float32)
        y = y.astype(int) if y.ndim == 1 else y.astype(np.float32)

        logits = self.forward(X)
        loss, _, probs = self.loss_fn(logits, y)

        preds = np.argmax(probs, axis=1)
        y_idx = y if y.ndim == 1 else np.argmax(y, axis=1)
        acc = float(np.mean(preds == y_idx))

        return {"logits": logits, "loss": float(loss), "accuracy": acc}