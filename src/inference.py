"""
Inference Script
Evaluate trained models on test sets
"""

import ast
import argparse
import numpy as np
import json
from keras.utils import to_categorical
from ann.neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from utils.data_loader import load_dataset
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("--model_path", help="Path to saved model weights", type=str)
    parser.add_argument("--dataset", default="mnist",  choices=["mnist", "fashion_mnist"], help="Dataset to evaluate on", type=str)
    parser.add_argument("--batch_size", help="Batch size for inference", type=int)
    parser.add_argument("--hidden_layers", help="List of hidden layer sizes", type=str)
    parser.add_argument("--num_neurons", help="Number of neurons in hidden layers", type=int)
    parser.add_argument("--activation", help="Activation function ('relu', 'sigmoid', 'tanh')", type=str)
    return parser.parse_args()


def load_model(model_path):
    cfg = None
    config = Config()
    try:
        weights = np.load("src/"+model_path, allow_pickle=True)
        with open("src/best_config.json", "r") as f:
            cfg = json.load(f)
            config = Config(**cfg)

    except Exception as e:
        weights = np.load(model_path, allow_pickle=True)
        with open("best_config.json", "r") as f:
            cfg = json.load(f)
            config = Config(**cfg)

    model = NeuralNetwork(config)
    model.set_weights(weights.tolist())

    return model


def evaluate_model(model, X_test, y_test): 
    evals = model.evaluate(X_test, y_test)
    return evals


def main():
    args = parse_arguments()
    dataset = getattr(args, "dataset", "mnist")
    (train_images, train_labels), (test_images, test_labels) = load_dataset(dataset)

    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_images = train_images.reshape((-1, 28 * 28))
    test_images = test_images.reshape((-1, 28 * 28))

    output_size = len(np.unique(train_labels).tolist())

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size = 0.1, random_state = 42) 
    model_path = args.model_path
    model = load_model(model_path=model_path)
    evals = evaluate_model(model, test_images, test_labels)
    print(evals)
    return evals

if __name__ == '__main__':
    main()
