"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))
import ast
import wandb
import argparse
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from utils.data_loader import load_dataset
from ann.neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument("--dataset", "-d", default="mnist", choices=["mnist", "fashion_mnist"], help="'mnist' or 'fashion_mnist'", type=str)
    parser.add_argument("--epochs", "-e", default=5,  help="Number of training epochs", type=int)
    parser.add_argument("--batch_size", "-b", default=64,  help="Mini-batch Size: any of [8, 16, 32, 64, 128]", type=int)
    parser.add_argument("--learning_rate", "-lr", default=0.001, help="Learning rate for optimizer", type=float)
    parser.add_argument("--optimizer", "-o", default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'", type=str)
    parser.add_argument("--loss", "-l", default="cross_entropy", choices=["cross_entropy", "mse", "mean_squared_error"], help="Loss function ('cross_entropy', 'mse', 'mean_squared_error')", type=str)
    parser.add_argument("--activation", "-a", default="sigmoid", choices=["sigmoid", "tanh", "relu"], help="Activation function ('relu', 'sigmoid', 'tanh')", type=str)
    parser.add_argument("--hidden_layers", 
                        "-sz",
                        type=int,
                        nargs="+",
                        help="List of hidden layer sizes"
                        )

    parser.add_argument("--hidden_size",
                        dest="hidden_layers",
                        type=int,
                        nargs="+",
                        help='Sizes of the hidden layers'
                        )

    parser.add_argument("--num_neurons", 
                        help="Number of neurons in hidden layers", 
                        type=int)
    
    parser.add_argument("--num_layers", 
                        "-nhl", 
                        help="List of hidden layer sizes")
    
    
    parser.add_argument("--weight_init", 
                        "-w_i", 
                        "-wi", 
                        help="Weight initialization method ('zero' or 'random' or 'xavier')", 
                        type=str)
    
    parser.add_argument("--weight_decay", 
                        "-wd",
                        "-w_d", 
                        help="Weight decay for L2 regularization", 
                        type=float)  
    
    parser.add_argument("--wandb_project", 
                        "-wp", default="", 
                        help="W&B project name",type=str)
    
    parser.add_argument("--model_save_path", 
                        help="Path to save trained model (do not give absolute path, rather provide relative path)", 
                        type=str)
    
    parser.add_argument("--question", 
                        help="Question id or header string", 
                        type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()

    dataset = args.dataset

    (train_images, train_labels), (test_images, test_labels) = load_dataset(dataset)
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = train_images.reshape((-1, 28 * 28))
    test_images = test_images.reshape((-1, 28 * 28))

    output_size = len(np.unique(train_labels).tolist())
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, test_size = 0.1, random_state = 42) 

    output_activation = "softmax"

    if args.loss == "mse":
        output_activation = "sigmoid"
 
    model = NeuralNetwork(
        cli_args=args,
        input_size = train_images[0].shape[0],
        output_size = output_size,
        output_activation = output_activation
        )
    
    run = None
    if args.wandb_project:
        run = wandb.init(
            project = args.wandb_project
        )
        run_name = f"lr={args.learning_rate}_bs={args.batch_size}_opt={args.optimizer}_act={args.activation}_loss={args.loss}_wd={args.weight_decay}_wi={args.weight_init}_sz={args.hidden_layers}_wandbid={wandb.run.id}"
        if args.question:
            run_name = "[" + args.question + "]-" + run_name
        wandb.run.name = run_name
    
    epochs = args.epochs
    batch_size = args.batch_size
    model.train(x_train, y_train, x_val=x_val, y_val=y_val, epochs=epochs, batch_size=batch_size, wandb=run)

    if args.model_save_path:
        np.save(args.model_save_path.split(".npy")[0], model.get_weights(), allow_pickle=True)
        config = vars(args)
        with open("src/best_config.json", "w") as f:
            json.dump(config, f)
 
if __name__ == '__main__':
    main()