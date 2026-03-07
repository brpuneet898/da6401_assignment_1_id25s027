# DA6401 -- Assignment 1

## Multi-Layer Perceptron for Image Classification (NumPy Implementation)

This repository contains the implementation for **Assignment 1 of DA6401
-- Introduction to Deep Learning (IIT Madras)**.

The objective of this assignment is to **implement a fully configurable
Multi-Layer Perceptron (MLP) from scratch using only NumPy** and train
it for **image classification on the MNIST and Fashion-MNIST datasets**.

The implementation includes the **complete deep learning pipeline**,
including:

-   Forward propagation
-   Backpropagation
-   Multiple optimization algorithms
-   Weight initialization strategies
-   Modular neural network architecture
-   Experiment tracking with **Weights & Biases**

The project strictly follows the official assignment skeleton and
adheres to the structural requirements specified in the assignment
instructions.

------------------------------------------------------------------------

# Important Links

## W&B Project Report

https://api.wandb.ai/links/id25s027-iit-madras/yj5fbfmq

## GitHub Repository

https://github.com/brpuneet898/da6401_assignment_1_id25s027

The W&B report includes detailed experimental analysis including:

-   Dataset exploration
-   Hyperparameter sweeps
-   Optimizer comparisons
-   Gradient analysis
-   Dead neuron investigation
-   Loss function comparison
-   Error analysis
-   Fashion-MNIST transfer experiments

------------------------------------------------------------------------

# Repository Structure

    da6401_assignment_1_id25s027
    │
    ├── src
    │   │
    │   ├── ann
    │   │   ├── activations.py
    │   │   ├── neural_layer.py
    │   │   ├── neural_network.py
    │   │   ├── objective_functions.py
    │   │   ├── optimizers.py
    │   │   └── utils.py
    │   │
    │   ├── utils
    │   │   └── data_loader.py
    │   │
    │   ├── train.py
    │   ├── inference.py
    │   │
    │   ├── best_model.npy
    │   ├── best_config.json
    │   │
    │   ├── wandb_sweep.ipynb
    │   └── wandb_report.ipynb
    │
    ├── requirements.txt
    ├── README.md
    └── .gitignore

### Folder Description

**ann/**\
Contains the core neural network implementation including activation
functions, layers, optimizers, and the neural network class.

**utils/**\
Contains utility functions such as dataset loading and preprocessing.

**train.py**\
Main training script for the neural network. Supports configurable
hyperparameters via CLI.

**inference.py**\
Loads the saved model weights and evaluates model performance using
standard classification metrics.

------------------------------------------------------------------------

# Model Overview

The implemented architecture is a **fully connected Multi-Layer
Perceptron (MLP)**.

### Supported Activation Functions

-   Sigmoid
-   Tanh
-   ReLU

### Supported Loss Functions

-   Mean Squared Error (MSE)
-   Cross Entropy

### Supported Optimizers

-   SGD
-   Momentum
-   Nesterov Accelerated Gradient (NAG)
-   RMSProp

### Weight Initialization

-   Random initialization
-   Xavier initialization

------------------------------------------------------------------------

# Installation

Clone the repository:

``` bash
git clone https://github.com/brpuneet898/da6401_assignment_1_id25s027.git
cd da6401_assignment_1_id25s027
```

Create a virtual environment (optional):

``` bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Training the Model

Run training using:

``` bash
python src/train.py
```

Example command:

``` bash
python src/train.py --dataset mnist --epochs 10 --batch_size 64 --loss cross_entropy --optimizer rmsprop --learning_rate 0.001 --num_layers 3 --hidden_size 128 --activation relu --weight_init xavier --wandb_project da6401-assignment1
```

Training runs are logged automatically to **Weights & Biases**.

------------------------------------------------------------------------

# Hyperparameter Sweep

Hyperparameter sweeps were conducted using **Weights & Biases Sweeps**.

Run sweeps using:

``` bash
jupyter notebook src/wandb_sweep.ipynb
```

The sweep explores:

-   optimizers
-   learning rates
-   hidden layer sizes
-   activation functions
-   weight initialization

More than **100 runs** were executed to determine the best
configuration.

------------------------------------------------------------------------

# Libraries Used

The implementation relies on the following libraries:

-   NumPy
-   matplotlib
-   scikit-learn
-   keras.datasets
-   wandb

Deep learning frameworks such as **PyTorch, TensorFlow, or JAX were
intentionally not used** as required by the assignment.

------------------------------------------------------------------------

# Author

Puneet (ID25S027)\
DA6401 -- Introduction to Deep Learning\
IIT Madras

------------------------------------------------------------------------

# License

This repository is intended strictly for **academic submission and
evaluation purposes**.
