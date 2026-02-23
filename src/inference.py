"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.utils.data_loader import load_dataset, batch_iterator
from src.ann.neural_network import NeuralNetwork
from src.ann.objective_functions import get_loss
from src.ann.activations import softmax

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument(
        "--model_path",
        type=str,
        default="models/best_model.npy",
        help="Path to saved model weights (prefer relative path)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "fashion-mnist", "fmnist"],
        help="Dataset to evaluate on",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Batch size for inference")

    parser.add_argument(
        "--hidden_layers",
        "--nhl",
        type=int,
        default=2,
        help="Number of hidden layers (used with --num_neurons)",
    )
    parser.add_argument(
        "--num_neurons",
        "--nn",
        type=int,
        default=128,
        help="Neurons per hidden layer (used with --hidden_layers)",
    )
    parser.add_argument(
        "--hidden_sizes",
        "--sz",
        type=int,
        nargs="*",
        default=None,
        help="Explicit hidden layer sizes, e.g. --hidden_sizes 256 128 64 (overrides hidden_layers/num_neurons)",
    )
    parser.add_argument(
        "--activation",
        "-a",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh"],
        help="Activation function",
    )

    parser.add_argument("--input_dim", type=int, default=784, help="Input dimension (default 784)")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes (default 10)")

    parser.add_argument(
        "--loss",
        "-l",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "mse", "ce"],
        help="Loss function used for reporting loss",
    )

    parser.add_argument(
        "--weight_init",
        "--wi",
        type=str,
        default="xavier",
        choices=["xavier", "he", "random", "normal", "zeros"],
        help="Weight initialization (ignored after loading weights)",
    )

    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        default="sgd",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        help="Optimizer name (unused in inference, kept for compatibility)",
    )
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Unused (compatibility)")
    parser.add_argument("--weight_decay", "--wd", type=float, default=0.0, help="Unused (compatibility)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (compatibility)")

    return parser.parse_args()

def load_model(model_path: str):
    obj = np.load(model_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict) or "layers" not in obj:
        raise ValueError("Invalid model file format. Expected dict with key 'layers'.")
    return obj

def _set_weights(model: NeuralNetwork, weights_dict: Dict[str, Any]) -> None:
    layers = weights_dict["layers"]
    if len(layers) != len(model.layers):
        raise ValueError(
            f"Loaded weights have {len(layers)} layers but model expects {len(model.layers)} layers."
        )
    for layer_obj, w in zip(model.layers, layers):
        layer_obj.W = w["W"].astype(np.float32)
        layer_obj.b = w["b"].astype(np.float32)

def evaluate_model(model: NeuralNetwork, X_test: np.ndarray, y_test: np.ndarray, batch_size: int, loss_name: str):
    loss_fn = get_loss("cross_entropy" if loss_name == "ce" else loss_name)

    all_logits = []
    all_preds = []
    all_true = []

    losses = []

    for xb, yb in batch_iterator(X_test, y_test, batch_size=int(batch_size), shuffle=False):
        logits = model.forward(xb)
        loss, _, probs = loss_fn(logits, yb)

        preds = np.argmax(probs, axis=1)

        all_logits.append(logits)
        all_preds.append(preds)
        all_true.append(yb)
        losses.append(loss)

    logits_full = np.vstack(all_logits) if all_logits else np.zeros((0, model.num_classes), dtype=np.float32)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)

    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return {
        "logits": logits_full,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }

def main():
    args = parse_arguments()

    ds = args.dataset.lower().strip()
    if ds in ["fashion-mnist", "fmnist"]:
        ds = "fashion_mnist"
    args.dataset = ds

    (_, _), (_, _), (X_test, y_test) = load_dataset(
        dataset=args.dataset,
        val_split=0.1,          
        seed=args.seed,
        flatten=True,
        normalize=True,
        one_hot=False,
    )

    model = NeuralNetwork(args)

    weights_dict = load_model(args.model_path)
    _set_weights(model, weights_dict)

    results = evaluate_model(model, X_test, y_test, batch_size=args.batch_size, loss_name=args.loss)

    print("Evaluation complete!")
    print(
        f"loss={results['loss']:.6f} | "
        f"acc={results['accuracy']:.4f} | "
        f"prec={results['precision']:.4f} | "
        f"rec={results['recall']:.4f} | "
        f"f1={results['f1']:.4f}"
    )

    return results

if __name__ == "__main__":
    main()