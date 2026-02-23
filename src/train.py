"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
from src.utils.data_loader import load_dataset, batch_iterator
from src.ann.neural_network import NeuralNetwork
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def _safe_makedirs(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _try_init_wandb(project: str, config: Dict[str, Any]):
    try:
        import wandb
        wandb.init(project=project, config=config)
        return wandb
    except Exception:
        return None

def _extract_layer_sizes_from_args(args) -> Tuple[int, int]:
    input_dim = int(getattr(args, "input_dim", 784))
    num_classes = int(getattr(args, "num_classes", 10))
    return input_dim, num_classes

def _resolve_hidden_sizes(args) -> list:
    if getattr(args, "hidden_sizes", None):
        return [int(x) for x in args.hidden_sizes]

    hl = int(getattr(args, "hidden_layers", 2))
    nn = int(getattr(args, "num_neurons", 128))
    return [nn] * hl

def _save_model_and_config(model: NeuralNetwork, args, model_save_path: str, config_save_path: str, best_metrics: Dict[str, Any],) -> None:
    _safe_makedirs(os.path.dirname(model_save_path))
    _safe_makedirs(os.path.dirname(config_save_path))

    weights = {
        "layers": [{"W": layer.W.astype(np.float32), "b": layer.b.astype(np.float32)} for layer in model.layers]
    }
    np.save(model_save_path, weights, allow_pickle=True)

    cfg = vars(args).copy()
    cfg["hidden_sizes"] = _resolve_hidden_sizes(args)
    cfg["best_val_metrics"] = best_metrics

    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network (NumPy MLP)")

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "fashion-mnist", "fmnist"],
        help="Dataset to use",
    )
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training set to use as validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--learning_rate", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", "--wd", type=float, default=0.0, help="L2 weight decay")

    parser.add_argument("--hidden_layers", "--nhl", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--num_neurons", "--nn", type=int, default=128, help="Neurons per hidden layer")
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
    parser.add_argument(
        "--loss",
        "-l",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "mse", "ce"],
        help="Loss function",
    )
    parser.add_argument(
        "--weight_init",
        "--wi",
        type=str,
        default="xavier",
        choices=["xavier", "he", "random", "normal", "zeros"],
        help="Weight initialization",
    )

    parser.add_argument(
        "--optimizer",
        "-o",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        help="Optimizer",
    )

    parser.add_argument("--wandb_project", type=str, default="da6401_assignment_1", help="W&B project name")
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (won't crash if wandb missing, but will skip)",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=os.path.join("models", "best_model.npy"),
        help="Relative path to save best_model.npy",
    )
    parser.add_argument(
        "--config_save_path",
        type=str,
        default=os.path.join("models", "best_config.json"),
        help="Relative path to save best_config.json",
    )

    parser.add_argument("--input_dim", type=int, default=784, help="Input dimension (default 784 for MNIST)")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes (default 10)")

    return parser.parse_args()

def _evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
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

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(
        dataset=args.dataset,
        val_split=float(args.val_split),
        seed=int(args.seed),
        flatten=True,
        normalize=True,
        one_hot=False, 
    )

    model = NeuralNetwork(args)

    wandb = None
    if args.use_wandb:
        run_config = vars(args).copy()
        run_config["hidden_sizes"] = _resolve_hidden_sizes(args)
        wandb = _try_init_wandb(args.wandb_project, run_config)

    best_f1 = -1.0
    best_metrics: Dict[str, Any] = {}

    n_train = X_train.shape[0]

    for epoch in range(1, int(args.epochs) + 1):
        epoch_losses = []
        train_preds_all = []
        train_true_all = []

        for xb, yb in batch_iterator(X_train, y_train, batch_size=int(args.batch_size), shuffle=True, seed=args.seed + epoch):
            logits = model.forward(xb)
            loss, _, _ = model.backward(yb, logits)
            model.update_weights()
            epoch_losses.append(loss)

            preds = model.predict(xb)
            train_preds_all.append(preds)
            train_true_all.append(yb)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_preds = np.concatenate(train_preds_all) if train_preds_all else np.array([], dtype=int)
        train_true = np.concatenate(train_true_all) if train_true_all else np.array([], dtype=int)
        train_metrics = _evaluate_classification(train_true, train_preds) if train_true.size else {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0
        }

        val_preds = model.predict(X_val)
        val_metrics = _evaluate_classification(y_val, val_preds)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            _save_model_and_config(
                model=model,
                args=args,
                model_save_path=args.model_save_path,
                config_save_path=args.config_save_path,
                best_metrics=best_metrics,
            )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_metrics["accuracy"],
                    "train/precision": train_metrics["precision"],
                    "train/recall": train_metrics["recall"],
                    "train/f1": train_metrics["f1"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/f1": val_metrics["f1"],
                    "best/val_f1": best_f1,
                }
            )

    test_preds = model.predict(X_test)
    test_metrics = _evaluate_classification(y_test, test_preds)
    print(
        f"Final Test | acc={test_metrics['accuracy']:.4f} "
        f"prec={test_metrics['precision']:.4f} "
        f"rec={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f}"
    )

    if wandb is not None:
        wandb.log(
            {
                "test/accuracy": test_metrics["accuracy"],
                "test/precision": test_metrics["precision"],
                "test/recall": test_metrics["recall"],
                "test/f1": test_metrics["f1"],
            }
        )
        wandb.finish()

    print(f"Best model saved to: {args.model_save_path}")
    print(f"Best config saved to: {args.config_save_path}")

if __name__ == "__main__":
    main()