import numpy as np

def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y_feature = y.astype(int).reshape(-1)
    outp = np.zeros((y_feature.shape[0], num_classes), dtype=np.float32)
    outp[np.arange(y_feature.shape[0]), y_feature] = 1.0
    return outp

def load_dataset(dataset: str, val_split: float = 0.1, seed: int = 42, flatten: bool = True, normalize: bool = True, one_hot:bool = False,):
    dataset = dataset.lower().strip()
    if dataset == "mnist":
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset in ["fashion_mnist", "fashion-mnist"]:
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"The given dataset: {dataset} is not supported. Please choose either mnist or fashion_mnist.")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    rng = np.random.default_rng(seed)
    idx = np.arange(X_train.shape[0])
    rng.shuffle(idx)

    val_size = int(round(val_split * X_train.shape[0]))
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    if one_hot:
        y_train = _one_hot(y_train, 10)
        y_val = _one_hot(y_val, 10)
        y_test = _one_hot(y_test, 10)
    else:
        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def batch_iterator(X, y, batch_size: int, shuffle: bool = True, seed: int = 42):
        n = X.shape[0]
        indic = np.arange(n)

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indic)

        for start in range(0, n, batch_size):
            batch_idx = indic[start:start + batch_size]
            yield X[batch_idx], y[batch_idx]