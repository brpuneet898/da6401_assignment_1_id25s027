import keras

def load_dataset(dataset):
    # this is the customizable function that will act as a dataloader. 
    ## we have two main arguments, one is mnist and fashion mnist dataset. 
    ## both of them correspond to classification task and that too multi-class classification problem.
    ds = None
    if dataset == "mnist":
        ds = keras.datasets.mnist.load_data(path="mnist.npz")
    elif dataset == "fashion_mnist":
        ds = keras.datasets.fashion_mnist.load_data()
    
    (train_images, train_labels), (test_images, test_labels)  = ds

    assert train_images.shape == (60000, 28, 28)
    assert test_images.shape == (10000, 28, 28)
    assert train_labels.shape == (60000,)
    assert test_labels.shape == (10000,)

    return (train_images, train_labels), (test_images, test_labels)