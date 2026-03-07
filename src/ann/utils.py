import numpy as np

def get_dead_neurons(a: np.ndarray):
    alive_mask = np.any(a > 0, axis = 0)
    dead_indices = np.where(~alive_mask)[0]
    death_rate = len(dead_indices) / a.shape[1]
    return dead_indices, death_rate