import numpy as np


def sample_skewness(data):
    n = len(data)
    x_bar = np.mean(data)
    std_hat = np.std(data)
    skewness = ((1 / n) * np.sum((data - x_bar) ** 3)) / std_hat ** 3
    return skewness
