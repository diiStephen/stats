import numpy as np
import matplotlib.pyplot as plt


def plot_ecdf_simple(data):
    plt.plot(np.sort(data), np.linspace(0, 1, len(data), endpoint=False))


def ecdf(x, data):
    return sum(map(lambda r: r <= x, data)) / len(data)


def epsilon_n(n, alpha=0.05):
    return np.sqrt(1 / (2 * n) * np.log(2 / alpha))


def l(x, data):
    eps_n = epsilon_n(len(data))
    return np.max([ecdf(x, data) - eps_n, 0])


def u(x, data):
    eps_n = epsilon_n(len(data))
    return np.min([ecdf(x, data) + eps_n, 1])


def plot_ecdf_with_ci(data, dist, alpha=.05):
    data_sorted = np.sort(data)

    L = [l(x, data) for x in data_sorted]
    U = [u(x, data) for x in data_sorted]
    ecdf_imgs = [ecdf(x, data) for x in data_sorted]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(data_sorted, ecdf_imgs, label="ecdf")
    ax.plot(data_sorted, [dist.cdf(x) for x in data_sorted], color='lightgreen', label="cdf")
    ax.fill_between(data_sorted, L, U, color='b', alpha=.1, label="95% confidence")
    ax.legend(loc=4, prop={'size': 15})
    plt.grid()
