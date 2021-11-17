import matplotlib.pyplot as plt
import numpy as np


def plot1d_order2(t, x, xp, P):
    colors = ['r', 'b']
    fig, ax = plt.subplots(2)
    for dim in range(0, 2):
        ax[dim].plot(t, x[0][dim, :], colors[dim])
        ax[dim].plot(t, xp[0][dim, :], colors[dim] + '--')
        s = P[0][:, dim, dim]
        s = 3 * np.sqrt(np.array(s))
        ax[dim].fill_between(t, (xp[0][dim, :] - s).T, (xp[0][dim, :] + s).T, color=colors[dim], alpha=.6)
    plt.show()


def plot2d_order2(t, x, xp, P):
    colors = ['r', 'b', 'g', 'y']
    fig, ax = plt.subplots(4)
    for dim in range(0, 4):
        ax[dim].plot(t, x[0][dim, :], colors[dim])
        ax[dim].plot(t, xp[0][dim, :], colors[dim] + '--')
        s = P[0][:, dim, dim]
        s = 3 * np.sqrt(np.array(s))
        ax[dim].fill_between(t, (xp[0][dim, :] - s).T, (xp[0][dim, :] + s).T, color=colors[dim], alpha=.6)
    plt.show()

