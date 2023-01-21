import math

import numpy as np
from matplotlib import pyplot as plt

all_human_aa = "AGILPVFWYDERHKSTCMNQ".split()


def plot_4d_array(cubes: np.ndarray, title: str = "4d array", s=100, marker='o', alpha=0.1):
    zone_num = cubes.shape[0]
    # calculate the number of zones
    zone_size = math.ceil(math.sqrt(zone_num))
    # calculate the size of each zone
    fig = plt.figure()
    plt.set_cmap('seismic')
    fig.suptitle(title, fontsize=16)
    for i in range(zone_num):
        x = np.indices(cubes[i].shape)[0]
        y = np.indices(cubes[i].shape)[1]
        z = np.indices(cubes[i].shape)[2]
        col = cubes[i].flatten()

        # 3D Plot
        ax3D = fig.add_subplot(zone_size, zone_size, i + 1, projection='3d')
        ax3D.scatter(x.flatten(), y.flatten(), z.flatten(), c=col, s=s, marker=marker, alpha=alpha)
    plt.show()


def plot_3d_array(cube: np.ndarray, title: str = "3d array", s=100, marker='o', alpha=0.1):
    x = np.indices(cube.shape)[0]
    y = np.indices(cube.shape)[1]
    z = np.indices(cube.shape)[2]
    col = cube.flatten()

    # 3D Plot
    fig = plt.figure()
    # set title
    fig.suptitle(title, fontsize=16)
    plt.set_cmap('seismic')
    ax3D = fig.add_subplot(projection='3d')
    # if dot value is 0, then the dot is not shown, and sensitive to very small values.
    ax3D.scatter(x.flatten(), y.flatten(), z.flatten(), c=col, s=s, marker=marker, alpha=alpha)
    plt.show()


def plot_2d_array(matrix: np.ndarray, title: str = "2d array", s=100, marker='o', alpha=0.1):
    x = np.indices(matrix.shape)[0]
    y = np.indices(matrix.shape)[1]
    col = matrix.flatten() * 100

    # 2D Plot
    fig = plt.figure()
    # set title
    fig.suptitle(title, fontsize=16)
    # set color map to viridis
    plt.set_cmap('seismic')
    ax2D = fig.add_subplot()
    ax2D.scatter(x.flatten(), y.flatten(), c=col, s=s, marker=marker, alpha=alpha)
    # marker is the shape of the dot
    # alpha is the transparency of the dot
    # s is the size of the dot
    # c is the color of the dot. Here, we use the value of the dot as the color.
    plt.show()


def plot_bar(Xdata, Ydata, title, xlabel, ylabel, xticks=None, yticks=None, xticklabels=None, yticklabels=None):
    fig = plt.figure()
    plt.bar(Xdata, Ydata)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    # test 4d array
    # cubes__ = np.random.rand(8, 8, 8, 8)
    # plot_4d_array(cubes__, "4d array", alpha=0.05)

    plot_bar([1,2],[1,2])