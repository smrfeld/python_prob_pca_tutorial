import numpy as np
import matplotlib.pyplot as plt

from typing import List

def get_scatter_range_2d(data : np.array) -> List[List[float]]:
    x_min = np.min(data[:,0])
    x_max = np.max(data[:,0])
    y_min = np.min(data[:,1])
    y_max = np.max(data[:,1])

    x_width = x_max - x_min
    y_width = y_max - y_min

    if x_width > y_width:
        y_mean = np.mean(data[:,1])
        y_min = y_mean - 0.5 * x_width
        y_max = y_mean + 0.5 * x_width
    else:
        x_mean = np.mean(data[:,0])
        x_min = x_mean - 0.5 * y_width
        x_max = x_mean + 0.5 * y_width

    return [[x_min,x_max],[y_min,y_max]]

def plot_hist_1d(data : np.array, title : str):
    fig = plt.figure()
    plt.hist(data, color='g', histtype='step', lw=2)
    plt.title(title)
    
    return fig

def plot_hist_2d(data : np.array, title : str):
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    axs[0].hist(data[:,0], color='g', histtype='step', lw=2)
    axs[1].hist(data[:,1], color='g', histtype='step', lw=2)
    axs[0].title.set_text(title + " dim 1")
    axs[1].title.set_text(title + " dim 2")

    return fig

def plot_data_visible_and_activated_2d(
    data : np.array, 
    actVisible : np.array
    ):
    # dim_vis = data.shape[1]

    fig, axs = plt.subplots(1,2,figsize=(12,5))
    for i in range(0, 2):
        axs[i].hist(data[:,i], color='r', histtype='step', lw=2)
        axs[i].hist(actVisible[:,i], color='b', histtype='step', lw=2)
        axs[i].title.set_text("Visibles dim %d" % i)

    return fig

def plot_scatter_visible_and_activated_2d(
    data : np.array, 
    act_visible : np.array):

    fig = plt.figure()
    plt.scatter(data[:,0],data[:,1], color='r')
    plt.scatter(act_visible[:,0],act_visible[:,1], color='b')

    lims = get_scatter_range_2d(data)
    plt.xlim(lims[0])
    plt.ylim(lims[1])

    plt.xlabel("Visible dim 1")
    plt.ylabel("Visible dim 2")

    return fig