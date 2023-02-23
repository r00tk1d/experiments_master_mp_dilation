import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def discord(mp, T, m, d) -> None:
    discord_idx = np.argsort(mp[:, 0])[-1]

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Top Discord (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')

    axs[0].plot(T)
    axs[0].set_ylabel('Sample TS', fontsize='10')
    # rect = Rectangle((discord_idx, 0), m, np.amax(T)-np.amin(T), facecolor='lightgrey')
    # axs[0].add_patch(rect)
    axs[1].set_xlabel('Time', fontsize ='10')
    axs[1].set_ylabel('Matrix Profile', fontsize='10')
    axs[1].axvline(x=discord_idx, linestyle="dashed")
    axs[1].plot(mp[:, 0])
    plt.show()

def motif_pair(mp, T, m, d) -> None:
    motif_idx = np.argsort(mp[:, 0])[0]
    motif_nearest_neighbor_idx = mp[motif_idx, 1]

    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Top Motif Pair (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')

    axs[0].plot(T)
    axs[0].set_ylabel('Time Series', fontsize='10')
    rect = Rectangle((motif_idx, 0), m, np.amax(T), facecolor='lightgrey')
    axs[0].add_patch(rect)
    # rect = Rectangle((motif_nearest_neighbor_idx, 0), m, np.amax(T)-np.amin(T), facecolor='lightgrey')
    # axs[0].add_patch(rect)
    axs[1].set_xlabel('Time', fontsize ='10')
    axs[1].set_ylabel('Matrix Profile', fontsize='10')
    axs[1].axvline(x=motif_idx, linestyle="dashed")
    axs[1].axvline(x=motif_nearest_neighbor_idx, linestyle="dashed")
    axs[1].plot(mp[:, 0])
    plt.show()

def chain_unanchored(T, unanchored_chain, m, d):
    r = ((m-1)*d + 1)
    T = pd.DataFrame(T)
    plt.plot(T, linewidth=1, color='black')
    for i in range(unanchored_chain.shape[0]):
        y = T[unanchored_chain[i]:unanchored_chain[i]+r]
        x = y.index.values
        plt.plot(x, y, linewidth=3)
    plt.suptitle('Unanchored Chain (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    plt.show()

def chain_unanchored_snippets(T, unanchored_chain, m, d):
    r = ((m-1)*d + 1)
    T = pd.DataFrame(T)
    plt.axis('off')
    for i in range(unanchored_chain.shape[0]):
        data = T[unanchored_chain[i]:unanchored_chain[i]+r].reset_index().values
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x-x.min()+(r+5)*i, y-y.min(), linewidth=3)
    plt.suptitle('Unanchored Chain (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    plt.show()