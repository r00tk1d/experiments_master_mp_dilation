import numpy as np
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