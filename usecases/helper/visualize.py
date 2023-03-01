import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def discord(T, m, d, mp, discord_idx):
    w = ((m-1)*d + 1)
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Top Discord (m = ' + str(m) + ', d = ' + str(d) + ', w = ' + str(w) + ')', fontsize='15')

    axs[0].plot(T)
    axs[0].set_ylabel('Sample TS', fontsize='10')
    rect = Rectangle((discord_idx, 0), w, np.amax(T)-np.amin(T), facecolor='lightgrey')
    axs[0].add_patch(rect)
    axs[1].set_xlabel('Time', fontsize ='10')
    axs[1].set_ylabel('Matrix Profile', fontsize='10')
    axs[1].axvline(x=discord_idx, linestyle="dashed")
    axs[1].plot(mp[:, 0])
    return plt

def motif_pair(T, m, d, mp, top_motif_pair_idxs):
    w = ((m-1)*d + 1)
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    plt.suptitle('Top Motif Pair (m = ' + str(m) + ', d = ' + str(d) + ', w = ' + str(w) + ')', fontsize='15')

    axs[0].plot(T)
    axs[0].set_ylabel('Time Series', fontsize='10')
    rect = Rectangle((top_motif_pair_idxs[0], 0), w, np.amax(T), facecolor='lightgrey')
    axs[0].add_patch(rect)
    rect = Rectangle((top_motif_pair_idxs[1], 0), w, np.amax(T), facecolor='lightgrey')
    axs[0].add_patch(rect)
    axs[1].set_xlabel('Time', fontsize ='10')
    axs[1].set_ylabel('Matrix Profile', fontsize='10')
    axs[1].axvline(x=top_motif_pair_idxs[0], linestyle="dashed")
    axs[1].axvline(x=top_motif_pair_idxs[1], linestyle="dashed")
    axs[1].plot(mp[:, 0])
    return plt

def chain_unanchored(T, unanchored_chain, m, d):
    w = ((m-1)*d + 1)
    T = pd.DataFrame(T)
    plt.figure(figsize=(12, 4))
    plt.plot(T, linewidth=1, color='black')
    for i in range(unanchored_chain.shape[0]):
        y = T.iloc[unanchored_chain[i]:unanchored_chain[i]+w]
        x = y.index.values
        plt.plot(x, y, linewidth=3)
    plt.suptitle('Unanchored Chain (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    return plt

def chain_unanchored_snippets(T, unanchored_chain, m, d):
    w = ((m-1)*d + 1)
    T = pd.DataFrame(T)
    plt.figure(figsize=(12, 4))
    plt.axis('off')
    for i in range(unanchored_chain.shape[0]):
        data = T.iloc[unanchored_chain[i]:unanchored_chain[i]+w].reset_index().values
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x-x.min()+(w+15)*i, y-y.min(), linewidth=3)
    plt.suptitle('Unanchored Chain (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    return plt

def segmentation_regimecac(T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations):
    plt.figure(figsize=(16, 4))
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
    axs[0].plot(range(T.shape[0]), T)
    axs[1].plot(range(cac.shape[0]), cac, color='C1')
    for regime_location in regime_locations:
        axs[0].axvline(x=regime_location, linestyle="dashed")
        axs[1].axvline(x=regime_location, linestyle="dashed")
    plt.suptitle('Regimes (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    axs[1].set_xlabel('Time', fontsize ='10')
    axs[0].set_ylabel('Time Series', fontsize='10')
    axs[1].set_ylabel('Arc Curve', fontsize='10')
    return plt

# def segmentation_regimecac_snippets(T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations):
#     w = ((m-1)*d + 1)
#     T = pd.DataFrame(T)
#     start = 25000 - 2500
#     stop = 25000 + 2500
#     plt.figure(figsize=(12, 4))
#     plt.axis('off')
#     for i in range(unanchored_chain.shape[0]):
#         data = T.iloc[unanchored_chain[i]:unanchored_chain[i]+w].reset_index().values
#         x = data[:, 0]
#         y = data[:, 1]
#         plt.plot(x-x.min()+(w+15)*i, y-y.min(), linewidth=3)
#     plt.suptitle('Regimes (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
#     return plt