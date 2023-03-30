import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import results
from . import utils

def chains(ds, target_w, data_name, use_case):
    plt.rcParams.update({'figure.max_open_warning': 0})

    lengths_unanchored_chains = []
    lengths_non_overlapping_unanchored_chains = []

    for d in ds:
        m = round((target_w-1)/d) + 1
        file_name = data_name + "_d" + str(d) + "_m" + str(m)
        file_path = "../results/" + use_case + "/" + data_name + "/" + "target_w" + str(target_w) + "/" + file_name

        T, m, d, mp, all_chain_set, all_non_overlapping_chain_set, unanchored_chain, non_overlapping_unanchored_chain, length_unanchored_chain, length_overlapping_unanchored_chain = results.load(file_path + ".npy")

        lengths_unanchored_chains.append(length_unanchored_chain)
        lengths_non_overlapping_unanchored_chains.append(length_overlapping_unanchored_chain)
        
        # unanchored chain
        plot = _chain_unanchored(T, unanchored_chain, m, d, "Unanchored Chain")
        plot.savefig(file_path + "_unanchored")
        plot = _chain_unanchored_snippets(T, unanchored_chain, m, d, "Unanchored Chain")
        plot.savefig(file_path + "_unanchored_snippets")

        # non overlapping unanchored chain
        plot = _chain_unanchored(T, non_overlapping_unanchored_chain, m, d, "Non Overlapping Unanchored Chain")
        plot.savefig(file_path + "_non_overlapping_unanchored")
        plot = _chain_unanchored_snippets(T, non_overlapping_unanchored_chain, m, d, "Non Overlapping Unanchored Chain")
        plot.savefig(file_path + "_non_overlapping_unanchored_snippets")

    # elbow plot for ds
    y_lim = utils.get_min_max_from_lists(lengths_unanchored_chains, lengths_non_overlapping_unanchored_chains)
    plot = _chain_elbowplot(lengths_unanchored_chains, ds, y_lim, "Elbow Plot Unanchored Chains")
    plot.savefig("../results/" + use_case + "/" + data_name + "/" + "target_w" + str(target_w) + "/" + data_name + "_elbowplot")
    plot = _chain_elbowplot(lengths_non_overlapping_unanchored_chains, ds, y_lim, "Elbow Plot Non Overlapping Unanchored Chains")
    plot.savefig("../results/" + use_case + "/" + data_name + "/" + "target_w" + str(target_w) + "/" + data_name + "_non_overlapping_elbowplot")

def _discord(T, m, d, mp, discord_idx):
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

def _motif_pair(T, m, d, mp, top_motif_pair_idxs):
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

def _chain_unanchored(T, unanchored_chain, m, d, title):
    w = ((m-1)*d + 1)
    T = pd.DataFrame(T)
    plt.figure(figsize=(12, 2))
    plt.plot(T, linewidth=1, color='black')
    for i in range(unanchored_chain.shape[0]):
        y = T.iloc[unanchored_chain[i]:unanchored_chain[i]+w]
        x = y.index.values
        plt.plot(x, y, linewidth=3)
    plt.suptitle(title + ' (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    return plt

def _chain_unanchored_snippets(T, unanchored_chain, m, d, title):
    w = ((m-1)*d + 1)
    T = pd.DataFrame(T)
    plt.figure(figsize=(12, 2))
    plt.axis('off')
    for i in range(unanchored_chain.shape[0]):
        data = T.iloc[unanchored_chain[i]:unanchored_chain[i]+w].reset_index().values
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x-x.min()+(w+15)*i, y-y.min(), linewidth=3)
    plt.suptitle(title + ' (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    return plt

def _chain_elbowplot(max_distances: list, ds: list, y_lim: tuple, title: str):
    plt.figure(figsize=(8,4))
    plt.plot(ds, max_distances, 'bx-')
    plt.xlabel('Dilation Size')
    plt.ylabel('Length Unanchored Chain')
    plt.title(title)
    plt.xticks(range(1,ds[-1]+1))
    plt.ylim(y_lim[0] - 50, y_lim[1] + 50)
    return plt

def _segmentation_regimecac(T, m, d, L, n_regimes, excl_factor, mp, cac, regime_locations):
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