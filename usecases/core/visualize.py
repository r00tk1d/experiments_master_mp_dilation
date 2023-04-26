import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import results
from . import utils

def chains(max_dilation, data_name, use_case, offset, non_overlapping, target_w, m, ground_truth_chain):
    plt.rcParams.update({'figure.max_open_warning': 0})

    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    ground_truth_given = ground_truth_chain is not None

    chain_scores = []
    ds = [d for d in range(1, max_dilation+1)]
    print(f"Visualizing Experiment: target_w={target_w}, m={m}, offset={offset!s}, groundtruthD1={not ground_truth_given!s}, nonoverlapping={non_overlapping!s}")

    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1
        file_path, folder_path = utils.build_file_path(use_case, data_name, d, actual_w, target_w, m, offset, ground_truth_given, calculate_m, non_overlapping)

        result = results.load(file_path + ".npy")
        print(f"chain for d={result.d}, m={result.m}, w={actual_w}: {result.chain}")

        chain_scores.append(result.chain_score)
        
        # detected chain
        title = f"Detected Non Overlapping Chain" if non_overlapping else f"Detected Chain"
        plot = _chain(result.T, result.chain, m, d, title)
        plot.savefig(file_path + "_chain")
        plot = _chain_snippets(result.T, result.chain, m, d, title)
        plot.savefig(file_path + "_chain_snippets")
    
    _chain_elbowplots(ds, chain_scores, target_w, data_name, use_case, offset, non_overlapping, folder_path)

def _chain_elbowplots(ds, chain_scores, target_w, data_name, use_case, offset, non_overlapping, folder_path):
    # elbow plot for ds length
    lengths_chains = [chain_score.length for chain_score in chain_scores]
    y_lim = (0, max(lengths_chains))
    title = f"Length Non Overlapping Chain" if non_overlapping else f"Length Chain"
    plot = _chain_elbowplot(lengths_chains, ds, y_lim, title)
    plot.savefig(folder_path + "/_plot_length")

    # elbow plot for ds effective length
    effective_lengths_chains = [chain_score.effective_length for chain_score in chain_scores]
    y_lim = (0, max(effective_lengths_chains))
    title = f"Effective Length Non Overlapping Chain" if non_overlapping else f"Effective Length Chain"
    plot = _chain_elbowplot(effective_lengths_chains, ds, y_lim, title)
    plot.savefig(folder_path + "/_plot_effective_length")

    # elbow plot for ds correlation length
    correlation_lengths_chains = [chain_score.correlation_length for chain_score in chain_scores]
    y_lim = (0, max(correlation_lengths_chains))
    title = f"Correlation Length Non Overlapping Chain" if non_overlapping else f"Correlation Length Chain"
    plot = _chain_elbowplot(correlation_lengths_chains, ds, y_lim, title)
    plot.savefig(folder_path + "/_plot_correlation_length")

    # elbow plot for ds recall
    recall_chains = [chain_score.recall for chain_score in chain_scores]
    y_lim = (0, max(recall_chains))
    title = f"Recall Non Overlapping Chain" if non_overlapping else f"Recall Chain"
    plot = _chain_elbowplot(recall_chains, ds, y_lim, title)
    plot.savefig(folder_path + "/_plot_recall")

    # elbow plot for ds precision
    precision_chains = [chain_score.precision for chain_score in chain_scores]
    y_lim = (0, max(precision_chains))
    title = f"Precision Non Overlapping Chain" if non_overlapping else f"Precision Chain"
    plot = _chain_elbowplot(precision_chains, ds, y_lim, title)
    plot.savefig(folder_path + "/_plot_precision")

    # elbow plot for ds f1 score
    f1_chains = [chain_score.f1_score for chain_score in chain_scores]
    y_lim = (0, max(f1_chains))
    title = f"F1-Score Non Overlapping Chain" if non_overlapping else f"F1-Score Chain"
    plot = _chain_elbowplot(f1_chains, ds, y_lim, title)
    plot.savefig(folder_path + "/_plot_f1")
    

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

def _chain(T, chain, m, d, title):
    w = ((m-1)*d + 1)
    T = pd.DataFrame(T)
    plt.figure(figsize=(12, 2))
    plt.plot(T, linewidth=1, color='black')
    for i in range(chain.shape[0]):
        y = T.iloc[chain[i]:chain[i]+w]
        x = y.index.values
        plt.plot(x, y, linewidth=3)
    plt.suptitle(title + ' (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    return plt

def _chain_snippets(T, chain, m, d, title):
    w = ((m-1)*d + 1)
    T = pd.DataFrame(T)
    plt.figure(figsize=(12, 2))
    plt.axis('off')
    for i in range(chain.shape[0]):
        data = T.iloc[chain[i]:chain[i]+w].reset_index().values
        x = data[:, 0]
        y = data[:, 1]
        plt.plot(x-x.min()+(w+15)*i, y-y.min(), linewidth=3)
    plt.suptitle(title + ' (m = ' + str(m) + ', d = ' + str(d) + ')', fontsize='15')
    return plt

def _chain_elbowplot(lengths: list, ds: list, y_lim: tuple, ylabel: str):
    plt.figure(figsize=(8,4))
    plt.plot(ds, lengths, 'bx-')
    plt.xlabel('Dilation Size')
    plt.ylabel(ylabel)
    plt.xticks(range(1,ds[-1]+1))
    plt.ylim(y_lim[0] - 0.1 * y_lim[0], y_lim[1] + 0.1 * y_lim[1])
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