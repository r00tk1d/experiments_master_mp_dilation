import numpy as np
import os


def calculate_ds(target_w):
    d = 1
    ds = []
    m = round((target_w-1)/d) + 1
    while d < m:
        ds.append(d)
        d += 1
        m = round((target_w-1)/d) + 1
    return ds

def remove_overlapping_chains(all_chain_set, m, d):
    w = (m-1)*d + 1
    all_non_overlapping_chain_set = []
    non_overlapping_unanchored_chain = []
    for i in range(len(all_chain_set)):
        unique = True
        for j in range(1, len(all_chain_set[i])):
            unique = _check_unique(
                all_chain_set[i][j-1], all_chain_set[i][j], w)
            if not unique:
                break
        if unique:
            all_non_overlapping_chain_set.append(all_chain_set[i])
            if len(all_chain_set[i]) > len(non_overlapping_unanchored_chain):
                non_overlapping_unanchored_chain = all_chain_set[i]
    return all_non_overlapping_chain_set, non_overlapping_unanchored_chain

def _check_unique(seq_one, seq_two, w):
    return seq_one + w < seq_two
