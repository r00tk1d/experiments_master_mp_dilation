import numpy as np
import os


def calculate_max_dilation_size(target_w):
    """ Calculates the maximum dilation size to get a target range w 
    with the constraint that each dilation size d needs to be 
    smaller than the number of values m

    Parameters
    ----------
    target_w: int
        target range for the subsequence

    Returns
    -------
    dilation_sizes: list
        list of possible dilation sizes
    """
    d = 1
    m = round((target_w-1)/d) + 1
    while d < m:
        d += 1
        m = round((target_w-1)/d) + 1
    return d-1

def get_min_max_from_lists(list1, list2) -> tuple:
    min_value = min(min(list1), min(list2))
    max_value = max(max(list1), max(list2))
    return (min_value, max_value)

def remove_overlapping_chains(all_chain_set, m, d):
    w = (m-1)*d + 1
    all_non_overlapping_chain_set = []
    non_overlapping_unanchored_chain = []
    for i in range(len(all_chain_set)):
        non_overlap = True
        for j in range(1, len(all_chain_set[i])):
            non_overlap = _check_non_overlap(all_chain_set[i][j-1], all_chain_set[i][j], w)
            if not non_overlap:
                break
        if non_overlap:
            all_non_overlapping_chain_set.append(all_chain_set[i])
            non_overlapping_unanchored_chain = _set_best_chain(non_overlapping_unanchored_chain, all_chain_set[i])
    return all_non_overlapping_chain_set, non_overlapping_unanchored_chain



def _check_non_overlap(seq_one, seq_two, w):
    return seq_one + w < seq_two

def _set_best_chain(current_longest_chain, new_chain):
    if len(new_chain) > len(current_longest_chain):
        return new_chain
    elif len(new_chain) == len(current_longest_chain) and new_chain[0] < current_longest_chain[0]:
        return new_chain
    return current_longest_chain
