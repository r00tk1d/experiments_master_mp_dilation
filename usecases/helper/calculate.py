from typing import NamedTuple
import stumpy
from . import results
from . import utils
from tssb.evaluation import covering
import numpy as np
import copy
import math

class ChainScore(NamedTuple):
    length: int
    effective_length: int
    correlation_length: float
    recall: float
    precision: float
    f1_score: float

def chains(T, max_dilation, data_name, use_case, ground_truth_chain, offset, non_overlapping, target_w, m):
    """
    Calculates the chains for a given time series {T} and a given list of dilations {ds}.
    The chains are calculated for a given target window range {target_w}.
    If {offset} is set to true, the chains with a dilation size above 1 are calculated with an offset determined by the starting point of the unanchored chain without dilation. 
    If {ground_truth} is None, ground_truth set to the unanchored chain without dilation.
    """
    assert (target_w is None) != (m is None), "only one of target_w and m can be set"
    assert not (offset and non_overlapping), "offset and non_overlapping cannot be true at the same time"	
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    offset_start = 0
    ground_truth_given = ground_truth_chain is not None

    print(f"Running Experiment: target_w={target_w}, m={m}, offset={offset!s}, groundtruthD1={not ground_truth_given!s}, nonoverlapping={non_overlapping!s}")

    for d in range(1, max_dilation+1):
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1
        file_path, _ = utils.build_file_path(use_case, data_name, d, actual_w, target_w, m, offset, ground_truth_given, calculate_m, non_overlapping)

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        print(f"Calculated MP for: w={actual_w}, m={m}, d={d}, offset={offset_start}, groundtruthD1={not ground_truth_given!s}, nonoverlapping={non_overlapping!s}")

        if offset and d != 1:
            chain = stumpy.atsc(mp[:, 2], mp[:, 3], offset_start)
            all_chain_set = [chain]
        else:
            all_chain_set, chain = stumpy.allc(mp[:, 2], mp[:, 3])

        if non_overlapping:
            all_chain_set, chain = utils.remove_overlapping_chains(all_chain_set, m, d)
            chain = np.array([x + offset_start for x in chain])

        if not ground_truth_chain and d==1:
            ground_truth_chain = list(chain)

        chain_score = _get_chain_score(chain, T, d, m, ground_truth_chain)

        results.save([T, m, d, mp, all_chain_set, chain, chain_score, ground_truth_chain, offset_start], file_path + ".npy")

        if offset and d==1:
            offset_start = chain[0]

def _get_chain_score(chain, T, d, m, ground_truth) -> ChainScore:
    if len(chain) == 0:
        return ChainScore(*(0, 0, 0, 0, 0, -1))
    elif len(chain) == 1:
        return ChainScore(*(1, 0, 0, 0, 0, -1))
    
    T_norm = (T - np.mean(T)) / np.std(T)

    # obtain chain subsequences
    w = (m-1)*d + 1
    chain_subsequences = []
    for start_idx in chain:
        stop_idx = start_idx + w
        subsequence = T_norm[start_idx:stop_idx:d]
        chain_subsequences.append(subsequence)

    # length (number of nodes)
    chain_length = len(chain)

    # effective length (the greater the better) (considers divergence and graduality)
    distances = []
    for i in range(len(chain_subsequences)-1):
        distance = np.linalg.norm(chain_subsequences[i]-chain_subsequences[i+1])
        distances.append(distance)
    max_distance_between_nodes = max(distances)

    distance_first_last_node = np.linalg.norm(chain_subsequences[0]-chain_subsequences[-1])
    effective_length = round(distance_first_last_node / max_distance_between_nodes)

    # correlation length (the greater the better) (considers similarity of consecutive subsequences)
    corr_lengths = []
    for i in range(len(chain_subsequences)-1):
        corr = np.corrcoef(chain_subsequences[i], chain_subsequences[i+1])[0,1]
        corr_lengths.append(abs(corr) * corr)
    correlation_length = sum(corr_lengths)

    # Recall, Precision, F1 Score (a hit is if overlap is > 50%)
    recall = None
    precision = None
    f1 = None
    n_hits = 0
    for start_idx in chain:
        closest_ground_truth = min(ground_truth, key=lambda x:abs(x-start_idx))
        if start_idx == closest_ground_truth:
            n_hits += 1
        elif start_idx < closest_ground_truth and start_idx + math.ceil(w/2) > closest_ground_truth:
            n_hits += 1
        elif start_idx > closest_ground_truth and start_idx - math.ceil(w/2) < closest_ground_truth:
            n_hits += 1
    recall = n_hits / len(ground_truth)
    precision = n_hits / len(chain)
    if precision+recall == 0:
        f1 = -1
    else:
        f1 = (2*precision*recall)/(precision+recall)
    return ChainScore(*(chain_length, effective_length, correlation_length, recall, precision, f1))


def segmentation_fluss_known_cps(T, T_name, cps, ds, L, n_regimes, target_w, m):
    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    scores = []
    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        cac, found_cps = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes)
        score = covering({0: cps}, found_cps, T.shape[0])
        print(
            f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, Score: {score} for d={d}, m={m}, w={actual_w}")
        scores.append(score)
    return scores


def segmentation_fluss_unknown_cps(T, T_name, cps, ds, L, threshold, target_w, m):
    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    scores = []
    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        # dont use the _rea function with n_regimes inside fluss
        cac, _ = stumpy.fluss(mp[:, 1], L=L, n_regimes=1)
        found_cps = _rea_unknown_cps(cac, L, threshold)
        score = covering({0: cps}, found_cps, T.shape[0])
        print(
            f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, Score: {score} for d={d}, m={m}, w={actual_w}")
        scores.append(score)
    return scores

# uses a threshold to determine the cps
def _rea_unknown_cps(cac, L, threshold, excl_factor=5):
    found_cps = []
    tmp_cac = copy.deepcopy(cac)
    current_min_idx = np.argmin(tmp_cac)
    while tmp_cac[current_min_idx] <= threshold:
        found_cps.append(current_min_idx)
        excl_start = max(current_min_idx - excl_factor * L, 0)
        excl_stop = min(current_min_idx + excl_factor * L, cac.shape[0])
        tmp_cac[excl_start:excl_stop] = 1.0
        current_min_idx = np.argmin(tmp_cac)      
    found_cps.sort()
    return np.asarray(found_cps)