import itertools
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

def chains(T, max_dilation, data_name, use_case, ground_truth_chain, offset, non_overlapping, target_w, m, offset_value=None):
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

        if offset and offset_value is not None:
            offset_start = offset_value
            chain = stumpy.atsc(mp[:, 2], mp[:, 3], offset_start)
            all_chain_set = [chain]
        elif offset and d != 1:
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

        if offset and d==1 and offset_value is None:
            offset_start = chain[0]

def chains_ensemble_mp_min(T, max_dilation, data_name, use_case, ground_truth_chain, offset, non_overlapping, target_w, m, offset_value=None):
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

    mps = []
    for d in range(1, max_dilation+1):
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        mps.append(mp)

    # ensemble mp with min at each offset
    mps = _adjust_mps_to_same_length(mps)
    mp = _min_mp(mps)

    print(f"Calculated min MP")
    if offset and offset_value is not None:
        offset_start = offset_value
        chain = stumpy.atsc(mp[:, 2], mp[:, 3], offset_start)
        all_chain_set = [chain]
    elif offset and d != 1:
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

    return chain_score

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
        f1 = 0
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
            f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, CAC values: {[cac[cp] for cp in found_cps]}, Score: {score} for d={d}, m={m}, w={actual_w}")
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

def segmentation_fluss_known_cps_ensemble_min_BOTHWINDOWSETTINGS(T, T_name, cps, ds, L, n_regimes, target_w, m):
    # fixed m
    cacs = []
    for d in ds:
        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        cac, _ = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes)
        cacs.append(cac)

    # fixed target window
    for d in ds:
        m = round((target_w-1)/d) + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        cac, _ = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes)
        cacs.append(cac)
    
    cacs = _adjust_cacs_to_same_length(cacs)
    min_cac = _find_min_values(cacs)
    found_cps = _rea(min_cac, n_regimes, L)

    score = covering({0: cps}, found_cps, T.shape[0])
    print(f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, CAC values: {[cac[cp] for cp in found_cps]}, Score: {score}")
    return score

def segmentation_fluss_known_cps_ensemble_min(T, T_name, cps, ds, L, n_regimes, target_w, m):
    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    cacs = []
    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        cac, _ = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes)
        cacs.append(cac)
    
    cacs = _adjust_cacs_to_same_length(cacs)
    min_cac = _find_min_values(cacs)
    found_cps = _rea(min_cac, n_regimes, L)

    score = covering({0: cps}, found_cps, T.shape[0])
    print(f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, CAC values: {[cac[cp] for cp in found_cps]}, Score: {score}")
    return score


def segmentation_fluss_unknown_cps_ensemble_min(T, T_name, cps, ds, L, threshold, target_w, m):
    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    cacs = []
    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        cac, _ = stumpy.fluss(mp[:, 1], L=L, n_regimes=1)
        cacs.append(cac)
    
    cacs = _adjust_cacs_to_same_length(cacs)
    min_cac = _find_min_values(cacs)
    found_cps = _rea_unknown_cps(min_cac, L, threshold)

    score = covering({0: cps}, found_cps, T.shape[0])
    print(f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, CAC values: {[cac[cp] for cp in found_cps]}, Score: {score}")
    return score

def segmentation_fluss_known_cps_ensemble_sum(T, T_name, cps, ds, L, n_regimes, target_w, m):
    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    cacs = []
    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        cac, _ = stumpy.fluss(mp[:, 1], L=L, n_regimes=n_regimes)
        cacs.append(cac)
    
    cacs = _adjust_cacs_to_same_length(cacs)
    sum_cac = np.array([sum(x) for x in zip(*cacs)]).astype(np.float64)
    found_cps = _rea(sum_cac, n_regimes, L)

    score = covering({0: cps}, found_cps, T.shape[0])
    print(f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, CAC values: {[cac[cp] for cp in found_cps]}, Score: {score}")
    return score

def segmentation_fluss_unknown_cps_ensemble_sum(T, T_name, cps, ds, L, threshold, target_w, m):
    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    cacs = []
    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        cac, _ = stumpy.fluss(mp[:, 1], L=L, n_regimes=1)
        cacs.append(cac)
    
    cacs = _adjust_cacs_to_same_length(cacs)
    sum_cac = np.array([sum(x) for x in zip(*cacs)]).astype(np.float64)
    found_cps = _rea_unknown_cps(sum_cac, L, threshold)

    score = covering({0: cps}, found_cps, T.shape[0])
    print(f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, CAC values: {[cac[cp] for cp in found_cps]}, Score: {score}")
    return score


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

# from stumpy:
def _rea(cac, n_regimes, L, excl_factor=5):
    regime_locs = np.empty(n_regimes - 1, dtype=np.int64)
    tmp_cac = copy.deepcopy(cac)
    for i in range(n_regimes - 1):
        regime_locs[i] = np.argmin(tmp_cac)
        excl_start = max(regime_locs[i] - excl_factor * L, 0)
        excl_stop = min(regime_locs[i] + excl_factor * L, cac.shape[0])
        tmp_cac[excl_start:excl_stop] = 1.0

    return regime_locs

def _adjust_cacs_to_same_length(cacs):
    max_length = max(len(cac) for cac in cacs)
    adjusted_cacs = []
    for mp in cacs:
        len_diff = max_length - len(mp)
        if len_diff != 0:
            mp = np.append(mp, [1.0]* len_diff) 
        adjusted_cacs.append(mp)
    return adjusted_cacs

def _find_min_values(lists) -> np.ndarray:
    min_values = []
    list_length = len(lists[0])

    for i in range(list_length):
        values_at_index = [lst[i] for lst in lists]
        min_value = min(values_at_index)
        min_values.append(min_value)

    return np.array(min_values).astype(np.float64)

def _min_mp(mps):
    min_mp = []
    for i in range(len(mps[0])):
        distances = [mp[i,0] for mp in mps]
        index_min = min(range(len(distances)), key=distances.__getitem__)
        min_mp.append(mps[index_min][i])
    return np.array(min_mp).astype(np.float64)

def _adjust_mps_to_same_length(mps):
    max_length = max(len(mp) for mp in mps)
    adjusted_mps = []
    for mp in mps:
        len_diff = max_length - len(mp)
        if len_diff == 0:
            adjusted_mps.append(mp)
            continue
        else:
            filler = np.array([np.array([np.float64(-9999),np.uint64(-1),np.uint64(-1),np.uint64(-1)], dtype=object) for _ in range(len_diff)])
            adjusted_mp = np.concatenate((mp, filler), axis=0)
            adjusted_mps.append(adjusted_mp)
    return adjusted_mps