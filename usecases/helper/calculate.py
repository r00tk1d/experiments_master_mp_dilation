import stumpy
from . import results
from . import utils
from tssb.evaluation import covering
import numpy as np
import copy
import math

def chains(T, ds, target_w, data_name, use_case, ground_truth = None):
    for d in ds:
        m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1
        file_name = data_name + "_d" + str(d) + "_m" + str(m)
        file_path = "../results/" + use_case + "/" + data_name + "/" + "target_w" + str(target_w) + "/" + file_name

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        print("Calculated MP for: w=" + str(actual_w) + ", m=" + str(m) + ", d=" + str(d))
        all_chain_set, unanchored_chain = stumpy.allc(mp[:, 2], mp[:, 3])
        all_non_overlapping_chain_set, non_overlapping_unanchored_chain = utils.remove_overlapping_chains(all_chain_set, m, d)

        length_unanchored_chain = unanchored_chain[-1] - unanchored_chain[0]
        length_non_overlapping_unanchored_chain = non_overlapping_unanchored_chain[-1] - non_overlapping_unanchored_chain[0]

        unanchored_chain_score = _chain_score(unanchored_chain, T, d, m, ground_truth)
        non_overlapping_unanchored_chain_score = _chain_score(non_overlapping_unanchored_chain, T, d, m, ground_truth)

        results.save([T, m, d, mp, all_chain_set, all_non_overlapping_chain_set, unanchored_chain, non_overlapping_unanchored_chain, unanchored_chain_score, non_overlapping_unanchored_chain_score, ground_truth], file_path + ".npy")

def _chain_score(chain, T, d, m, ground_truth = None):
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

    # Recall and Precision (a hit is if overlap is > 50%)
    recall = None
    precision = None
    f1 = None
    if ground_truth:
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

    return {"Length": chain_length,
            "Effective Length": effective_length,
            "Correlation Length": correlation_length,
            "Recall": recall,
            "Precision": precision,
            "F1-Score": f1
            }


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