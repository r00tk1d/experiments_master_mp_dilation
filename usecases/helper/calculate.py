import stumpy
from . import results
from . import utils
from tssb.evaluation import covering
import numpy as np
import copy


def chains(T, ds, target_w, data_name, use_case):
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

        unanchored_chain_score = _chain_score(unanchored_chain, T, d, m)
        non_overlapping_unanchored_chain_score = _chain_score(non_overlapping_unanchored_chain, T, d, m)

        results.save([T, m, d, mp, all_chain_set, all_non_overlapping_chain_set, unanchored_chain, non_overlapping_unanchored_chain, unanchored_chain_score, non_overlapping_unanchored_chain_score], file_path + ".npy")

def _chain_score(chain, T, d, m):
    T_norm = (T - np.mean(T)) / np.std(T)

    # obtain subseqeunces
    subsequences = []
    for start_idx in chain:
        stop_idx = start_idx + (m-1)*d + 1
        subsequence = T_norm[start_idx:stop_idx:d]
        subsequences.append(subsequence)
    

    # length (number of nodes)
    chain_length = len(chain)


    # effective length (the greater the better) (considers divergence and graduality)
    distances = []
    for i in range(len(subsequences)-1):
        distance = np.linalg.norm(subsequence[i]-subsequence[i+1])
        distances.append(distance)
    max_distance_between_nodes = max(distances)
    # distances = np.linalg.norm(subsequences[:-1] - subsequences[1:], axis=1)

    distance_first_last_node = np.linalg.norm(subsequence[0]-subsequence[-1])
    effective_length = round(distance_first_last_node / max_distance_between_nodes)

    # correlation length (the greater the better) (considers similarity of consecutive subsequences)
    corr_lengths = []
    for i in range(len(subsequences)-1):
        corr = np.corrcoef(subsequences[i], subsequences[i+1])[0,1]
        corr_lengths.append(abs(corr) * corr)
    correlation_length = sum(corr_lengths)

    return {"Length": chain_length,
            "Effective Length": effective_length,
            "Correlation Length": correlation_length,
            }


def segmentation_fluss_known_cps(T, T_name, cps, ds, target_w, L, n_regimes):
    scores = []
    for d in ds:
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


def segmentation_fluss_unknown_cps(T, T_name, cps, ds, target_w, L, threshold):
    scores = []
    for d in ds:
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