import numpy as np

from . import results

def find_common_strings(list_of_lists: list[list[str]]):
    if not list_of_lists:
        return []

    common_strings = set(list_of_lists[0])
    for sublist in list_of_lists[1:]:
        common_strings = common_strings.intersection(sublist)

    return list(common_strings)


def build_file_path(use_case, data_name, d, actual_w, target_w, m, offset, ground_truth_given, calculate_m, non_overlapping):
    file_name = f"{data_name}_d{d}_" + (f"m{m}" if calculate_m else f"w{actual_w}")
    folder_name = (f"targetw{target_w}" if calculate_m else f"m{m}") + (f"_offsetTRUE" if offset else f"_offsetFALSE") + (f"_groundtruthGIVEN" if ground_truth_given else f"_groundtruthD1") + (f"_nonoverlappingTRUE" if non_overlapping else f"_nonoverlappingFALSE")
    file_path = f"../results/{use_case}/{data_name}/{folder_name}/{file_name}"
    folder_path = f"../results/{use_case}/{data_name}/{folder_name}"
    return file_path, folder_path

def calculate_max_d_from_target_w(target_w: int) -> int:
    d = 1
    m = round((target_w-1)/d) + 1
    while d < m:
        d += 1
        m = round((target_w-1)/d) + 1
    return d-1

def calculate_max_d_from_m(m: int, time_series_length: int, max_d: int) -> int:
    d = 1
    w = (m-1)*d + 1
    while w < time_series_length:
        d += 1
        w = (m-1)*d + 1
    return min(max_d, d-1)

# def get_min_max_from_lists(list1, list2) -> tuple:
#     min_value = min(min(list1), min(list2))
#     max_value = max(max(list1), max(list2))
#     return (min_value, max_value)

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

def get_metrics_for_experiment(max_dilation, data_name, use_case, offset, non_overlapping, target_w, m, ground_truth_chain):
    recall = []
    precision = []
    f1_scores = []
    correlation_lengths = []

    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    ground_truth_given = ground_truth_chain is not None

    
    ds = [d for d in range(1, max_dilation+1)]
    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1
        file_path, _ = build_file_path(use_case, data_name, d, actual_w, target_w, m, offset, ground_truth_given, calculate_m, non_overlapping)
        result = results.load(file_path + ".npy")
        recall.append(result.chain_score.recall)
        precision.append(result.chain_score.precision)
        f1_scores.append(result.chain_score.f1_score)
        correlation_lengths.append(result.chain_score.correlation_length)
    return recall, precision, f1_scores, correlation_lengths


def _check_non_overlap(seq_one, seq_two, w):
    return seq_one + w < seq_two

def _set_best_chain(current_longest_chain, new_chain):
    if len(new_chain) > len(current_longest_chain):
        return new_chain
    elif len(new_chain) == len(current_longest_chain) and new_chain[0] < current_longest_chain[0]:
        return new_chain
    return current_longest_chain