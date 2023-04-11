from typing import NamedTuple
import numpy as np
import os

class Result(NamedTuple):
    T: np.ndarray
    m: int
    d: int
    mp: np.ndarray
    all_chain_set: np.ndarray
    all_non_overlapping_chain_set: np.ndarray
    unanchored_chain: np.ndarray
    non_overlapping_unanchored_chain: np.ndarray
    unanchored_chain_score: dict
    non_overlapping_unanchored_chain_score: dict
    ground_truth_chain: list
    ground_truth_non_overlapping_chain: list
    offset_start: int

def save(np_arrays: list, file_path):
    result = np.array(np_arrays, dtype=object)
    # if os.path.exists(file_path):
    #     print('\033[38;5;208m File: ', file_path, ' already exists, but will be overwritten. \033[0m')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, result, allow_pickle=True)
        

def load(file_path) -> Result:
    result = np.load(file_path, allow_pickle=True)
    return Result(*tuple(result))