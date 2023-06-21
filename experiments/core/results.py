from typing import NamedTuple
import numpy as np
import os
import pandas as pd
from . import calculate

class Result(NamedTuple):
    T: np.ndarray
    m: int
    d: int
    mp: np.ndarray
    all_chain_set: np.ndarray
    chain: np.ndarray
    chain_score: calculate.ChainScore
    ground_truth_chain: list
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

def save_stats(df: pd.DataFrame(), path: str):
    df.to_csv(path, index=False)