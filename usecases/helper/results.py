import numpy as np
import os


def save(np_arrays: list, file_path):
    result = np.array(np_arrays)
    if os.path.exists(file_path):
        print('\033[38;5;208m File: ', file_path, ' already exists, but will be overwritten. \033[0m')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, result, allow_pickle=True)
        

def load(file_path):
    result = np.load(file_path, allow_pickle=True)
    return tuple(result)