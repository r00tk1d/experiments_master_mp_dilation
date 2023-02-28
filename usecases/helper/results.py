import numpy as np
import os


def save(np_arrays: list, file_path):
    result = np.array(np_arrays)
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, result, allow_pickle=True)
    else:
        print('\033[38;5;208m File already exists, not saving. \033[0m')

def load(file_path):
    result = np.load(file_path, allow_pickle=True)
    return tuple(result)
    