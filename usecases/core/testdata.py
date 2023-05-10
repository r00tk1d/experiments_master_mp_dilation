import numpy.typing as npt

import numpy as np

from numpy import loadtxt
import pandas as pd
from scipy.io import loadmat

### Robustness Datanames ###


def load_ucr_dataset_from_tsv(ucr_path: str, dataset_name: str, train_test: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:

    path = ucr_path + "/" + dataset_name + "/" + dataset_name + "_" + train_test + ".tsv"
    with open(path, 'r') as f:
        data = [line.strip().split('\t') for line in f]

    data_arr = np.array(data)

    labels = data_arr[:, 0].astype(np.int64)
    time_series = data_arr[:, 1:].astype(np.float64)

    return time_series, labels

def load_all_ucr_datasets(ucr_path: str) -> any:
    pass

def load_fast_ucr_datasets(ucr_path: str) -> any:
    pass

### .txt ###
def load_from_txt(path: str) -> npt.NDArray[np.float64]:
    return loadtxt(path, dtype=np.float64)

### .csv ###
def load_from_csv(path: str, column: str) -> npt.NDArray[np.float64]:
    return pd.read_csv(path, sep='\t', lineterminator='\r', header=None)[5].to_numpy(dtype=np.float64)

### .mat ###
def load_from_mat(path: str, column: str) -> npt.NDArray[np.float64]:
    mat_content = loadmat(path)
    return np.array([x[0] for x in mat_content[column]]).astype(np.float64)

def load_gt(path: str, column: str) -> npt.NDArray[np.float64]:
    mat_content = loadmat(path)
    return list(mat_content[column][0])