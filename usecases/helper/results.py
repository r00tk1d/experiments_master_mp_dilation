import numpy as np
import os


def save(np_arrays: list, file_path):
    result = np.array(np_arrays)
    if not os.path.exists(file_path + ".npy"):
        np.save(file_path + ".npy", result, allow_pickle=True)
    else:
        print('\033[38;5;208m File already exists, not saving. \033[0m')

def load(use_case, data_name, m, d):
    result = np.load("../results/" + use_case + "/" + data_name + "/" + data_name +"_d" + str(d) + "_m" + str(m) + ".npy", allow_pickle=True)
    return tuple(result)

def load_chains(data_name, m, d): # TODO usage überall ersetzen durch save methode
    result = np.load("../results/chains/" + data_name + "/" + data_name +"_d" + str(d) + "_m" + str(m) + ".npy", allow_pickle=True)
    return tuple(result)
    