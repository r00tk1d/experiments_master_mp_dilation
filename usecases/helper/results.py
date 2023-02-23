import numpy as np
import os

def save_chains(T, m, d, mp, all_chain_set, unanchored_chain, file_path):
    result = np.array([T, m, d, mp, all_chain_set, unanchored_chain])
    if not os.path.exists(file_path):
        np.save(file_path, result, allow_pickle=True)
    else:
        print('File already exists, not saving.')

def load_chains(data_name, m, d):
    result = np.load("../results/chains/" + data_name +"_d" + str(d) + "_m" + str(m) + ".npy", allow_pickle=True)
    return tuple(result)
    