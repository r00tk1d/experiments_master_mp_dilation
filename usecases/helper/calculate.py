import stumpy
from . import results
from . import utils

def chains(T, ds, target_w, data_name, use_case):
    for d in ds:
        m = round((target_w-1)/d) + 1
        file_name =  data_name + "_d" + str(d) + "_m" + str(m)
        file_path = "../results/" + use_case + "/" + data_name + "/" + file_name

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        all_chain_set, unanchored_chain = stumpy.allc(mp[:, 2], mp[:, 3])
        all_non_overlapping_chain_set, non_overlapping_unanchored_chain = utils.remove_overlapping_chains(all_chain_set, m, d)

        max_distance_in_unanchored_chain = unanchored_chain[-1] - unanchored_chain[0]
        max_distance_in_non_overlapping_unanchored_chain = non_overlapping_unanchored_chain[-1] - non_overlapping_unanchored_chain[0]
        
        results.save([T, m, d, mp, all_chain_set, all_non_overlapping_chain_set, unanchored_chain, non_overlapping_unanchored_chain, max_distance_in_unanchored_chain, max_distance_in_non_overlapping_unanchored_chain], file_path + ".npy")