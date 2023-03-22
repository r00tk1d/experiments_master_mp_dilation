import stumpy
from . import results
from . import utils
from tssb.evaluation import covering

def chains(T, ds, target_w, data_name, use_case):
    for d in ds:
        m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1
        file_name =  data_name + "_d" + str(d) + "_m" + str(m)
        file_path = "../results/" + use_case + "/" + data_name + "/" + "target_w" + str(target_w) + "/" + file_name

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        print("Calculated MP for: w=" + str(actual_w) + ", m=" + str(m) + ", d=" + str(d))
        all_chain_set, unanchored_chain = stumpy.allc(mp[:, 2], mp[:, 3])
        all_non_overlapping_chain_set, non_overlapping_unanchored_chain = utils.remove_overlapping_chains(all_chain_set, m, d)

        max_distance_in_unanchored_chain = unanchored_chain[-1] - unanchored_chain[0]
        max_distance_in_non_overlapping_unanchored_chain = non_overlapping_unanchored_chain[-1] - non_overlapping_unanchored_chain[0]
        
        results.save([T, m, d, mp, all_chain_set, all_non_overlapping_chain_set, unanchored_chain, non_overlapping_unanchored_chain, max_distance_in_unanchored_chain, max_distance_in_non_overlapping_unanchored_chain], file_path + ".npy")

def segmentation(T, T_name, cps, ds, target_w, L, n_regimes):
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
            print(f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, Score: {score} for d={d}, m={m}, w={actual_w}")
            scores.append(score)
        return scores
