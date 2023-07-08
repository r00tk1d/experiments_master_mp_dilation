def segmentation_fluss_unknown_cps_ensemble_min(T, T_name, cps, ds, L, threshold, target_w, m):
    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    mps = []

    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
            mps.append(mp)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
            mps.append(mp)

    mps = _adjust_mps_to_same_length(mps)
    mp = _min_mp(mps)

    cac, _ = stumpy.fluss(mp[:, 1], L=L, n_regimes=1)
    found_cps = _rea_unknown_cps(cac, L, threshold)
    score = covering({0: cps}, found_cps, T.shape[0])
    print(
        f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, Score: {score}")
    return score

def segmentation_fluss_known_cps_ensemble_min(T, T_name, cps, ds, L, n_regimes, target_w, m):
    assert (target_w is None) != (m is None)
    if target_w:
        calculate_m = True
    else:
        calculate_m = False

    mps = []
    
    for d in ds:
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
            mps.append(mp)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
            mps.append(mp)

    mps = _adjust_mps_to_same_length(mps)
    mp = _min_mp(mps)
        
    cac, found_cps = stumpy.fluss(mp[:,1], L=L, n_regimes=n_regimes)
    score = covering({0: cps}, found_cps, T.shape[0])
    print(f"Time Series: {T_name}: True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, CAC values: {[cac[cp] for cp in found_cps]}, Score: {score}")
    return score

def _min_mp(mps):
    min_mp = []
    for i in range(len(mps[0])):
        distances = [mp[i,0] for mp in mps]
        index_min = min(range(len(distances)), key=distances.__getitem__)
        min_mp.append(mps[index_min][i])
    return np.array(min_mp).astype(np.float64)

def _adjust_mps_to_same_length(mps):
    max_length = max(len(mp) for mp in mps)
    adjusted_mps = []
    for mp in mps:
        len_diff = max_length - len(mp)
        if len_diff == 0:
            adjusted_mps.append(mp)
            continue
        else:
            filler = np.array([np.array([np.NINF,-1,-1,-1], dtype = object) for _ in range(len_diff)])
            adjusted_mp = np.concatenate((mp, filler), axis=0)
            adjusted_mps.append(adjusted_mp)
    return adjusted_mps


def chains_ensemble_mp_min(T, max_dilation, data_name, use_case, ground_truth_chain, offset, non_overlapping, target_w, m, offset_value=None):
    """
    Calculates the chains for a given time series {T} and a given list of dilations {ds}.
    The chains are calculated for a given target window range {target_w}.
    If {offset} is set to true, the chains with a dilation size above 1 are calculated with an offset determined by the starting point of the unanchored chain without dilation. 
    If {ground_truth} is None, ground_truth set to the unanchored chain without dilation.
    """
    assert (target_w is None) != (m is None), "only one of target_w and m can be set"
    assert not (offset and non_overlapping), "offset and non_overlapping cannot be true at the same time"	
    if target_w:
        calculate_m = True
    else:
        calculate_m = False
    
    offset_start = 0
    ground_truth_given = ground_truth_chain is not None

    print(f"Running Experiment: target_w={target_w}, m={m}, offset={offset!s}, groundtruthD1={not ground_truth_given!s}, nonoverlapping={non_overlapping!s}")

    mps = []
    for d in range(1, max_dilation+1):
        if calculate_m:
            m = round((target_w-1)/d) + 1
        actual_w = (m-1)*d + 1

        if d == 1:
            mp = stumpy.stump(T, m=m)
        else:
            mp = stumpy.stump_dil(T, m=m, d=d)
        mps.append(mp)

    # ensemble mp with min at each offset
    mps = _adjust_mps_to_same_length(mps)
    mp = _min_mp(mps)

    print(f"Calculated min MP")
    if offset and offset_value is not None:
        offset_start = offset_value
        chain = stumpy.atsc(mp[:, 2], mp[:, 3], offset_start)
        all_chain_set = [chain]
    elif offset and d != 1:
        chain = stumpy.atsc(mp[:, 2], mp[:, 3], offset_start)
        all_chain_set = [chain]
    else:
        all_chain_set, chain = stumpy.allc(mp[:, 2], mp[:, 3])

    if non_overlapping:
        all_chain_set, chain = utils.remove_overlapping_chains(all_chain_set, m, d)
        chain = np.array([x + offset_start for x in chain])

    if not ground_truth_chain and d==1:
        ground_truth_chain = list(chain)

    chain_score = _get_chain_score(chain, T, d, m, ground_truth_chain)

    return chain_score