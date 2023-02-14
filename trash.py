
# Arguments:
ts = np.array([1, 0, 3, 9, 2, 1, 1, 15, 3, 14, 2, 10, 7]).astype(np.float64)
index_original = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12])
window_size = 3
d = 2

# mp_orig = stumpy.stump(ts, m=window_size, ignore_trivial=False, d=d)

def dilation_mapping(X: np.ndarray, d: int) -> np.ndarray:
    result = X[0::d]
    for i in range(1, d):
        next = X[i::d]
        result = np.concatenate((result, next))
    return result


# 1. TS -dilationMaping-> dilated TS
ts_d = dilation_mapping(ts, d)
index_dilated = dilation_mapping(index_original, d)
print(f"TS: {ts}")
print(f"Index: {index_original}")
print(f"TS dilated: {ts_d}")
print(f"Index dilated: {index_dilated}")

# 2. dilated TS, m -stumpy-> MP 
mp_dilated_unfixed = stumpy.stump(ts_d, m=window_size)
print(f"MP dilated unfixed:  {mp_dilated_unfixed}")

# 3. MP, NNIndex -dilationMapping-> fixedMP                            muss hier auch MP nochmal gemappt werden?
mp_dilated_fixed = dilation_mapping(mp_dilated_unfixed, d)
index_dilated_fixed = dilation_mapping(index_dilated, d)
print(f"Index dilated fixed: {index_dilated_fixed}")
print(f"MP dilated fixed: {mp_dilated_fixed}")

# 4. Test Use Cases
# ...



# mp[mp_idx, 0] -> Euclidean Distance
# mp[mp_idx, 1] -> Index Nearest Neighbor
# mp[mp_idx, 2] -> Index “left” Nearest Neighbor
# mp[mp_idx, 3] -> Index “right” Nearest Neighbor


# Altes Index Mapping:
def ceildiv(a, b):
    return -(a // -b)

def fix_index(x,d,n):
    return ceildiv(n,d) * (x % d) + (x // d)

T_A = np.array([1,2,3,6,6,4,7,1,2,3,2,3,12,3,1,2,4,12,13,4,5,6,21,1,4,2]).astype(np.float64)
n = T_A.shape[0]
d = 5

i = 1
new_i = fix_index(i,d,n)
print(new_i)

# for testing all indice:
index = np.arange(n)
new_list = [fix_index(x,d,n) for x in index]
print(new_list)