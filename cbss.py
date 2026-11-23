from itertools import combinations
import numpy as np

def to_bits(s):  return np.array([int(c) for c in s], dtype=np.uint8)
def to_str(b):   return "".join(str(int(x)) for x in b)

def gen_shares(sk, method=1):
    """1-bit method: share_i = sk with the i-th ZERO flipped to 1  -> n = number of zeros.
       2-bit method (Table 2): the 1-bit shares followed by every pair of zeros flipped together."""
    sk = np.asarray(sk, dtype=np.uint8)
    zeros = np.flatnonzero(sk == 0)
    groups = [(z,) for z in zeros]
    if method == 2:
        groups += list(combinations(zeros, 2))
    shares = []
    for g in groups:
        s = sk.copy()
        s[list(g)] = 1
        shares.append(s)
    return shares

def reconstruct(shares, k):
    """Column-wise count of ones; output bit = 1 if count >= k else 0 (Algorithm 1, step 7)."""
    counts = np.sum(np.asarray(shares, dtype=np.int64), axis=0)
    return (counts >= k).astype(np.uint8), counts

def find_valid_subsets(shares, sk, k, limit=None):
    """Algorithm 1 steps 4-10: try k-subsets until reconstruction equals sk."""
    out = []
    for idx in combinations(range(len(shares)), k):
        rec, _ = reconstruct([shares[i] for i in idx], k)
        if np.array_equal(rec, sk):
            out.append(idx)
            if limit and len(out) >= limit:
                break
    return out

if __name__ == "__main__":
    # --- Table 1: 1-bit shares of sk = 101001100
    sk = to_bits("101001100")
    t1 = [to_str(s) for s in gen_shares(sk, 1)]
    print("Table 1 matches:", t1 == ["111001100","101101100","101011100","101001110","101001101"])
    # --- Table 2: 2-bit shares (15 of them)
    t2 = [to_str(s) for s in gen_shares(sk, 2)]
    exp2 = ["111001100","101101100","101011100","101001110","101001101","111101100","111011100",
            "111001110","111001101","101111100","101101110","101101101","101011110","101011101","101001111"]
    print("Table 2 matches:", t2 == exp2)
    # --- Table 3: SH1, SH3, SH4, SH10 with k = 4
    sh = gen_shares(sk, 2)
    rec, counts = reconstruct([sh[0], sh[2], sh[3], sh[9]], 4)
    print("Table 3 counts:", to_str(counts), "(paper: 414124410)  output:", to_str(rec), "(paper: 101001100)")
    # --- Worked example, sk = 101101010, k = 3, shares SH2,SH3,SH4
    sk2 = to_bits("101101010")
    sh2 = gen_shares(sk2, 1)
    print("Example shares:", [to_str(s) for s in sh2])
    rec, counts = reconstruct(sh2[1:4], 3)
    print("Example counts:", to_str(counts), "(paper: 303313131)  output:", to_str(rec))
    # --- Threshold behaviour
    rec, _ = reconstruct(sh2[1:3], 3)
    print("Only 2 of k=3 shares ->", to_str(rec), "(should NOT equal sk)")
    print("Valid 3-subsets of the 4 shares:", len(find_valid_subsets(sh2, sk2, 3)), "of 4")