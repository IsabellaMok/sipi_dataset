"""Hamming (7,4) [r=3] vs Hamming (15,11) [r=4] matrix embedding in the IWT detail sub-bands (HL, LH, HH).

Both use the SAME cover-bit stream: for every 2x2 block of a sub-band, 7 bits (2 LSBs of the first three
coefficients + LSB of the 4th, the convention of hamming_me.py). A group of n = 2^r - 1 stream bits hides r secret
bits and changes at most 1 of them.
   r=3: 7 bits hide 3   (0.43 secret bit per cover bit, expected changes 0.292 per secret bit)
   r=4: 15 bits hide 4  (0.27 secret bit per cover bit, expected changes 0.234 per secret bit)
=> (15,11) distorts less per hidden bit but has lower capacity. This script measures it.
Run:  python hamming_compare.py        Output: results/hamming_compare.csv
"""
import csv, sys
from pathlib import Path
import numpy as np
import hamming_me as hm, metrics as mt
from iwt import iwt2, iiwt2

OUT = Path("results"); OUT.mkdir(exist_ok=True)
PAYLOADS = (1024, 4000, 16384, 65536)


def _blocks4(A):                                   # (h,w) -> (n_blocks, 4): a,b,c,d of every 2x2 block, raster order
    h, w = A.shape
    return A.reshape(h // 2, 2, w // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)


def _unblocks4(B, shape):
    h, w = shape
    return B.reshape(h // 2, w // 2, 2, 2).transpose(0, 2, 1, 3).reshape(h, w)


def to_stream(A):
    B = _blocks4(A.astype(np.int64))
    X = np.stack([B[:, 0] & 1, (B[:, 0] >> 1) & 1, B[:, 1] & 1, (B[:, 1] >> 1) & 1,
                  B[:, 2] & 1, (B[:, 2] >> 1) & 1, B[:, 3] & 1], axis=1)
    return X.ravel().astype(np.int64)


def from_stream(S, A):
    X = S.reshape(-1, 7); B = _blocks4(A.astype(np.int64)).copy()
    for c in range(3):
        B[:, c] = (B[:, c] & ~3) | X[:, 2 * c] | (X[:, 2 * c + 1] << 1)
    B[:, 3] = (B[:, 3] & ~1) | X[:, 6]
    return _unblocks4(B, A.shape)


def capacity(img_shape, r):
    n = 2 ** r - 1
    nblocks = 3 * (img_shape[0] // 2 // 2) * (img_shape[1] // 2 // 2)
    return (nblocks * 7 // n) * r


def embed_bits(img, bits, r):
    n = 2 ** r - 1; H = hm.H_matrix(r); w = 1 << np.arange(r - 1, -1, -1)
    LL, HL, LH, HH = iwt2(img)
    subs = [HL, LH, HH]
    S = np.concatenate([to_stream(s) for s in subs]); S0 = S.copy()
    bits = np.asarray(bits, dtype=np.int64)
    bits = np.concatenate([bits, np.zeros((-len(bits)) % r, dtype=np.int64)])
    G = len(bits) // r
    assert G * n <= len(S), "payload exceeds capacity"
    X = S[:G * n].reshape(G, n)
    syn = (X @ H.T) % 2
    j = ((syn ^ bits.reshape(G, r)) * w).sum(1)
    idx = np.flatnonzero(j)
    X[idx, j[idx] - 1] ^= 1
    S[:G * n] = X.ravel()
    per = len(S) // 3
    new = [from_stream(S[k * per:(k + 1) * per], subs[k]) for k in range(3)]
    raw = iiwt2(LL, *new)
    n_clip = int(np.sum((raw < 0) | (raw > 255)))
    return np.clip(raw, 0, 255).astype(np.uint8), int((S != S0).sum()), n_clip


def extract_bits(stego, n_bits, r):
    n = 2 ** r - 1; H = hm.H_matrix(r)
    _, HL, LH, HH = iwt2(stego)
    S = np.concatenate([to_stream(s) for s in (HL, LH, HH)])
    G = (n_bits + r - 1) // r
    return ((S[:G * n].reshape(G, n) @ H.T) % 2).ravel()[:n_bits].astype(np.uint8)


def self_test():
    rng = np.random.default_rng(0)
    img = rng.integers(40, 215, (64, 64)).astype(np.uint8)
    for r in (3, 4):
        m = rng.integers(0, 2, 500)
        st, _, nc = embed_bits(img, m, r)
        assert nc == 0 and np.array_equal(extract_bits(st, 500, r), m), f"round trip failed for r={r}"
    # r=3 must match the block-wise embed_block of hamming_me.py on one block
    LL, HL, LH, HH = iwt2(img)
    S = to_stream(HL)[:7]; H = hm.H_matrix(3); m3 = np.array([1, 0, 1])
    s = (H @ S) % 2; j = int(((s ^ m3) * np.array([4, 2, 1])).sum())
    if j: S[j - 1] ^= 1
    ref = hm.embed_block(HL[0:2, 0:2], m3)
    got = from_stream(np.concatenate([S, to_stream(HL)[7:]]), HL)[0:2, 0:2]
    assert np.array_equal(ref, got), "does not match hamming_me.embed_block"
    print("self-test passed (round trip r=3 and r=4; r=3 equals hamming_me.embed_block)\n")


def load_images():
    try:
        from dataset import load_dataset
        return load_dataset()
    except Exception as e:
        print(f"(dataset not found: {e}) using a synthetic image")
        rng = np.random.default_rng(0)
        return {"synthetic": np.clip(np.cumsum(rng.normal(0, 3, (512, 512)), axis=1) % 256, 0, 255).astype(np.uint8)}


if __name__ == "__main__":
    self_test()
    images = load_images(); rng = np.random.default_rng(7); rows = []
    for name, img in images.items():
        for r, label in ((3, "Hamming(7,4) r=3"), (4, "Hamming(15,11) r=4")):
            cap = capacity(img.shape, r)
            for L in PAYLOADS:
                if L > cap:
                    continue
                bits = rng.integers(0, 2, L).astype(np.uint8)
                st, changed, n_clip = embed_bits(img, bits, r)
                got = extract_bits(st, L, r)
                rows.append(dict(image=name, scheme=label, payload_bits=L, capacity_bits=cap,
                                 bpp=round(L / img.size, 5), cover_bits_changed=changed,
                                 changes_per_secret_bit=round(changed / L, 4), pixels_clipped=n_clip,
                                 BER=round(mt.ber(bits, got), 5), MSE=round(mt.mse(img, st), 6),
                                 PSNR=round(mt.psnr(img, st), 2), SSIM=round(mt.ssim(img, st), 6)))
    with open(OUT / "hamming_compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{'scheme':<22}{'payload':<9}{'capacity':<10}{'chg/bit':<9}{'BER':<9}{'PSNR':<9}{'SSIM':<10}{'clipped'}   (mean over images)")
    for sc in ("Hamming(7,4) r=3", "Hamming(15,11) r=4"):
        for L in PAYLOADS:
            R = [x for x in rows if x["scheme"] == sc and x["payload_bits"] == L]
            if not R:
                print(f"{sc:<22}{L:<9}exceeds capacity"); continue
            m = lambda k: np.mean([x[k] for x in R])
            print(f"{sc:<22}{L:<9}{R[0]['capacity_bits']:<10}{m('changes_per_secret_bit'):<9.3f}{m('BER'):<9.4f}"
                  f"{m('PSNR'):<9.2f}{m('SSIM'):<10.6f}{int(sum(x['pixels_clipped'] for x in R))}")
    print("\nsaved results/hamming_compare.csv")