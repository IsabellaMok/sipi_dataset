"""Hamming matrix embedding on 2x2 blocks, following Saeidi et al. (2024) Sec. 3.2-3.3.
r = 3 -> Hamming(7,4): 3 secret bits hidden in a 7-bit vector x (at most 1 bit of x changes).
r = 4 -> Hamming(15,11): 4 secret bits in 15 bits (use for your extension experiments).
Convention (deduced from the paper's HL and LH worked examples): x is built from the 2 LSBs of
the first three coefficients (LSB first, then 2nd LSB) plus the LSB of the 4th coefficient."""
# --- project layout bootstrap (added by reorganize.py) ---
import os as _os, sys as _sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent
while not (_ROOT / "full_pipeline.py").exists() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
for _d in ("cbss", "data", "ecc", "eval", "iwt", "lsh"):
    _sys.path.insert(0, str(_ROOT / _d))
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# --- end bootstrap ---
import numpy as np
from iwt import iwt2, iiwt2

def H_matrix(r=3):
    n = 2**r - 1
    return np.array([[(j >> (r - 1 - i)) & 1 for j in range(1, n + 1)] for i in range(r)], dtype=np.int64)

H3 = H_matrix(3)   # [[0001111],[0110011],[1010101]]  == paper's H_r

def block_to_x(block):
    v = np.asarray(block, dtype=np.int64).ravel()          # raster order a,b,c,d
    x = []
    for c in v[:3]:
        x += [int(c) & 1, (int(c) >> 1) & 1]
    x.append(int(v[3]) & 1)
    return np.array(x, dtype=np.int64)

def x_to_block(x, block):
    v = np.asarray(block, dtype=np.int64).ravel().copy()
    for i in range(3):
        v[i] = (v[i] & ~3) | int(x[2*i]) | (int(x[2*i+1]) << 1)
    v[3] = (v[3] & ~1) | int(x[6])
    return v.reshape(2, 2)

def embed_block(block, m3):
    """Hide 3 bits m3 (tuple/array, first = MSB) in one 2x2 block."""
    x = block_to_x(block)
    s = H3.dot(x) % 2
    r = s ^ np.asarray(m3, dtype=np.int64)
    j = int(r[0]) * 4 + int(r[1]) * 2 + int(r[2])
    if j:
        x[j - 1] ^= 1
    return x_to_block(x, block)

def extract_block(block):
    return H3.dot(block_to_x(block)) % 2

# ---------- whole-image helpers: a share (bit array) -> stego image ----------
def _blocks(sub):                      # raster list of 2x2 block coordinates
    h, w = sub.shape
    return [(i, j) for i in range(0, h, 2) for j in range(0, w, 2)]

def embed_share(img, share_bits, first_block=0):
    """9 bits -> (HL, LH, HH) blocks. Longer shares use consecutive blocks (pad with 0 to a multiple of 9)."""
    bits = np.asarray(share_bits, dtype=np.int64)
    pad = (-len(bits)) % 9
    bits = np.concatenate([bits, np.zeros(pad, dtype=np.int64)])
    LL, HL, LH, HH = iwt2(img)
    subs = [HL.copy(), LH.copy(), HH.copy()]
    pos = _blocks(HL)
    for t in range(len(bits) // 9):
        i, j = pos[first_block + t]
        for b, sub in enumerate(subs):
            sub[i:i+2, j:j+2] = embed_block(sub[i:i+2, j:j+2], bits[9*t + 3*b: 9*t + 3*b + 3])
    raw = iiwt2(LL, *subs)
    n_out = int(np.sum((raw < 0) | (raw > 255)))
    return np.clip(raw, 0, 255).astype(np.uint8), n_out

def extract_share(stego, n_bits, first_block=0):
    _, HL, LH, HH = iwt2(stego)
    pos = _blocks(HL)
    out = []
    for t in range((n_bits + 8) // 9):
        i, j = pos[first_block + t]
        for sub in (HL, LH, HH):
            out += list(extract_block(sub[i:i+2, j:j+2]))
    return np.array(out[:n_bits], dtype=np.uint8)

if __name__ == "__main__":
    # Paper's worked examples (Fig. 2): HL block and LH block
    HL = np.array([[160, 64], [123, 119]]); LH = np.array([[88, 117], [125, 101]])
    print("H3 =\n", H3)
    print("HL x =", block_to_x(HL), "(paper 0000111) s =", H3.dot(block_to_x(HL)))
    print("HL' =", embed_block(HL, (1, 1, 1)).tolist(), "(paper [[160,65],[123,119]])")
    print("LH x =", block_to_x(LH), "(paper 0010101)")
    print("LH' =", embed_block(LH, (1, 0, 1)).tolist(), "(paper [[88,119],[125,101]])")
    print("extract HL' ->", extract_block(embed_block(HL, (1, 1, 1))), "(paper 111)")