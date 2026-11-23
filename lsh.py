"""Three locality-sensitive / similarity-preserving hashes for image authentication.
All return a binary numpy array of n_bits (uint8) and are KEYED by `secret`.
They hash the LL sub-band of the 1-level integer 5/3 IWT: the Hamming-based embedding only touches the
HL/LH/HH sub-bands, so the LL sub-band of the stego image equals that of the cover (no clipping assumed).

  simhash  : random-hyperplane LSH (Charikar 2002)  -> P[bit differs] = angle/pi   (cosine similarity)
  phash    : DCT-based perceptual hash (low-frequency DCT coefficients vs median), XOR-masked with a key
  minhash  : 1-bit b-bit minwise hashing (Broder 1997; Li & Konig 2010) over (cell, level) features -> Jaccard
"""
import numpy as np
from iwt import iwt2

def ll_subband(img):
    return iwt2(img)[0].astype(np.float64)

def block_means(a, g):
    h, w = a.shape
    return a.reshape(g, h // g, g, w // g).mean(axis=(1, 3))

def hamming(a, b):
    return int(np.sum(np.asarray(a) != np.asarray(b)))

# ------------------------------------------------------------------ SimHash
def simhash_features(img, g=16):
    f = block_means(ll_subband(img), g).ravel()
    return f - f.mean()

def simhash(img, n_bits=256, g=16, secret=1234):
    f = simhash_features(img, g)
    W = np.random.default_rng(secret).normal(size=(n_bits, f.size))
    return (W @ f > 0).astype(np.uint8)

def simhash_predicted_distance(img_a, img_b, n_bits=256, g=16):
    """Expected Hamming distance n*theta/pi, where theta is the angle between the two feature vectors."""
    fa, fb = simhash_features(img_a, g), simhash_features(img_b, g)
    c = fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12)
    return n_bits * float(np.arccos(np.clip(c, -1, 1))) / np.pi

# ------------------------------------------------------------------ pHash (DCT)
def _dct_matrix(N):
    k, n = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    C = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    C[0, :] = np.sqrt(1.0 / N)
    return C

def phash(img, n_bits=256, size=64, secret=1234):
    small = block_means(ll_subband(img), size)                 # size x size
    C = _dct_matrix(size)
    D = C @ small @ C.T
    s = int(np.ceil(np.sqrt(n_bits + 1)))
    low = D[:s, :s].ravel()[1:n_bits + 1]                      # low-frequency AC coefficients (DC dropped)
    bits = (low > np.median(low)).astype(np.uint8)
    mask = np.random.default_rng(secret).integers(0, 2, n_bits).astype(np.uint8)
    return bits ^ mask                                         # keyed; Hamming distances unchanged

# ------------------------------------------------------------------ 1-bit MinHash
_P = 2**31 - 1

def minhash(img, n_bits=256, g=16, levels=16, secret=1234):
    bm = block_means(ll_subband(img), g)
    q = np.clip((bm / 256.0 * levels).astype(np.int64), 0, levels - 1)
    elems = (np.arange(g * g, dtype=np.int64) * levels + q.ravel())       # one (cell, level) element per cell
    rng = np.random.default_rng(secret)
    a = rng.integers(1, _P, n_bits, dtype=np.int64); b = rng.integers(0, _P, n_bits, dtype=np.int64)
    hv = (a[:, None] * elems[None, :] + b[:, None]) % _P
    return (hv.min(axis=1) & 1).astype(np.uint8)

HASHES = {"SimHash": simhash, "pHash": phash, "MinHash": minhash}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = np.clip(128 + 50 * np.sin(np.add.outer(np.arange(512), np.arange(512)) / 37) + rng.normal(0, 5, (512, 512)), 0, 255).astype(np.uint8)
    other = np.clip(rng.normal(128, 40, (512, 512)), 0, 255).astype(np.uint8)
    for name, f in HASHES.items():
        h = f(img)
        print(f"{name:<8} bits={h.size}  same image d={hamming(h, f(img))}  "
              f"different image d={hamming(h, f(other))}  different key d={hamming(h, f(img, secret=99))}")