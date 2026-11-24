"""Error-correcting codes for protecting a 256-bit hash sent over a noisy public channel.
 - Cyclic codes (Hamming, BCH) with vectorised syndrome-table decoding (bounded distance, t errors/block)
 - Reed-Solomon over GF(2^8) (via the `reedsolo` package) - symbol (byte) oriented, good for bursts
 - r-fold repetition of the whole codeword with bitwise majority vote (optionally around an inner code)
Every scheme exposes: name, encode(bits[B,256]) -> bits[B,T], decode(bits[B,T]) -> bits[B,256], tx_bits, n_trials."""
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
from itertools import combinations
import numpy as np

HASH_BITS = 256

def _polymod(a, g):
    dg = g.bit_length() - 1
    while a.bit_length() - 1 >= dg and a:
        a ^= g << (a.bit_length() - 1 - dg)
    return a

def _bits_of(x, n):
    return [(x >> i) & 1 for i in range(n)]

class NoECC:
    name, n_trials = "none", 4000
    tx_bits = HASH_BITS
    def encode(self, b): return b.copy()
    def decode(self, r): return r.copy()

class Cyclic:
    """Systematic cyclic (n,k) code, generator polynomial g (int bitmask, bit i = coeff of x^i), corrects t errors."""
    n_trials = 4000
    def __init__(self, name, n, g, t):
        self.name, self.n, self.t = name, n, t
        self.r = g.bit_length() - 1; self.k = n - self.r
        assert _polymod((1 << n) | 1, g) == 0, f"{name}: g does not divide x^{n}+1"
        # generator matrix rows: x^(r+i) + (x^(r+i) mod g); message occupies positions r..n-1
        self.G = np.array([_bits_of((1 << (self.r + i)) ^ _polymod(1 << (self.r + i), g), n) for i in range(self.k)], dtype=np.uint8)
        self.S = np.array([_bits_of(_polymod(1 << j, g), self.r) for j in range(n)], dtype=np.uint8)   # syndrome map
        self.w = (1 << np.arange(self.r)).astype(np.int64)
        size = 1 << self.r
        self.pat = np.zeros((size, n), dtype=np.uint8); self.ok = np.zeros(size, dtype=bool); self.ok[0] = True
        for wt in range(1, t + 1):
            for pos in combinations(range(n), wt):
                e = np.zeros(n, dtype=np.uint8); e[list(pos)] = 1
                s = int(((e.astype(np.int64) @ self.S) % 2) @ self.w)
                assert not self.ok[s], f"{name}: cannot correct {t} errors (syndrome collision)"
                self.ok[s] = True; self.pat[s] = e
        self.blocks = -(-HASH_BITS // self.k)
        self.tx_bits = self.blocks * n
    def encode(self, b):
        B = b.shape[0]
        m = np.zeros((B, self.blocks * self.k), dtype=np.uint8); m[:, :HASH_BITS] = b
        c = (m.reshape(B, self.blocks, self.k).astype(np.int32) @ self.G.astype(np.int32)) % 2
        return c.reshape(B, -1).astype(np.uint8)
    def decode(self, r):
        B = r.shape[0]; R = r.reshape(B, self.blocks, self.n)
        syn = ((R.astype(np.int32) @ self.S.astype(np.int32)) % 2) @ self.w
        fix = np.where(self.ok[syn][..., None], self.pat[syn], 0).astype(np.uint8)
        C = R ^ fix
        return C[:, :, self.r:].reshape(B, -1)[:, :HASH_BITS]

class RS:
    """Reed-Solomon RS(32+2t, 32) over GF(256); corrects t byte errors."""
    n_trials = 300
    def __init__(self, t):
        import reedsolo
        self.rs = reedsolo.RSCodec(2 * t); self.t = t
        self.name = f"RS({32 + 2*t},32) t={t}B"; self.tx_bits = (32 + 2 * t) * 8
    def encode(self, b):
        return np.array([np.unpackbits(np.frombuffer(bytes(self.rs.encode(bytes(np.packbits(row)))), dtype=np.uint8)) for row in b], dtype=np.uint8)
    def decode(self, r):
        out = []
        for row in r:
            data = bytes(np.packbits(row))
            try:
                msg = bytes(self.rs.decode(data)[0])
            except Exception:
                msg = data[:32]                       # uncorrectable: pass the received data bytes through
            out.append(np.unpackbits(np.frombuffer(msg, dtype=np.uint8)))
        return np.array(out, dtype=np.uint8)

class Repeat:
    """Send the (inner-coded) word `reps` times; majority vote per bit, then decode with the inner code."""
    def __init__(self, inner=None, reps=3):
        self.inner = inner or NoECC(); self.reps = reps
        self.name = f"rep{reps}" + ("" if isinstance(self.inner, NoECC) else "+" + self.inner.name)
        self.tx_bits = reps * self.inner.tx_bits; self.n_trials = self.inner.n_trials
    def encode(self, b):
        return np.tile(self.inner.encode(b), (1, self.reps))
    def decode(self, r):
        B = r.shape[0]; T = self.inner.tx_bits
        votes = r.reshape(B, self.reps, T).sum(axis=1)
        return self.inner.decode((2 * votes > self.reps).astype(np.uint8))

def all_schemes():
    ham15 = Cyclic("Hamming(15,11)", 15, 0b10011, 1)
    bch31 = Cyclic("BCH(31,21) t=2", 31, 0b11101101001, 2)
    rs4 = RS(2)
    return [NoECC(), Cyclic("Hamming(7,4)", 7, 0b1011, 1), ham15,
            Cyclic("BCH(15,7) t=2", 15, 0b111010001, 2), bch31, Cyclic("BCH(63,51) t=2", 63, 0b1010100111001, 2),
            rs4, RS(4), Repeat(), Repeat(ham15), Repeat(bch31), Repeat(rs4)]

# ---------- channels ----------
def bsc(bits, p, rng):
    return bits ^ (rng.random(bits.shape) < p).astype(np.uint8)

def burst(bits, length, rng):
    """One burst: `length` consecutive transmitted bits replaced by random bits (lost data)."""
    B, T = bits.shape; length = min(length, T)
    start = rng.integers(0, T - length + 1, B)
    idx = start[:, None] + np.arange(length)[None, :]
    out = bits.copy(); out[np.arange(B)[:, None], idx] = rng.integers(0, 2, (B, length))
    return out

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for s in all_schemes():
        h = rng.integers(0, 2, (50, HASH_BITS)).astype(np.uint8)
        c = s.encode(h)
        ok_clean = np.array_equal(s.decode(c), h)
        print(f"{s.name:<24} tx={s.tx_bits:<4} overhead={s.tx_bits/HASH_BITS:.2f}x  clean round-trip: {ok_clean}")
    # t-error guarantee check on one block of each cyclic code
    for s in all_schemes():
        if isinstance(s, Cyclic):
            h = rng.integers(0, 2, (200, HASH_BITS)).astype(np.uint8); c = s.encode(h)
            bad = c.copy()
            for i in range(200):                       # exactly t errors in the first block
                pos = rng.choice(s.n, s.t, replace=False); bad[i, pos] ^= 1
            print(f"{s.name:<24} corrects {s.t} errors in a block:", np.array_equal(s.decode(bad), h))