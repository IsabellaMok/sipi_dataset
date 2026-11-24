"""Integer 5/3 (LeGall) wavelet transform via lifting. Lossless: integer in, integer out."""
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

def _fwd_1d(x):
    # x: int64 array, even length along axis 0
    e, o = x[0::2].copy(), x[1::2].copy()
    e_next = np.concatenate([e[1:], e[-1:]], axis=0)        # symmetric extension at right edge
    d = o - ((e + e_next) >> 1)                              # detail (high-pass)
    d_prev = np.concatenate([d[:1], d[:-1]], axis=0)         # symmetric extension at left edge
    s = e + ((d_prev + d + 2) >> 2)                          # approximation (low-pass)
    return s, d

def _inv_1d(s, d):
    d_prev = np.concatenate([d[:1], d[:-1]], axis=0)
    e = s - ((d_prev + d + 2) >> 2)
    e_next = np.concatenate([e[1:], e[-1:]], axis=0)
    o = d + ((e + e_next) >> 1)
    x = np.empty((s.shape[0] * 2,) + s.shape[1:], dtype=np.int64)
    x[0::2], x[1::2] = e, o
    return x

def iwt2(img):
    """One-level 2D IWT. Returns LL, HL, LH, HH (each half size)."""
    x = img.astype(np.int64)
    L, H = _fwd_1d(x)                      # along rows (axis 0)
    LL, LH = [t.T for t in _fwd_1d(L.T)]   # along columns
    HL, HH = [t.T for t in _fwd_1d(H.T)]
    return LL, HL, LH, HH

def iiwt2(LL, HL, LH, HH):
    L = _inv_1d(LL.T, LH.T).T
    H = _inv_1d(HL.T, HH.T).T
    return _inv_1d(L, H)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for name, im in [("random", rng.integers(0, 256, (512, 512))),
                     ("all-0", np.zeros((512, 512), int)),
                     ("all-255", np.full((512, 512), 255))]:
        rec = iiwt2(*iwt2(im))
        print(f"{name:8s} perfect reconstruction:", np.array_equal(rec, im))