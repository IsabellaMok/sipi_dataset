"""Baseline pipeline: key -> CBSS shares -> one stego image per share -> extract -> reconstruct key."""
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
import cbss, hamming_me as hm, metrics as mt

def run(img, key_bits=64, k=3, seed=0, method=1):
    rng = np.random.default_rng(seed)
    sk = rng.integers(0, 2, key_bits).astype(np.uint8)
    shares = cbss.gen_shares(sk, method)
    stegos, psnrs, oor = [], [], 0
    for s in shares[:k + 2]:                      # embed a few shares (enough to test)
        st, n_out = hm.embed_share(img, s)
        stegos.append(st); oor += n_out
        psnrs.append(mt.psnr(img, st))
    # receiver: extract shares from stego images, then reconstruct with k of them
    got = [hm.extract_share(st, key_bits) for st in stegos]
    share_ber = np.mean([mt.ber(a, b) for a, b in zip(shares, got)])
    rec, _ = cbss.reconstruct(got[:k], k)
    return dict(n_shares=len(shares), share_BER=share_ber, key_BER=mt.ber(sk, rec),
                PSNR=float(np.mean(psnrs)), SSIM=mt.ssim(img, stegos[0]), out_of_range=oor)

if __name__ == "__main__":
    try:
        from dataset import load_dataset
        images = load_dataset()
    except Exception as e:
        print("Could not load dataset (", e, ") -> using synthetic image")
        rng = np.random.default_rng(0)
        images = {"synthetic": np.clip(np.cumsum(rng.normal(0, 3, (512, 512)), axis=1) % 256, 0, 255).astype(np.uint8)}
    print(f"{'Image':<10}{'key':<6}{'n_sh':<6}{'shareBER':<10}{'keyBER':<8}{'PSNR':<9}{'SSIM':<8}{'oor'}")
    for name, img in images.items():
        for L in (64, 256, 512, 1024, 3072):
            r = run(img, key_bits=L)
            print(f"{name:<10}{L:<6}{r['n_shares']:<6}{r['share_BER']:<10.3f}{r['key_BER']:<8.3f}"
                  f"{r['PSNR']:<9.2f}{r['SSIM']:<8.4f}{r['out_of_range']}")