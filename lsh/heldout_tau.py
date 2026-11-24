"""Held-out check of the LSH threshold (answers: "tau was tuned on only 5 images").

Calibration set = your 5 SIPI images (via dataset.load_dataset). tau is chosen there: the smallest tau with
FRR <= 1%. Then tau is FROZEN and tested on images that were never used to choose it.

Usage (folders hold image files: png, pgm, tif, jpg, bmp; any size, they are centre-cropped/resized to 512x512):
    python heldout_tau.py heldout_images
    python heldout_tau.py heldout_images bossbase_subset      # several held-out sets, reported separately
Outputs: results/heldout_summary.csv, results/heldout_sweep.csv
"""
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
import sys, csv
from pathlib import Path
import numpy as np
from PIL import Image
import lsh, lsh_experiment as le

OUT = Path("results"); OUT.mkdir(exist_ok=True)
N = 256
TRIALS = 10
PATCHES = (8, 16, 32, 64)
MIN_PATCH = 16
FRR_TARGET = 0.01
TAU_FIXED = 3
EXTS = {".png", ".pgm", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def load_folder(folder, limit=None):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)
    if limit:
        files = files[:limit]
    out = {}
    for p in files:
        im = Image.open(p).convert("L")
        w, h = im.size; s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        im = im.crop((l, t, l + s, t + s))
        if s != 512:
            im = im.resize((512, 512), Image.LANCZOS)
        out[p.stem] = np.array(im, dtype=np.uint8)
    return out


def collect(images):
    """Hamming distances: benign (embedding + 6 channel distortions) and malicious (patch noise / copy-move)."""
    B, M = [], {}
    for img in images.values():
        st = le.make_stego(img); ref = lsh.simhash(st, N)
        B.append(lsh.hamming(lsh.simhash(img, N), ref))                       # cover vs stego (embedding itself)
        for f in le.BENIGN.values():
            for t in range(TRIALS):
                B.append(lsh.hamming(ref, lsh.simhash(f(st, np.random.default_rng(t)), N)))
        for kind, f in le.TAMPER.items():
            for s in PATCHES:
                for t in range(TRIALS):
                    bad = f(st, np.random.default_rng(500 + t), s)
                    M.setdefault((kind, s), []).append(lsh.hamming(ref, lsh.simhash(bad, N)))
    return np.array(B), {k: np.array(v) for k, v in M.items()}


def pooled(M):
    return np.concatenate([v for (k, s), v in M.items() if s >= MIN_PATCH])


def tau_for_frr(B):
    taus = np.arange(0, N + 1)
    frr = np.array([(B > t).mean() for t in taus])
    return int(taus[np.argmax(frr <= FRR_TARGET)])


def evaluate(name, B, M, tau_cal, rows, sweep):
    Mp = pooled(M)
    t_own = tau_for_frr(B)
    r = dict(set=name, n_images=None, n_benign=len(B), tau_calibrated=tau_cal,
             FRR_at_tau_cal=round(float((B > tau_cal).mean()), 4), FAR_at_tau_cal=round(float((Mp <= tau_cal).mean()), 4),
             FRR_at_tau3=round(float((B > TAU_FIXED).mean()), 4), FAR_at_tau3=round(float((Mp <= TAU_FIXED).mean()), 4),
             tau_own_FRR1pct=t_own, AUC=round(le.auc(B, Mp), 4), benign_mean=round(float(B.mean()), 2), benign_max=int(B.max()))
    for (kind, s), v in sorted(M.items()):
        r[f"detect_{kind}_{s}px_at_tau_cal"] = round(float((v > tau_cal).mean()), 3)
    rows.append(r)
    for t in range(0, 9):
        sweep.append(dict(set=name, tau=t, FRR=round(float((B > t).mean()), 4), FAR_16px_and_up=round(float((Mp <= t).mean()), 4)))
    return r


def calibration_images():
    try:
        from dataset import load_dataset
        return load_dataset()
    except Exception as e:
        sys.exit(f"Could not load the calibration images via dataset.load_dataset(): {e}")


if __name__ == "__main__":
    folders = sys.argv[1:]
    if not folders:
        sys.exit(__doc__)
    cal = calibration_images()
    print(f"Calibration: {len(cal)} images ({', '.join(cal)})")
    Bc, Mc = collect(cal)
    tau_cal = tau_for_frr(Bc)
    print(f"tau chosen on calibration (smallest tau with FRR <= {FRR_TARGET:.0%}) = {tau_cal}   (pipeline uses tau = {TAU_FIXED})")
    rows, sweep = [], []
    r = evaluate("calibration(5 SIPI)", Bc, Mc, tau_cal, rows, sweep); r["n_images"] = len(cal)
    for fol in folders:
        imgs = load_folder(fol)
        if not imgs:
            print(f"!! no images found in {fol}"); continue
        print(f"Held-out set '{Path(fol).name}': {len(imgs)} images ...")
        B, M = collect(imgs)
        r = evaluate(Path(fol).name, B, M, tau_cal, rows, sweep); r["n_images"] = len(imgs)

    print(f"\n{'set':<22}{'imgs':<6}{'FRR@tau_cal':<13}{'FAR@tau_cal':<13}{'FRR@tau3':<10}{'FAR@tau3':<10}{'tau_own':<9}{'AUC'}")
    for r in rows:
        print(f"{r['set']:<22}{r['n_images']:<6}{r['FRR_at_tau_cal']:<13}{r['FAR_at_tau_cal']:<13}{r['FRR_at_tau3']:<10}"
              f"{r['FAR_at_tau3']:<10}{r['tau_own_FRR1pct']:<9}{r['AUC']}")
    print("\nDetection rate of tampering at tau_calibrated (higher is better):")
    keys = [k for k in rows[0] if k.startswith("detect_")]
    print(f"{'set':<22}" + "".join(f"{k.replace('detect_', '').replace('_at_tau_cal', ''):<18}" for k in keys))
    for r in rows:
        print(f"{r['set']:<22}" + "".join(f"{r[k]:<18}" for k in keys))
    print("\nHow to read it: tau generalises if the held-out FRR stays close to the calibration FRR (about 1-2%).")
    print("A high FAR on small patches (8-16 px) is expected: LSH is tolerant by design. Report it as the limit; SHA-256 covers it.")
    for name, data in (("heldout_summary.csv", rows), ("heldout_sweep.csv", sweep)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
        print(f"saved results/{name}")