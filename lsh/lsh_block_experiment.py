"""Block-wise SimHash: one short SimHash per image region, reject if ANY region's distance exceeds tau.
Compares grids 1x1 (= whole image), 2x2, 4x4, 8x8 at the same total hash length, with two ways of
centring the features ('region' = subtract the region's own mean, 'global' = subtract the whole-image mean).
Uses the same benign distortions, tamper patches and random seeds as lsh_experiment.py.
Run:  python lsh_block_experiment.py        Outputs: results/lsh_block_distances.csv, results/lsh_block_summary.csv
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
import csv
from pathlib import Path
import numpy as np
import lsh
from lsh_experiment import BENIGN, make_stego, PATCH_SIZES, TRIALS, FRR_TARGET, MIN_PATCH_FOR_ROC, auc

OUT = Path("results"); OUT.mkdir(exist_ok=True)
GRIDS = (1, 2, 4, 8)
TOTAL_BITS = (256, 1024)
CENTRING = ("region", "global")
G_FEAT = 32                      # feature grid: 32x32 block means of the 256x256 LL sub-band (8x8 LL px each)

class RegionSimHash:
    def __init__(self, grid, total_bits, center, secret=1234):
        self.G, self.center = grid, center
        self.n_per = total_bits // (grid * grid)
        self.cells = G_FEAT // grid
        rng = np.random.default_rng(secret)
        self.W = [rng.normal(size=(self.n_per, self.cells ** 2)) for _ in range(grid * grid)]
    def __call__(self, f):                       # f: G_FEAT x G_FEAT feature grid (block means of LL)
        gm = f.mean(); out = []
        for idx in range(self.G ** 2):
            r, c = divmod(idx, self.G); k = self.cells
            sub = f[r*k:(r+1)*k, c*k:(c+1)*k].ravel()
            sub = sub - (sub.mean() if self.center == "region" else gm)
            out.append((self.W[idx] @ sub > 0).astype(np.uint8))
        return np.array(out)                     # (G*G, n_per)

def features(img):
    return lsh.block_means(lsh.ll_subband(img), G_FEAT)

# same RNG call order as lsh_experiment.py -> identical tamper instances; here we also return the position
def noise_patch(im, r, s):
    o = im.copy(); y, x = (int(v) for v in r.integers(0, 512 - s + 1, 2))
    o[y:y+s, x:x+s] = r.integers(0, 256, (s, s)); return o, (y, x)
def copy_move(im, r, s):
    o = im.copy(); y, x = (int(v) for v in r.integers(0, 512 - s + 1, 2))
    while True:
        y2, x2 = (int(v) for v in r.integers(0, 512 - s + 1, 2))
        if abs(y2 - y) >= s or abs(x2 - x) >= s: break
    o[y:y+s, x:x+s] = im[y2:y2+s, x2:x2+s]; return o, (y, x)
TAMPER = {"noise_patch": noise_patch, "copy_move": copy_move}

def configs():
    return [dict(grid=g, total=T, center=c, name=f"{g}x{g}_{T}b_{c}", hasher=RegionSimHash(g, T, c))
            for T in TOTAL_BITS for g in GRIDS for c in CENTRING]

def run(images):
    cfgs, rows = configs(), []
    stegos = {n: make_stego(im) for n, im in images.items()}
    for n, st in stegos.items():
        ref_f = features(st)
        refs = {c["name"]: c["hasher"](ref_f) for c in cfgs}
        def evaluate(att, cls, kind, param, pos):
            f = features(att)
            for c in cfgs:
                d = (c["hasher"](f) != refs[c["name"]]).sum(axis=1)       # distance per region
                rows.append(dict(config=c["name"], grid=c["grid"], total=c["total"], center=c["center"],
                                 n_per=c["hasher"].n_per, image=n, cls=cls, kind=kind, param=param,
                                 max_dist=int(d.max()), argmax=int(d.argmax()),
                                 py=pos[0] if pos else "", px=pos[1] if pos else ""))
        for kind, fn in BENIGN.items():
            for t in range(TRIALS):
                evaluate(fn(st, np.random.default_rng(t)), "benign", kind, "", None)
        for kind, fn in TAMPER.items():
            for s in PATCH_SIZES:
                for t in range(TRIALS):
                    att, pos = fn(st, np.random.default_rng(500 + t), s)
                    evaluate(att, "malicious", kind, s, pos)
    return rows

def localised(row, tau):
    """Was the region with the largest distance one that the tampered patch overlaps?"""
    G, s = row["grid"], row["param"]; size = 512 // G
    r, c = divmod(row["argmax"], G)
    return (row["py"] // size <= r <= (row["py"] + s - 1) // size) and (row["px"] // size <= c <= (row["px"] + s - 1) // size)

def summarise(rows):
    out = []
    for name in dict.fromkeys(r["config"] for r in rows):
        R = [r for r in rows if r["config"] == name]; n_per = R[0]["n_per"]
        B = np.array([r["max_dist"] for r in R if r["cls"] == "benign"])
        M = np.array([r["max_dist"] for r in R if r["cls"] == "malicious" and r["param"] >= MIN_PATCH_FOR_ROC])
        taus = np.arange(0, n_per + 1)
        frr = np.array([(B > t).mean() for t in taus]); far = np.array([(M <= t).mean() for t in taus])
        t = int(taus[np.argmax(frr <= FRR_TARGET)])
        row = dict(config=name, grid=R[0]["grid"], total_bits=R[0]["total"], bits_per_region=n_per, centring=R[0]["center"],
                   AUC=round(auc(B / n_per, M / n_per), 4), tau=t, FRR=round(float(frr[t]), 4), FAR=round(float(far[t]), 4))
        det_rows = [r for r in R if r["cls"] == "malicious" and r["max_dist"] > t]
        for s in PATCH_SIZES:
            allm = [r for r in R if r["cls"] == "malicious" and r["param"] == s]
            row[f"detect_{s}px"] = round(sum(r["max_dist"] > t for r in allm) / len(allm), 3)
        det_big = [r for r in det_rows if r["param"] >= MIN_PATCH_FOR_ROC]
        row["localisation_acc"] = round(float(np.mean([localised(r, t) for r in det_big])), 3) if det_big else ""
        out.append(row)
    return out

def save(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"saved results/{name} ({len(rows)} rows)")

def report(summ):
    print(f"\nOperating point: smallest tau with FRR <= {FRR_TARGET:.0%} (tau = max allowed bit differences in ANY one region)")
    print(f"{'grid':<6}{'bits':<6}{'b/reg':<7}{'centre':<8}{'AUC':<7}{'tau':<5}{'FRR':<7}{'FAR':<7}"
          f"{'8px':<7}{'16px':<7}{'32px':<7}{'64px':<7}{'locali.':<8}")
    for s in summ:
        print(f"{str(s['grid'])+'x'+str(s['grid']):<6}{s['total_bits']:<6}{s['bits_per_region']:<7}{s['centring']:<8}{s['AUC']:<7}"
              f"{s['tau']:<5}{s['FRR']:<7}{s['FAR']:<7}{s['detect_8px']:<7}{s['detect_16px']:<7}{s['detect_32px']:<7}"
              f"{s['detect_64px']:<7}{s['localisation_acc']!s:<8}")

if __name__ == "__main__":
    from dataset import load_dataset
    rows = run(load_dataset()); summ = summarise(rows)
    save("lsh_block_distances.csv", rows); save("lsh_block_summary.csv", summ); report(summ)