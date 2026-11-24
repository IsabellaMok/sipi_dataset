"""LSH authentication experiment: compare SimHash / pHash / MinHash and choose a Hamming-distance threshold.

Authentication rule:  accept the received stego image if  Hamming(hash_ref, hash_received) <= tau.
 - benign   = what the public channel may legitimately do (embedding, light noise, small bit errors, mild JPEG)
 - malicious = tampering (patch replaced with noise / copy-move) and a different image (impostor)
Outputs: results/lsh_distances.csv, results/lsh_summary.csv   (run:  python lsh_experiment.py)
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
import csv, io
from pathlib import Path
import numpy as np
from PIL import Image
import lsh, hamming_me as hm, cbss

OUT = Path("results"); OUT.mkdir(exist_ok=True)
N_BITS = 256
TRIALS = 20                 # random repeats per condition
PATCH_SIZES = (8, 16, 32, 64)
MIN_PATCH_FOR_ROC = 16      # patches smaller than this are reported separately as the sensitivity limit
FRR_TARGET = 0.01

# ---------- distortions ----------
def gauss(s):   return lambda im, r: np.clip(im + np.rint(r.normal(0, s, im.shape)), 0, 255).astype(np.uint8)
def sp(d):
    def f(im, r):
        o = im.copy(); m = r.random(im.shape); o[m < d / 2] = 0; o[(m >= d / 2) & (m < d)] = 255; return o
    return f
def lsbflip(p): return lambda im, r: (im ^ (r.random(im.shape) < p).astype(np.uint8)).astype(np.uint8)
def jpeg(q):
    def f(im, r):
        b = io.BytesIO(); Image.fromarray(im).save(b, format="JPEG", quality=q)
        return np.array(Image.open(io.BytesIO(b.getvalue())).convert("L"))
    return f

BENIGN = {"gaussian_s1": gauss(1), "gaussian_s2": gauss(2), "salt_pepper_0.1%": sp(0.001),
          "salt_pepper_0.5%": sp(0.005), "lsb_flip_1%": lsbflip(0.01), "jpeg_q90": jpeg(90)}

def noise_patch(im, r, s):
    o = im.copy(); y, x = r.integers(0, 512 - s + 1, 2)
    o[y:y+s, x:x+s] = r.integers(0, 256, (s, s)); return o

def copy_move(im, r, s):
    o = im.copy(); y, x = r.integers(0, 512 - s + 1, 2)
    while True:
        y2, x2 = r.integers(0, 512 - s + 1, 2)
        if abs(y2 - y) >= s or abs(x2 - x) >= s: break
    o[y:y+s, x:x+s] = im[y2:y2+s, x2:x2+s]; return o

TAMPER = {"noise_patch": noise_patch, "copy_move": copy_move}

# ---------- experiment ----------
def make_stego(img, key_bits=256):
    sk = np.random.default_rng(0).integers(0, 2, key_bits).astype(np.uint8)
    return hm.embed_share(img, cbss.gen_shares(sk, 1)[0])[0]

def run(images):
    rows, stegos = [], {n: make_stego(im) for n, im in images.items()}
    names = list(images)
    for hname, hf in lsh.HASHES.items():
        for n in names:
            cover, st = images[n], stegos[n]
            ref = hf(st, N_BITS)
            def add(cls, kind, param, d, pred=""):
                rows.append(dict(hash=hname, image=n, cls=cls, kind=kind, param=param, distance=d, predicted=pred))
            add("benign", "embedding", "256-bit share", lsh.hamming(hf(cover, N_BITS), ref))   # cover vs stego
            for kind, f in BENIGN.items():
                for t in range(TRIALS):
                    add("benign", kind, "", lsh.hamming(ref, hf(f(st, np.random.default_rng(t)), N_BITS)))
            for kind, f in TAMPER.items():
                for s in PATCH_SIZES:
                    for t in range(TRIALS):
                        bad = f(st, np.random.default_rng(500 + t), s)
                        pred = round(lsh.simhash_predicted_distance(st, bad, N_BITS), 2) if hname == "SimHash" else ""
                        add("malicious", kind, s, lsh.hamming(ref, hf(bad, N_BITS)), pred)
        for i, a in enumerate(names):                         # impostor: two different images
            for b in names[i + 1:]:
                rows.append(dict(hash=hname, image=f"{a}|{b}", cls="impostor", kind="different_image", param="",
                                 distance=lsh.hamming(hf(stegos[a], N_BITS), hf(stegos[b], N_BITS)), predicted=""))
    return rows

def auc(ben, mal):
    ben, mal = np.asarray(ben), np.asarray(mal)
    return float(np.mean([(m > ben).mean() + 0.5 * (m == ben).mean() for m in mal]))

def summarise(rows):
    out = []
    for hname in lsh.HASHES:
        R = [r for r in rows if r["hash"] == hname]
        B = np.array([r["distance"] for r in R if r["cls"] == "benign"])
        M = np.array([r["distance"] for r in R if r["cls"] == "malicious" and r["param"] >= MIN_PATCH_FOR_ROC])
        I = np.array([r["distance"] for r in R if r["cls"] == "impostor"])
        taus = np.arange(0, N_BITS + 1)
        frr = np.array([(B > t).mean() for t in taus]); far = np.array([(M <= t).mean() for t in taus])
        t_eer = int(taus[np.argmin(np.abs(far - frr))])
        t_frr = int(taus[np.argmax(frr <= FRR_TARGET)])        # smallest tau with FRR <= target
        t_zero = int(B.max())                                  # no benign sample rejected
        row = dict(hash=hname, n_bits=N_BITS, benign_mean=round(B.mean(), 2), benign_max=int(B.max()),
                   tamper_mean=round(M.mean(), 2), impostor_min=int(I.min()), AUC=round(auc(B, M), 4),
                   tau_EER=t_eer, EER=round(float((far[t_eer] + frr[t_eer]) / 2), 4),
                   tau_FRR1pct=t_frr, FAR_at_tau_FRR1pct=round(float(far[t_frr]), 4),
                   tau_zero_FRR=t_zero, FAR_at_tau_zero_FRR=round(float(far[t_zero]), 4))
        for kind in TAMPER:                                    # detection rate per patch size at tau_FRR1pct
            for s in PATCH_SIZES:
                d = [r["distance"] for r in R if r["cls"] == "malicious" and r["kind"] == kind and r["param"] == s]
                row[f"detect_{kind}_{s}px"] = round(float(np.mean(np.array(d) > t_frr)), 3)
        out.append(row)
    return out

def save(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"saved results/{name} ({len(rows)} rows)")

def report(rows, summ):
    print("\n=== Distance by condition (mean Hamming distance out of", N_BITS, "bits, all images) ===")
    print(f"{'kind':<20}{'param':<8}" + "".join(f"{h:<10}" for h in lsh.HASHES))
    keys = sorted({(r["cls"], r["kind"], r["param"]) for r in rows if r["cls"] != "impostor"}, key=lambda k: (k[0] != "benign", k[1], str(k[2])))
    for cls, kind, p in keys:
        vals = [np.mean([r["distance"] for r in rows if r["hash"] == h and r["cls"] == cls and r["kind"] == kind and r["param"] == p]) for h in lsh.HASHES]
        print(f"{cls[:3]+':'+kind:<20}{str(p):<8}" + "".join(f"{v:<10.1f}" for v in vals))
    print("\n=== Threshold selection ===")
    for s in summ:
        print(f"\n{s['hash']}: AUC={s['AUC']}  EER={s['EER']} at tau={s['tau_EER']}")
        print(f"  benign mean/max = {s['benign_mean']}/{s['benign_max']}   tamper(>= {MIN_PATCH_FOR_ROC}px) mean = {s['tamper_mean']}   impostor min = {s['impostor_min']}")
        print(f"  tau(FRR<=1%) = {s['tau_FRR1pct']} -> FAR = {s['FAR_at_tau_FRR1pct']}     tau(no false reject) = {s['tau_zero_FRR']} -> FAR = {s['FAR_at_tau_zero_FRR']}")
        print("  detection rate at tau(FRR<=1%):", {k.replace('detect_', ''): v for k, v in s.items() if k.startswith('detect_')})
    sim = [(float(r["predicted"]), r["distance"]) for r in rows if r["hash"] == "SimHash" and r["predicted"] != ""]
    if sim:
        p, m = np.array(sim).T
        print(f"\nSimHash theory check (n*theta/pi vs measured): correlation = {np.corrcoef(p, m)[0,1]:.3f}, mean predicted = {p.mean():.2f}, mean measured = {m.mean():.2f}")

if __name__ == "__main__":
    from dataset import load_dataset
    images = load_dataset()
    rows = run(images); summ = summarise(rows)
    save("lsh_distances.csv", rows); save("lsh_summary.csv", summ); report(rows, summ)