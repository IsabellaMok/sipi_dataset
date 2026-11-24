"""ECC comparison for protecting the 256-bit SimHash sent over a noisy public channel.
Part A  residual-error test on random hashes: exact-recovery rate and residual BER per scheme and channel.
Part B  end-to-end authentication with REAL hash pairs (your images, benign vs tampered), threshold TAU:
        false-rejection rate (FRR) on benign changes and false-accept rate (FAR) on tampering,
        after the reference hash crossed the noisy channel.
Part C  SHA-256 contrast (extra experiment): exact match, no tolerance.
Run:  python ecc_experiment.py      Outputs: results/ecc_residual.csv, ecc_end_to_end.csv, sha256_extra.csv
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
import csv, hashlib
from pathlib import Path
import numpy as np
import ecc, lsh
from lsh_experiment import BENIGN, TAMPER, PATCH_SIZES, make_stego

OUT = Path("results"); OUT.mkdir(exist_ok=True)
TAU = 3                                     # threshold chosen in the LSH experiment (256-bit whole-image SimHash)
MIN_PATCH = 16
REPS = 3                                    # channel realisations per hash pair in Part B
CHANNELS = [("clean", None, 0), ("bsc", 0.005, 0), ("bsc", 0.01, 0), ("bsc", 0.02, 0), ("bsc", 0.05, 0), ("bsc", 0.10, 0),
            ("burst", None, 8), ("burst", None, 16), ("burst", None, 32)]
def ch_name(c): return "clean" if c[0] == "clean" else (f"BSC {c[1]*100:g}%" if c[0] == "bsc" else f"burst {c[2]}b")

def apply_channel(bits, ch, rng):
    if ch[0] == "clean": return bits
    return ecc.bsc(bits, ch[1], rng) if ch[0] == "bsc" else ecc.burst(bits, ch[2], rng)

def save(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"saved results/{name} ({len(rows)} rows)")

# ---------------- Part A ----------------
def part_a(schemes):
    rows = []
    for s in schemes:
        for ch in CHANNELS:
            rng = np.random.default_rng(7)
            h = rng.integers(0, 2, (s.n_trials, ecc.HASH_BITS)).astype(np.uint8)
            got = s.decode(apply_channel(s.encode(h), ch, rng))
            err = (got != h).sum(axis=1)
            rows.append(dict(scheme=s.name, tx_bits=s.tx_bits, overhead=round(s.tx_bits / ecc.HASH_BITS, 2), channel=ch_name(ch),
                             trials=s.n_trials, exact_recovery=round(float((err == 0).mean()), 4),
                             residual_BER=round(float(err.mean() / ecc.HASH_BITS), 5),
                             P_errors_gt_tau=round(float((err > TAU).mean()), 4)))
    return rows

# ---------------- hash pairs from real images ----------------
def build_pairs(images):
    ref_h, new_h, label, kind, sha_same = [], [], [], [], []
    for n, img in images.items():
        st = make_stego(img); ref = lsh.simhash(st, 256); ref_sha = hashlib.sha256(lsh.ll_subband(st).astype(np.int64).tobytes()).digest()
        def add(att, lab, k):
            ref_h.append(ref); new_h.append(lsh.simhash(att, 256)); label.append(lab); kind.append(k)
            sha_same.append(hashlib.sha256(lsh.ll_subband(att).astype(np.int64).tobytes()).digest() == ref_sha)
        for k, f in BENIGN.items():
            for t in range(20): add(f(st, np.random.default_rng(t)), "benign", k)
        for k, f in TAMPER.items():
            for s in PATCH_SIZES:
                if s >= MIN_PATCH:
                    for t in range(20): add(f(st, np.random.default_rng(500 + t), s), "malicious", f"{k}_{s}px")
    return np.array(ref_h), np.array(new_h), np.array(label), np.array(kind), np.array(sha_same)

# ---------------- Part B ----------------
def part_b(schemes, H, Hn, label):
    rows = []; ben = label == "benign"; mal = ~ben
    for s in schemes:
        for ch in CHANNELS:
            rng = np.random.default_rng(11)
            Hr, Hnr = np.repeat(H, REPS, axis=0), np.repeat(Hn, REPS, axis=0)
            got = s.decode(apply_channel(s.encode(Hr), ch, rng))
            d = (got != Hnr).sum(axis=1); lab = np.repeat(label, REPS)
            rows.append(dict(scheme=s.name, overhead=round(s.tx_bits / ecc.HASH_BITS, 2), channel=ch_name(ch), tau=TAU,
                             FRR_benign=round(float((d[lab == "benign"] > TAU).mean()), 4),
                             FAR_tamper=round(float((d[lab == "malicious"] <= TAU).mean()), 4)))
    return rows

# ---------------- Part C ----------------
def part_c(kind, label, sha_same, schemes):
    rows = []
    for k in dict.fromkeys(kind):
        m = kind == k
        rows.append(dict(condition=k, class_=label[m][0], n=int(m.sum()), sha256_digest_unchanged=round(float(sha_same[m].mean()), 4)))
    return rows

def table(rows, value, schemes, title):
    chans = [ch_name(c) for c in CHANNELS]
    print(f"\n{title}\n{'scheme':<24}{'ovh':<6}" + "".join(f"{c:<11}" for c in chans))
    for s in schemes:
        vals = {r["channel"]: r[value] for r in rows if r["scheme"] == s.name}
        print(f"{s.name:<24}{s.tx_bits/ecc.HASH_BITS:<6.2f}" + "".join(f"{vals[c]:<11}" for c in chans))

if __name__ == "__main__":
    from dataset import load_dataset
    schemes = ecc.all_schemes()
    print("Part A: residual errors on random hashes ..."); A = part_a(schemes); save("ecc_residual.csv", A)
    print("Building real hash pairs (benign / tampered) ...")
    H, Hn, label, kind, sha_same = build_pairs(load_dataset())
    print(f"  {len(H)} pairs: {int((label=='benign').sum())} benign, {int((label=='malicious').sum())} tampered")
    print("Part B: end-to-end authentication ..."); B = part_b(schemes, H, Hn, label); save("ecc_end_to_end.csv", B)
    C = part_c(kind, label, sha_same, schemes); save("sha256_extra.csv", C)

    table(A, "exact_recovery", schemes, "A1. Hash recovered EXACTLY (rate; higher is better)")
    table(A, "P_errors_gt_tau", schemes, f"A2. P(more than {TAU} residual bit errors) (lower is better)")
    table(B, "FRR_benign", schemes, f"B1. End-to-end false-rejection rate on benign changes, tau={TAU} (lower is better)")
    table(B, "FAR_tamper", schemes, f"B2. End-to-end false-accept rate on tampering >= {MIN_PATCH}px, tau={TAU} (lower is better)")
    print("\nC. SHA-256 of the LL sub-band: share of cases where the digest is UNCHANGED (SHA accepts only these)")
    for r in C: print(f"  {r['class_']:<10}{r['condition']:<22}n={r['n']:<5}unchanged={r['sha256_digest_unchanged']}")
    print("\nSHA-256 over the same channel: an image that arrives unchanged is accepted only if the digest is recovered EXACTLY ->")
    print("  FRR(unchanged image) = 1 - exact_recovery (table A1);  LSH with tau=3 tolerates up to 3 residual errors -> table A2.")