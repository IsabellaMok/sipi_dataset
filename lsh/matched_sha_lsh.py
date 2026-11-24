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
import csv, hashlib, sys
import numpy as np
import ecc, lsh
import full_pipeline as fp
from full_pipeline import embed_payload, extract_payload, ecc_encode, ecc_decode, POS_SECRET, TAU
from integrated_experiment import sp, lsbflip, gauss, jpeg, noise_patch, copy_move
from median_experiment import median

N = 256
GENUINE = {"clean": lambda im, r: im, "salt_pepper_0.05%": sp(0.0005), "salt_pepper_0.1%": sp(0.001),
           "salt_pepper_0.5%": sp(0.005), "lsb_flip_0.1%": lsbflip(0.001), "gaussian_s1": gauss(1),
           "jpeg_q90": jpeg(90), "median_3x3": median(3)}
MALICIOUS = {"noise_patch_32px": noise_patch(32), "noise_patch_64px": noise_patch(64),
             "copy_move_32px": copy_move(32), "copy_move_64px": copy_move(64)}
PAYLOAD_ONLY = {"payload_BSC_0.5%": 0.005, "payload_BSC_1%": 0.01, "payload_BSC_2%": 0.02, "payload_BSC_5%": 0.05}


def sha_ll_bits(img):
    d = hashlib.sha256(lsh.ll_subband(img).astype(np.int64).tobytes()).digest()
    return np.unpackbits(np.frombuffer(d, dtype=np.uint8))


def schemes():
    return fp.schemes()          # none, rep3, rep3+BCH(31,21)


def decide(info_hat, ref_now, kind):
    """kind 'sha' -> exact equality; kind 'lsh' -> Hamming <= tau."""
    return bool(np.array_equal(info_hat, ref_now)) if kind == "sha" else lsh.hamming(info_hat, ref_now) <= TAU


def run(images, trials):
    rows = []
    for iname, cover in images.items():
        h_sha, h_lsh = sha_ll_bits(cover), lsh.simhash(cover, N)
        for sch in schemes():
            enc = {k: ecc_encode(sch, h) for k, h in (("sha", h_sha), ("lsh", h_lsh))}
            st = {k: embed_payload(cover, enc[k], POS_SECRET)[0:4:3] for k in enc}      # (stego, order)
            for cls, conds in (("genuine", GENUINE), ("malicious", MALICIOUS)):
                for cname, fn in conds.items():
                    for t in range(trials):
                        res = {}
                        for k in ("sha", "lsh"):
                            stego, order = st[k]
                            recv = fn(stego, np.random.default_rng(7000 + t))          # same noise seed for both codes
                            rx = extract_payload(recv, len(enc[k]), order)
                            hat = ecc_decode(sch, rx, N)
                            now = sha_ll_bits(recv) if k == "sha" else lsh.simhash(recv, N)
                            res[k] = decide(hat, now, k)
                        rows.append(dict(image=iname, scheme=sch.name, overhead=round(sch.tx_bits / N, 2), cls=cls,
                                         condition=cname, trial=t, sha_accept=int(res["sha"]), lsh_accept=int(res["lsh"]),
                                         dual_exact=int(res["sha"]), dual_tolerated=int(res["lsh"] and not res["sha"]),
                                         dual_rejected=int(not (res["sha"] or res["lsh"]))))
            for cname, p in PAYLOAD_ONLY.items():
                for t in range(trials):
                    res = {}
                    for k, ref in (("sha", h_sha), ("lsh", h_lsh)):
                        rx = ecc.bsc(enc[k].reshape(1, -1), p, np.random.default_rng(9000 + t)).reshape(-1)
                        res[k] = decide(ecc_decode(sch, rx, N), ref, k)               # image intact: recomputed hash == ref
                    rows.append(dict(image=iname, scheme=sch.name, overhead=round(sch.tx_bits / N, 2), cls="payload_only",
                                     condition=cname, trial=t, sha_accept=int(res["sha"]), lsh_accept=int(res["lsh"]),
                                     dual_exact=int(res["sha"]), dual_tolerated=int(res["lsh"] and not res["sha"]),
                                     dual_rejected=int(not (res["sha"] or res["lsh"]))))
    return rows


def report(rows):
    sch = list(dict.fromkeys(r["scheme"] for r in rows))
    conds = list(dict.fromkeys((r["cls"], r["condition"]) for r in rows))
    mean = lambda cls, c, s, k: np.mean([r[k] for r in rows if r["cls"] == cls and r["condition"] == c and r["scheme"] == s])
    print("\nAccept rate, SHA-256 / LSH.  genuine & payload_only: higher = better.  malicious: lower = better (false accept)")
    print(f"{'kind':<13}{'condition':<20}" + "".join(f"{s[:20]:<24}" for s in sch))
    for cls, c in conds:
        print(f"{cls:<13}{c:<20}" + "".join(f"{mean(cls, c, s, 'sha_accept'):.2f} / {mean(cls, c, s, 'lsh_accept'):.2f}".ljust(24) for s in sch))
    print("\nSummary per ECC scheme (mean over conditions of each kind)")
    print(f"{'scheme':<24}{'genuine: SHA':<14}{'genuine: LSH':<14}{'payload-only: SHA':<19}{'payload-only: LSH':<19}{'malicious: SHA':<16}{'malicious: LSH'}")
    for s in sch:
        g = lambda cls, k: np.mean([r[k] for r in rows if r["cls"] == cls and r["scheme"] == s])
        print(f"{s:<24}{g('genuine', 'sha_accept'):<14.3f}{g('genuine', 'lsh_accept'):<14.3f}{g('payload_only', 'sha_accept'):<19.3f}"
              f"{g('payload_only', 'lsh_accept'):<19.3f}{g('malicious', 'sha_accept'):<16.3f}{g('malicious', 'lsh_accept'):.3f}")
    print("\nDual rule (EXACT / TOLERATED / REJECTED share), genuine + malicious, scheme: " + sch[-1])
    for cls in ("genuine", "malicious"):
        R = [r for r in rows if r["cls"] == cls and r["scheme"] == sch[-1]]
        print(f"  {cls:<10}" + "  ".join(f"{k.replace('dual_', '')} {np.mean([r[k] for r in R]):.2f}" for k in ("dual_exact", "dual_tolerated", "dual_rejected")))


if __name__ == "__main__":
    trials = 3 if (len(sys.argv) > 1 and sys.argv[1] == "quick") else 10
    images = fp.load_images()
    images = {n: images[n] for n in list(images)[:5]}
    rows = run(images, trials)
    fp.save_csv("matched_sha_lsh.csv", rows)
    report(rows)