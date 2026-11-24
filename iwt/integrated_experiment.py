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
import csv, io, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import ecc, lsh, hamming_me as hm, metrics as mt
from iwt import iwt2, iiwt2

OUT = Path("results"); OUT.mkdir(exist_ok=True)
TAU = 3
TRIALS = 10

# ---------- embedding with side information / cover recovery ----------
def embed_with_side_info(img, bits):
    bits = np.asarray(bits, dtype=np.int64)
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 9, dtype=np.int64)])
    LL, HL, LH, HH = iwt2(img)
    subs = [HL.copy(), LH.copy(), HH.copy()]
    pos = hm._blocks(HL); side = []
    for t in range(len(bits) // 9):
        i, j0 = pos[t]
        for b, sub in enumerate(subs):
            blk = sub[i:i+2, j0:j0+2]
            before = hm.block_to_x(blk)
            new = hm.embed_block(blk, bits[9*t + 3*b: 9*t + 3*b + 3])
            diff = np.flatnonzero(before != hm.block_to_x(new))
            side.append(int(diff[0]) + 1 if len(diff) else 0)
            sub[i:i+2, j0:j0+2] = new
    return np.clip(iiwt2(LL, *subs), 0, 255).astype(np.uint8), side

def recover_cover(received, side):
    LL, HL, LH, HH = iwt2(received)
    subs = [HL.copy(), LH.copy(), HH.copy()]
    pos = hm._blocks(HL); n = 0
    for t in range(len(side) // 3):
        i, j0 = pos[t]
        for sub in subs:
            if side[n]:
                blk = sub[i:i+2, j0:j0+2]
                x = hm.block_to_x(blk); x[side[n] - 1] ^= 1
                sub[i:i+2, j0:j0+2] = hm.x_to_block(x, blk)
            n += 1
    return np.clip(iiwt2(LL, *subs), 0, 255).astype(np.uint8)

# ---------- channel conditions ----------
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
def noise_patch(s):
    def f(im, r):
        o = im.copy(); y, x = (int(v) for v in r.integers(0, 512 - s + 1, 2)); o[y:y+s, x:x+s] = r.integers(0, 256, (s, s)); return o
    return f
def copy_move(s):
    def f(im, r):
        o = im.copy(); y, x = (int(v) for v in r.integers(0, 512 - s + 1, 2))
        while True:
            y2, x2 = (int(v) for v in r.integers(0, 512 - s + 1, 2))
            if abs(y2 - y) >= s or abs(x2 - x) >= s: break
        o[y:y+s, x:x+s] = im[y2:y2+s, x2:x2+s]; return o
    return f

CONDITIONS = [("clean", "benign", lambda im, r: im),
              ("salt_pepper_0.05%", "benign", sp(0.0005)), ("salt_pepper_0.1%", "benign", sp(0.001)),
              ("salt_pepper_0.5%", "benign", sp(0.005)), ("salt_pepper_1%", "benign", sp(0.01)),
              ("lsb_flip_0.1%", "benign", lsbflip(0.001)), ("lsb_flip_1%", "benign", lsbflip(0.01)),
              ("gaussian_s1", "benign", gauss(1)), ("jpeg_q90", "benign", jpeg(90)),
              ("noise_patch_32px", "malicious", noise_patch(32)), ("copy_move_32px", "malicious", copy_move(32)),
              ("noise_patch_64px", "malicious", noise_patch(64)), ("copy_move_64px", "malicious", copy_move(64))]

def schemes():
    return [ecc.NoECC(), ecc.Repeat(), ecc.Cyclic("BCH(15,7) t=2", 15, 0b111010001, 2),
            ecc.Repeat(ecc.Cyclic("BCH(31,21) t=2", 31, 0b11101101001, 2)), ecc.RS(4)]

def sha_ll(img):
    return hashlib.sha256(lsh.ll_subband(img).astype(np.int64).tobytes()).digest()

# ---------- experiment ----------
def run(images, trials=TRIALS):
    rows = []
    for iname, cover in images.items():
        ref_h = lsh.simhash(cover, 256)
        for sc in schemes():
            payload = sc.encode(ref_h[None, :].astype(np.uint8))[0]
            stego, side = embed_with_side_info(cover, payload)
            assert np.array_equal(lsh.simhash(stego, 256), ref_h), "embedding changed the LL hash"
            ref_sha = sha_ll(stego)
            for ci, (cname, cls, fn) in enumerate(CONDITIONS):
                for t in range(trials):
                    seed = 1000 * ci + t
                    ns, nc = fn(stego, np.random.default_rng(seed)), fn(cover, np.random.default_rng(seed))
                    ext = hm.extract_share(ns, sc.tx_bits)
                    got = sc.decode(ext[None, :].astype(np.uint8))[0]
                    h_new = lsh.simhash(ns, 256)
                    rec = recover_cover(ns, side)
                    rows.append(dict(image=iname, scheme=sc.name, tx_bits=sc.tx_bits, condition=cname, cls=cls, trial=t,
                        payload_BER=round(float(np.mean(ext != payload)), 5),
                        hash_exact=int(np.array_equal(got, ref_h)), residual_errors=int((got != ref_h).sum()),
                        d_decoded=lsh.hamming(got, h_new), d_if_perfect=lsh.hamming(ref_h, h_new),
                        LSH_accept=int(lsh.hamming(got, h_new) <= TAU), SHA_accept=int(sha_ll(ns) == ref_sha),
                        MSE_received=round(mt.mse(ns, cover), 6), MSE_noise_floor=round(mt.mse(nc, cover), 6),
                        MSE_recovered=round(mt.mse(rec, cover), 6), MSE_rec_vs_noisycover=round(mt.mse(rec, nc), 6),
                        exact_cover=int(np.array_equal(rec, cover))))
    return rows

def save(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"saved results/{name} ({len(rows)} rows)")

def summarise(rows):
    out = []
    for sc in dict.fromkeys(r["scheme"] for r in rows):
        for cn, cls, _ in CONDITIONS:
            R = [r for r in rows if r["scheme"] == sc and r["condition"] == cn]
            m = lambda k: round(float(np.mean([r[k] for r in R])), 5)
            out.append(dict(scheme=sc, tx_bits=R[0]["tx_bits"], condition=cn, cls=cls, n=len(R), payload_BER=m("payload_BER"),
                            hash_exact=m("hash_exact"), mean_residual_errors=m("residual_errors"), LSH_accept=m("LSH_accept"),
                            SHA_accept=m("SHA_accept"), accept_if_hash_perfect=round(float(np.mean([r["d_if_perfect"] <= TAU for r in R])), 5),
                            MSE_received=m("MSE_received"), MSE_noise_floor=m("MSE_noise_floor"), MSE_recovered=m("MSE_recovered"),
                            MSE_rec_vs_noisycover=m("MSE_rec_vs_noisycover"), exact_cover=m("exact_cover")))
    return out

def table(summ, key, conds, title):
    names = list(dict.fromkeys(s["scheme"] for s in summ))
    print(f"\n{title}\n{'scheme':<24}" + "".join(f"{c[:15]:<16}" for c in conds))
    for sc in names:
        v = {s["condition"]: s[key] for s in summ if s["scheme"] == sc}
        print(f"{sc:<24}" + "".join(f"{v[c]:<16}" for c in conds))

def report(summ):
    ben = [c for c, k, _ in CONDITIONS if k == "benign"]; mal = [c for c, k, _ in CONDITIONS if k == "malicious"]
    table(summ, "payload_BER", ben, "1. Raw bit error rate of the embedded payload (before ECC)")
    table(summ, "hash_exact", ben, "2. Embedded hash recovered EXACTLY after ECC (rate)")
    table(summ, "LSH_accept", ben, f"3. LSH accepts the genuine noisy image, tau={TAU} (rate; higher is better)")
    table(summ, "LSH_accept", mal, f"4. LSH accepts a TAMPERED image = false accept (rate; lower is better)")
    one = [s for s in summ if s["scheme"] == summ[0]["scheme"]]
    print("\n5. SHA-256 vs LSH on every condition: how often SHA-256 accepts, vs how often LSH would accept if the hash arrived perfectly")
    print(f"{'condition':<22}{'SHA-256 accepts':<18}{'LSH accepts (ideal hash)':<26}")
    for s in one: print(f"{s['condition']:<22}{s['SHA_accept']:<18}{s['accept_if_hash_perfect']:<26}")
    print("\n6. Cover recovery (mean over all schemes). Noise floor = MSE of the noisy cover alone: no scheme can beat it.")
    print(f"{'condition':<22}{'MSE received':<14}{'noise floor':<14}{'recovered':<14}{'rec vs noisy cover':<20}{'exact cover':<12}")
    for cn, _, _ in CONDITIONS:
        R = [s for s in summ if s["condition"] == cn]; a = lambda k: np.mean([r[k] for r in R])
        print(f"{cn:<22}{a('MSE_received'):<14.4f}{a('MSE_noise_floor'):<14.4f}{a('MSE_recovered'):<14.4f}{a('MSE_rec_vs_noisycover'):<20.5f}{a('exact_cover'):<12.2f}")

if __name__ == "__main__":
    from dataset import load_dataset
    rows = run(load_dataset()); summ = summarise(rows)
    save("integrated_runs.csv", rows); save("integrated_summary.csv", summ); report(summ)