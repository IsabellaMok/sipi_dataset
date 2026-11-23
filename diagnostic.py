import csv, io
from pathlib import Path
import numpy as np
from PIL import Image
import cbss, hamming_me as hm, metrics as mt
from iwt import iwt2, iiwt2
from test_baseline import run as clean_run

OUT = Path("results"); OUT.mkdir(exist_ok=True)
K = 3                          # shares needed to rebuild the key
KEY_SIZES = (64, 256, 512, 1024, 3072)
ATTACK_KEYS = (64, 256)        # key sizes used in tamper / attack tests
TRIALS = 5                     # repeats for random attacks

def make_key(L, seed=0):
    return np.random.default_rng(seed).integers(0, 2, L).astype(np.uint8)

def save_csv(name, rows):
    if not rows: return
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  saved results/{name} ({len(rows)} rows)")

# ---------- embedding that also records side information (for exact cover recovery) ----------
def embed_with_side_info(img, share_bits):
    bits = np.asarray(share_bits, dtype=np.int64)
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 9, dtype=np.int64)])
    LL, HL, LH, HH = iwt2(img)
    subs = [HL.copy(), LH.copy(), HH.copy()]
    pos = hm._blocks(HL)
    side = []                                   # flipped position (0 = none) per (block, subband)
    for t in range(len(bits) // 9):
        i, j0 = pos[t]
        for b, sub in enumerate(subs):
            blk = sub[i:i+2, j0:j0+2]
            before = hm.block_to_x(blk)
            new = hm.embed_block(blk, bits[9*t + 3*b: 9*t + 3*b + 3])
            diff = np.flatnonzero(before != hm.block_to_x(new))
            side.append(int(diff[0]) + 1 if len(diff) else 0)
            sub[i:i+2, j0:j0+2] = new
    raw = iiwt2(LL, *subs)
    return np.clip(raw, 0, 255).astype(np.uint8), side

def recover_cover(stego, side):
    LL, HL, LH, HH = iwt2(stego)
    subs = [HL.copy(), LH.copy(), HH.copy()]
    pos = hm._blocks(HL)
    n = 0
    for t in range(len(side) // 3):
        i, j0 = pos[t]
        for sub in subs:
            if side[n]:
                blk = sub[i:i+2, j0:j0+2]
                x = hm.block_to_x(blk); x[side[n] - 1] ^= 1
                sub[i:i+2, j0:j0+2] = hm.x_to_block(x, blk)
            n += 1
    return np.clip(iiwt2(LL, *subs), 0, 255).astype(np.uint8)

# ---------- 1. cover recovery ----------
def cover_recovery(images):
    rows = []
    for name, img in images.items():
        for L in KEY_SIZES:
            share = cbss.gen_shares(make_key(L), 1)[0]
            stego, side = embed_with_side_info(img, share)
            rec = recover_cover(stego, side)
            rows.append(dict(image=name, key_bits=L,
                MSE_no_sideinfo=round(mt.mse(img, stego), 8),
                PSNR_no_sideinfo=round(mt.psnr(img, stego), 2),
                SSIM=round(mt.ssim(img, stego), 6),
                pixels_changed=int(np.sum(img != stego)),
                payload_bits=L, sideinfo_bits=3 * len(side),
                MSE_with_sideinfo=round(mt.mse(img, rec), 8),
                exact_recovery=bool(np.array_equal(img, rec))))
    return rows

# ---------- helper: embed K shares, attack, extract, reconstruct ----------
def setup(img, L):
    sk = make_key(L)
    shares = cbss.gen_shares(sk, 1)[:K]
    stegos = [hm.embed_share(img, s)[0] for s in shares]
    return sk, shares, stegos

def receive(sk, shares, attacked):
    got = [hm.extract_share(a, len(sk)) for a in attacked]
    share_ber = float(np.mean([mt.ber(s, g) for s, g in zip(shares, got)]))
    rec, _ = cbss.reconstruct(got, K)
    return share_ber, mt.ber(sk, rec), bool(np.array_equal(sk, rec))

# ---------- 2. tamper test ----------
def tamper(images):
    rows = []
    for name, img in images.items():
        for L in ATTACK_KEYS:
            sk, shares, stegos = setup(img, L)
            pieces = (L + 8) // 9
            regions = {"small_4x8_in_payload": (0, 4, 0, 8),
                       "strip_in_payload":     (0, 8, 0, min(512, 4 * pieces + 4)),
                       "block_32x32_outside":  (240, 272, 240, 272)}
            for rname, (r0, r1, c0, c1) in regions.items():
                sb, kb, kc, ch = [], [], [], []
                for t in range(TRIALS):
                    rng = np.random.default_rng(100 + t)
                    bad = stegos[0].copy()
                    bad[r0:r1, c0:c1] = rng.integers(0, 256, (r1 - r0, c1 - c0))
                    s_ber, k_ber, ok = receive(sk, shares, [bad] + stegos[1:])
                    sb.append(s_ber); kb.append(k_ber); kc.append(ok)
                    ch.append(int(np.sum(bad != stegos[0])))
                rows.append(dict(image=name, key_bits=L, region=rname,
                    pixels_changed=int(np.mean(ch)), share_BER=round(float(np.mean(sb)), 4),
                    key_BER=round(float(np.mean(kb)), 4), key_correct_rate=round(float(np.mean(kc)), 2),
                    flagged_by_baseline=False))   # the baseline has NO verification step, so it can never flag
    return rows

# ---------- 3. attacks ----------
def gauss(sigma):
    return lambda im, rng: np.clip(im + np.rint(rng.normal(0, sigma, im.shape)), 0, 255).astype(np.uint8)
def salt_pepper(d):
    def f(im, rng):
        out = im.copy(); m = rng.random(im.shape)
        out[m < d / 2] = 0; out[(m >= d / 2) & (m < d)] = 255
        return out
    return f
def bitflip(p):
    return lambda im, rng: (im ^ (rng.random(im.shape) < p).astype(np.uint8)).astype(np.uint8)
def crop_rows(n):                      # top rows cropped away and padded with black
    def f(im, rng):
        out = im.copy(); out[:n, :] = 0; return out
    return f
def crop_bottom_half(im, rng):
    out = im.copy(); out[256:, :] = 0; return out
def jpeg(q):
    def f(im, rng):
        buf = io.BytesIO(); Image.fromarray(im).save(buf, format="JPEG", quality=q)
        return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("L"))
    return f

ATTACKS = {"none": lambda im, rng: im,
           "gaussian_s1": gauss(1), "gaussian_s2": gauss(2), "gaussian_s5": gauss(5),
           "salt_pepper_0.1%": salt_pepper(0.001), "salt_pepper_1%": salt_pepper(0.01),
           "lsb_flip_0.1%": bitflip(0.001), "lsb_flip_1%": bitflip(0.01),
           "crop_top_16_rows": crop_rows(16), "crop_bottom_half": crop_bottom_half,
           "jpeg_q90": jpeg(90), "jpeg_q75": jpeg(75)}

def attacks(images):
    rows = []
    for name, img in images.items():
        for L in ATTACK_KEYS:
            sk, shares, stegos = setup(img, L)
            for aname, f in ATTACKS.items():
                sb, kb, kc, ps = [], [], [], []
                for t in range(TRIALS):
                    rng = np.random.default_rng(1000 + t)
                    att = [f(s, rng) for s in stegos]
                    s_ber, k_ber, ok = receive(sk, shares, att)
                    sb.append(s_ber); kb.append(k_ber); kc.append(ok)
                    ps.append(np.mean([min(mt.psnr(s, a), 99.0) for s, a in zip(stegos, att)]))
                rows.append(dict(image=name, key_bits=L, attack=aname,
                    attacked_PSNR=round(float(np.mean(ps)), 2), share_BER=round(float(np.mean(sb)), 4),
                    key_BER=round(float(np.mean(kb)), 4), key_correct_rate=round(float(np.mean(kc)), 2)))
    return rows

# ---------- summary ----------
def summarise(rows, group_key, cols):
    groups = {}
    for r in rows:
        groups.setdefault((r["key_bits"], r[group_key]), []).append(r)
    print(f"\n{group_key:<24}{'key':<6}" + "".join(f"{c:<18}" for c in cols))
    for (L, g), rs in groups.items():
        print(f"{g:<24}{L:<6}" + "".join(f"{np.mean([float(r[c]) for r in rs]):<18.4f}" for c in cols))

if __name__ == "__main__":
    try:
        from dataset import load_dataset
        images = load_dataset()
    except Exception as e:
        print("Could not load dataset (", e, ") -> synthetic image")
        rng = np.random.default_rng(0)
        images = {"synthetic": np.clip(128 + 60 * np.sin(np.add.outer(np.arange(512), np.arange(512)) / 40)
                                       + rng.normal(0, 4, (512, 512)), 0, 255).astype(np.uint8)}

    print("1/4 clean baseline ..."); save_csv("baseline_clean.csv",
        [dict(image=n, key_bits=L, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in clean_run(im, L).items()})
         for n, im in images.items() for L in KEY_SIZES])
    print("2/4 cover recovery ..."); rec = cover_recovery(images); save_csv("cover_recovery.csv", rec)
    print("3/4 tamper test ...");    tam = tamper(images);          save_csv("tamper.csv", tam)
    print("4/4 attacks ...");        att = attacks(images);         save_csv("attacks.csv", att)

    print("\n=== Cover recovery (mean over images) ===")
    for L in KEY_SIZES:
        rs = [r for r in rec if r["key_bits"] == L]
        print(f"key {L:<5} MSE(no side info)={np.mean([r['MSE_no_sideinfo'] for r in rs]):.2e}  "
              f"MSE(with side info)={np.mean([r['MSE_with_sideinfo'] for r in rs]):.1f}  "
              f"exact recovery in {sum(r['exact_recovery'] for r in rs)}/{len(rs)} images  "
              f"side info = {rs[0]['sideinfo_bits']} bits for {L} payload bits")
    summarise(tam, "region", ["pixels_changed", "key_BER", "key_correct_rate"])
    summarise(att, "attack", ["attacked_PSNR", "share_BER", "key_BER", "key_correct_rate"])
    print("\nDone. CSV files are in the results/ folder.")