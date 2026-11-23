"""
full_pipeline.py - end-to-end pipeline for
"A Multi-Layered Reversible Data Hiding Framework Integrating IWT and Dual Authentication Validation"

THE FIVE LAYERS
  L1 Secret sharing       CBSS splits the secret key into shares; any k shares rebuild it.
  L2 Embedding            Integer 5/3 IWT + Hamming(7,4) matrix embedding into HL/LH/HH.
                          Block positions are chosen by a secret key (keyed permutation).
  L3 Error protection     ECC over the whole payload (none / repetition x3 / repetition x3 + BCH(31,21)).
  L4 Dual authentication  Layer A = SHA-256 of the original cover (exact).  Layer B = keyed SimHash (LSH) of the
                          LL sub-band (tolerant).  Decision per share: EXACT / TOLERATED / REJECTED.
  L5 Recovery             Rebuild the key from k usable shares. Undo the embedding with side information
                          to restore the cover (exact on a clean channel only).

PAYLOAD HIDDEN IN EACH STEGO IMAGE (one image per share)
  [ share bits | share check (64) | LSH hash (256) | SHA-256 of cover (256) ]  -> ECC -> Hamming(7,4) embedding

ASSUMPTIONS (state them in the report)
  * Sender and receiver share two secrets: the LSH key and the embedding-position key.
  * The side information (flipped bit positions, 3 bits per sub-band block) reaches the receiver intact
    (separate authenticated channel). Its size is reported so it can be counted in the payload budget.

RUN
  python full_pipeline.py          full run (check table + walkthrough + scenarios), writes results/pipeline_*.csv
  python full_pipeline.py quick    fewer trials
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

import cbss
import ecc
import hamming_me as hm
import lsh
import metrics as mt
from iwt import iwt2, iiwt2
from integrated_experiment import sp, lsbflip, gauss, jpeg, noise_patch, copy_move

OUT = Path("results")
TAU = 3                 # LSH threshold (bits of difference tolerated), from the LSH experiment
K = 3                   # shares needed to rebuild the key
KEY_BITS = 128
SHARE_CHECK_BITS = 64   # truncated SHA-256 of the share bits (guards the key material)
HASH_BITS = ecc.HASH_BITS
LSH_SECRET = 1234
POS_SECRET = 777


# ----------------------------------------------------------------------------- small helpers
def sha_bits(data, n=256):
    """SHA-256 digest of bytes as an array of bits (first n bits).
    Benefit: exact, cryptographic integrity check; any single changed bit changes the digest."""
    return np.unpackbits(np.frombuffer(hashlib.sha256(data).digest(), dtype=np.uint8))[:n]


def share_check(share_bits):
    """64-bit guard over a share. Benefit: the tolerant LSH layer authenticates the IMAGE, so the key
    material needs its own exact check; a corrupted share is then never used to rebuild the key."""
    return sha_bits(np.packbits(share_bits).tobytes() + len(share_bits).to_bytes(4, "big"), SHARE_CHECK_BITS)


def build_payload(share, h_lsh, d_cover):
    return np.concatenate([share, share_check(share), h_lsh, d_cover]).astype(np.uint8)


def split_payload(info, share_len):
    a = share_len
    return info[:a], info[a:a + SHARE_CHECK_BITS], info[a + SHARE_CHECK_BITS:a + SHARE_CHECK_BITS + HASH_BITS], \
        info[a + SHARE_CHECK_BITS + HASH_BITS:a + SHARE_CHECK_BITS + 2 * HASH_BITS]


# ----------------------------------------------------------------------------- L3 error protection
def ecc_encode(scheme, info):
    """Encode a payload of any length in 256-bit rows. Benefit: repairs bits damaged in the public channel."""
    pad = (-len(info)) % HASH_BITS
    rows = np.concatenate([info, np.zeros(pad, np.uint8)]).reshape(-1, HASH_BITS)
    return scheme.encode(rows).reshape(-1)


def ecc_decode(scheme, rx, n_info):
    rows = rx.reshape(-1, scheme.tx_bits)
    return scheme.decode(rows).reshape(-1)[:n_info]


# ----------------------------------------------------------------------------- L2 embedding
def block_order(n_total, n_needed, secret, keyed=True):
    """Which 2x2 blocks carry the payload. Benefit: with a secret permutation an attacker cannot tell where
    the payload is, and a burst of damage is spread over the payload instead of wiping one stretch of it."""
    if not keyed:
        return np.arange(n_needed)
    return np.random.default_rng(secret).permutation(n_total)[:n_needed]


def embed_payload(cover, bits, secret, keyed=True):
    """IWT -> Hamming(7,4) matrix embedding (3 bits per sub-band block) -> inverse IWT.
    Returns stego image, side information, number of clipped pixels, block order.
    Benefit: at most one bit per block changes (high PSNR); the side information makes the change reversible."""
    bits = np.asarray(bits, dtype=np.int64)
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 9, dtype=np.int64)])
    LL, HL, LH, HH = iwt2(cover)
    subs = [HL.copy(), LH.copy(), HH.copy()]
    pos = hm._blocks(HL)
    nblk = len(bits) // 9
    order = block_order(len(pos), nblk, secret, keyed)
    side = []
    for t in range(nblk):
        i, j = pos[order[t]]
        for b, sub in enumerate(subs):
            blk = sub[i:i + 2, j:j + 2]
            before = hm.block_to_x(blk)
            new = hm.embed_block(blk, bits[9 * t + 3 * b: 9 * t + 3 * b + 3])
            d = np.flatnonzero(before != hm.block_to_x(new))
            side.append(int(d[0]) + 1 if len(d) else 0)
            sub[i:i + 2, j:j + 2] = new
    raw = iiwt2(LL, *subs)
    n_clip = int(np.sum((raw < 0) | (raw > 255)))
    return np.clip(raw, 0, 255).astype(np.uint8), np.array(side, dtype=np.uint8), n_clip, order


def extract_payload(img, n_bits, order):
    """IWT of the received image -> syndromes. Benefit: no original image is needed to read the payload."""
    _, HL, LH, HH = iwt2(img)
    pos = hm._blocks(HL)
    out = []
    for t in range((n_bits + 8) // 9):
        i, j = pos[order[t]]
        for sub in (HL, LH, HH):
            out += list(hm.extract_block(sub[i:i + 2, j:j + 2]))
    return np.array(out[:n_bits], dtype=np.uint8)


def recover_cover(received, side, order):
    """Undo the embedding flips using the side information. Benefit: exact cover (MSE 0) when the image
    arrived unchanged; the SHA-256 check then proves it."""
    LL, HL, LH, HH = iwt2(received)
    subs = [HL.copy(), LH.copy(), HH.copy()]
    pos = hm._blocks(HL)
    n = 0
    for t in range(len(side) // 3):
        i, j = pos[order[t]]
        for sub in subs:
            if side[n]:
                blk = sub[i:i + 2, j:j + 2]
                x = hm.block_to_x(blk)
                x[side[n] - 1] ^= 1
                sub[i:i + 2, j:j + 2] = hm.x_to_block(x, blk)
            n += 1
    return np.clip(iiwt2(LL, *subs), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- sender (L1 + L2 + L3 + L4 inputs)
def sender(covers, sk, scheme, keyed=True):
    """key -> CBSS shares -> per share: LSH hash + SHA-256 + guard -> ECC -> embed. One stego image per share."""
    shares = cbss.gen_shares(sk, 1)
    assert len(shares) >= len(covers), "key has too few zero bits to make one share per image; use another key"
    pkgs = []
    for i, cover in enumerate(covers):
        share = shares[i]
        h = lsh.simhash(cover, HASH_BITS, secret=LSH_SECRET)
        d = sha_bits(cover.tobytes())
        info = build_payload(share, h, d)
        coded = ecc_encode(scheme, info)
        stego, side, n_clip, order = embed_payload(cover, coded, POS_SECRET + i, keyed)
        pkgs.append(dict(cover=cover, share=share, info=info, coded=coded, stego=stego, side=side,
                         n_clip=n_clip, order=order))
    return pkgs


# ----------------------------------------------------------------------------- receiver (L3 + L4 + L5)
def receive_share(received, pkg, scheme, tau=TAU):
    """Extract -> ECC decode -> dual authentication. Decision rule:
         EXACT      recovered cover matches the embedded SHA-256 (image arrived bit-exact, reversibility proven)
         TOLERATED  SHA-256 differs but the LSH distance is <= tau (acceptable change)
         REJECTED   otherwise
       A share is USABLE only if it is not REJECTED and its own guard matches."""
    rx = extract_payload(received, len(pkg["coded"]), pkg["order"])
    info_hat = ecc_decode(scheme, rx, len(pkg["info"]))
    share_hat, chk, h_ref, d_ref = split_payload(info_hat, len(pkg["share"]))
    share_ok = bool(np.array_equal(chk, share_check(share_hat)))
    dist = lsh.hamming(h_ref, lsh.simhash(received, HASH_BITS, secret=LSH_SECRET))
    rec = recover_cover(received, pkg["side"], pkg["order"])
    sha_ok = bool(np.array_equal(sha_bits(rec.tobytes()), d_ref))
    decision = "EXACT" if sha_ok else ("TOLERATED" if dist <= tau else "REJECTED")
    return dict(rx=rx, info_hat=info_hat, share_hat=share_hat, share_ok=share_ok, lsh_dist=dist, sha_ok=sha_ok,
                decision=decision, usable=bool(share_ok and decision != "REJECTED"), recovered=rec)


def rebuild_authenticated(results, k=K):
    """Key from the first k USABLE shares, or None. Benefit: a damaged or tampered share is skipped, and the
    receiver is told when no key can be trusted instead of receiving a wrong key silently."""
    idx = [i for i, r in enumerate(results) if r["usable"]][:k]
    if len(idx) < k:
        return None
    return cbss.reconstruct([results[i]["share_hat"] for i in idx], k)[0]


def rebuild_blind(results, k=K):
    """Baseline behaviour: use the first k shares with no check."""
    return cbss.reconstruct([r["share_hat"] for r in results[:k]], k)[0]


# ----------------------------------------------------------------------------- scenarios
GLOBAL_ATTACKS = {"clean": lambda im, r: im, "salt_pepper_0.05%": sp(0.0005), "salt_pepper_0.1%": sp(0.001),
                  "salt_pepper_0.5%": sp(0.005), "lsb_flip_0.1%": lsbflip(0.001),
                  "gaussian_s1": gauss(1), "jpeg_q90": jpeg(90)}
TARGETED = {"tamper_share1_noise_32px": noise_patch(32), "tamper_share1_noise_64px": noise_patch(64),
            "tamper_share1_copy_64px": copy_move(64)}


def scenarios():
    return [(n, f, False) for n, f in GLOBAL_ATTACKS.items()] + [(n, f, True) for n, f in TARGETED.items()]


def schemes():
    ham = ecc.Cyclic("BCH(31,21) t=2", 31, 0b11101101001, 2)
    return [ecc.NoECC(), ecc.Repeat(), ecc.Repeat(ham)]


def run_trial(covers, scheme, trial, key_bits=KEY_BITS, keyed=True):
    sk = np.random.default_rng(10_000 + trial).integers(0, 2, key_bits).astype(np.uint8)
    pkgs = sender(covers, sk, scheme, keyed)
    psnr = float(np.mean([mt.psnr(p["cover"], p["stego"]) for p in pkgs]))
    rows = []
    for si, (name, fn, targeted) in enumerate(scenarios()):
        rng = np.random.default_rng(1000 * trial + si)
        res = []
        for i, p in enumerate(pkgs):
            recv = fn(p["stego"], rng) if (not targeted or i == 0) else p["stego"]
            r = receive_share(recv, p, scheme)
            r["pre_ber"] = float(np.mean(r["rx"] != p["coded"]))
            r["post_ber"] = float(np.mean(r["info_hat"] != p["info"]))
            res.append(r)
        key_a, key_b = rebuild_authenticated(res), rebuild_blind(res)
        auth = "no_key_detected" if key_a is None else ("correct" if np.array_equal(key_a, sk) else "silent_wrong")
        dec = [r["decision"] for r in res]
        rows.append(dict(scenario=name, scheme=scheme.name, trial=trial, key_bits=key_bits,
                         overhead=round(scheme.tx_bits / HASH_BITS, 2),
                         pre_ECC_BER=round(float(np.mean([r["pre_ber"] for r in res])), 5),
                         post_ECC_BER=round(float(np.mean([r["post_ber"] for r in res])), 5),
                         n_exact=dec.count("EXACT"), n_tolerated=dec.count("TOLERATED"), n_rejected=dec.count("REJECTED"),
                         n_usable=sum(r["usable"] for r in res), auth_outcome=auth,
                         blind_outcome="correct" if np.array_equal(key_b, sk) else "silent_wrong",
                         share1_decision=dec[0], share1_usable=int(res[0]["usable"]),
                         n_cover_exact=sum(r["sha_ok"] for r in res), mean_stego_PSNR=round(psnr, 2),
                         payload_bits=len(pkgs[0]["coded"]), sideinfo_bits=3 * len(pkgs[0]["side"]),
                         bpp=round(len(pkgs[0]["coded"]) / covers[0].size, 5),
                         pixels_clipped=sum(p["n_clip"] for p in pkgs)))
    return rows


def summarise(rows, n_images):
    print(f"\n{'scenario':<26}{'scheme':<22}{'preBER':<9}{'postBER':<9}{'EXACT':<7}{'TOL':<6}{'REJ':<6}"
          f"{'auth: correct/detected/SILENT-WRONG':<38}{'blind: correct/WRONG':<22}")
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        for sh in dict.fromkeys(r["scheme"] for r in rows):
            R = [r for r in rows if r["scenario"] == sc and r["scheme"] == sh]
            n = len(R); tot = n * n_images
            a = [r["auth_outcome"] for r in R]; b = [r["blind_outcome"] for r in R]
            a_txt = f"{a.count('correct')/n:.2f} / {a.count('no_key_detected')/n:.2f} / {a.count('silent_wrong')/n:.2f}"
            b_txt = f"{b.count('correct')/n:.2f} / {b.count('silent_wrong')/n:.2f}"
            print(f"{sc:<26}{sh:<22}{np.mean([r['pre_ECC_BER'] for r in R]):<9.4f}{np.mean([r['post_ECC_BER'] for r in R]):<9.4f}"
                  f"{sum(r['n_exact'] for r in R)/tot:<7.2f}{sum(r['n_tolerated'] for r in R)/tot:<6.2f}{sum(r['n_rejected'] for r in R)/tot:<6.2f}"
                  f"{a_txt:<38}{b_txt:<22}")


# ----------------------------------------------------------------------------- supervisor's check: IWT, inverse IWT, BER
def check_iwt_ber(images, n_bits=4000, seed=5):
    """For each image: IWT -> inverse IWT is exact; embed random bits -> inverse IWT -> IWT -> extract gives BER.
    Benefit: shows the transform pair and the embedding are correct before anything else is built on them."""
    rng = np.random.default_rng(seed)
    rows = []
    for name, img in images.items():
        pr = bool(np.array_equal(iiwt2(*iwt2(img)), img))
        bits = rng.integers(0, 2, n_bits).astype(np.uint8)
        stego, side, n_clip, order = embed_payload(img, bits, POS_SECRET)
        got = extract_payload(stego, n_bits, order)
        rec = recover_cover(stego, side, order)
        rows.append(dict(image=name, IWT_inverse_exact=pr, payload_bits=n_bits, BER_clean=mt.ber(bits, got),
                         pixels_clipped=n_clip, MSE_stego=round(mt.mse(img, stego), 6),
                         PSNR_stego=round(mt.psnr(img, stego), 2), SSIM_stego=round(mt.ssim(img, stego), 6),
                         MSE_recovered=mt.mse(img, rec), sideinfo_bits=3 * len(side)))
    return rows


# ----------------------------------------------------------------------------- walkthrough
def walkthrough(covers, scheme):
    sk = np.random.default_rng(1).integers(0, 2, KEY_BITS).astype(np.uint8)
    pkgs = sender(covers, sk, scheme)
    p = pkgs[0]
    n_blocks_total = (covers[0].shape[0] // 4) * (covers[0].shape[1] // 4)
    print(f"\n=== Walkthrough (scheme: {scheme.name}) ===")
    print(f"L1 CBSS      key {KEY_BITS} bits -> {len(cbss.gen_shares(sk, 1))} shares of {KEY_BITS} bits; any k={K} rebuild the key")
    print(f"L4 inputs    LSH hash {HASH_BITS} bits + SHA-256 of cover {HASH_BITS} bits + share guard {SHARE_CHECK_BITS} bits")
    print(f"             payload before ECC = {len(p['info'])} bits")
    print(f"L3 ECC       -> {len(p['coded'])} bits sent (x{scheme.tx_bits / HASH_BITS:.2f} per 256-bit row)")
    nb = (len(p['coded']) + 8) // 9
    print(f"L2 embed     {nb} of {n_blocks_total} blocks used ({100 * nb / n_blocks_total:.1f}%), keyed positions; "
          f"bpp = {len(p['coded']) / covers[0].size:.5f}")
    print(f"             stego PSNR {mt.psnr(p['cover'], p['stego']):.2f} dB, SSIM {mt.ssim(p['cover'], p['stego']):.5f}, "
          f"clipped pixels {p['n_clip']}, side info {3 * len(p['side'])} bits")
    res = [receive_share(q["stego"], q, scheme) for q in pkgs]
    r = res[0]
    print(f"L5 receive   clean: decision {r['decision']}, LSH distance {r['lsh_dist']}, SHA match {r['sha_ok']}, "
          f"cover MSE after recovery {mt.mse(p['cover'], r['recovered']):.1f}")
    key = rebuild_authenticated(res)
    print(f"             key rebuilt from {sum(x['usable'] for x in res)} usable shares: {'correct' if key is not None and np.array_equal(key, sk) else 'FAILED'}")
    att = sp(0.001)(p["stego"], np.random.default_rng(3))
    ra = receive_share(att, p, scheme)
    print(f"             salt-and-pepper 0.1% on this image: pre-ECC BER {np.mean(ra['rx'] != p['coded']):.4f}, "
          f"post-ECC BER {np.mean(ra['info_hat'] != p['info']):.4f}, decision {ra['decision']} (LSH distance {ra['lsh_dist']})")


# ----------------------------------------------------------------------------- main
def load_images():
    try:
        from dataset import load_dataset
        return load_dataset()
    except Exception as e:                      # fall back so the script still runs
        print(f"(dataset not found: {e}) using 5 synthetic images")
        out = {}
        for s in range(5):
            r = np.random.default_rng(s)
            f = np.fft.fft2(r.normal(size=(512, 512)))
            ky, kx = np.meshgrid(np.fft.fftfreq(512), np.fft.fftfreq(512), indexing="ij")
            f *= 1 / (1e-3 + np.sqrt(kx ** 2 + ky ** 2)) ** 1.2
            a = np.real(np.fft.ifft2(f)); a = (a - a.min()) / (a.max() - a.min())
            out[f"synthetic{s}"] = np.clip(30 + 190 * a + r.normal(0, 3, (512, 512)), 0, 255).astype(np.uint8)
        return out


def save_csv(name, rows):
    OUT.mkdir(exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved results/{name} ({len(rows)} rows)")


if __name__ == "__main__":
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    trials = 3 if quick else 10
    images = load_images()
    names, covers = list(images), [images[n] for n in images][:5]

    print("=== 1. IWT, inverse IWT and BER check (clean channel) ===")
    chk = check_iwt_ber(images)
    print(f"{'image':<12}{'IWT exact':<11}{'BER':<8}{'clipped':<9}{'MSE stego':<11}{'PSNR':<8}{'SSIM':<10}{'MSE recovered':<14}{'side info bits'}")
    for r in chk:
        print(f"{r['image']:<12}{str(r['IWT_inverse_exact']):<11}{r['BER_clean']:<8.3f}{r['pixels_clipped']:<9}{r['MSE_stego']:<11.5f}"
              f"{r['PSNR_stego']:<8.2f}{r['SSIM_stego']:<10.5f}{r['MSE_recovered']:<14.1f}{r['sideinfo_bits']}")
    save_csv("pipeline_check.csv", chk)

    sch = schemes()
    walkthrough(covers, sch[2])

    print(f"\n=== 2. Scenarios: {trials} trials x {len(covers)} shares per scheme ===")
    rows = []
    for s in sch:
        for t in range(trials):
            rows += run_trial(covers, s, t)
        print(f"  done: {s.name}")
    save_csv("pipeline_runs.csv", rows)
    summarise(rows, len(covers))
