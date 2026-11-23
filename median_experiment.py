"""Median-filter attack (3x3 and 5x5) through the full pipeline, for every ECC scheme.
Median filtering is a common image-processing operation, so it tests how far the scheme tolerates 'benign' processing.
Run:  python median_experiment.py          Output: results/median_runs.csv, results/median_lsh_distance.csv
"""
import csv, sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import full_pipeline as fp
import lsh


def median(k):
    def f(im, rng):
        p = k // 2
        pad = np.pad(im, p, mode="reflect")
        return np.median(sliding_window_view(pad, (k, k)), axis=(2, 3)).astype(np.uint8)
    return f


if __name__ == "__main__":
    trials = 3 if (len(sys.argv) > 1 and sys.argv[1] == "quick") else 10
    images = fp.load_images()
    covers = [images[n] for n in images][:5]
    fp.GLOBAL_ATTACKS = {"median_3x3": median(3), "median_5x5": median(5)}     # run only these scenarios
    fp.TARGETED = {}
    rows = []
    for s in fp.schemes():
        for t in range(trials):
            rows += fp.run_trial(covers, s, t)
        print(f"  done: {s.name}")
    fp.save_csv("median_runs.csv", rows)
    fp.summarise(rows, len(covers))

    # LSH distance alone: how far does a median filter move the keyed SimHash of the LL sub-band? (tau = 3)
    d_rows = []
    print(f"\nLSH Hamming distance after median filtering (tau = {fp.TAU}; 256-bit SimHash):")
    for name, img in images.items():
        ref = lsh.simhash(img, 256)
        for lab, k in (("median_3x3", 3), ("median_5x5", 5)):
            d = lsh.hamming(ref, lsh.simhash(median(k)(img, None), 256))
            d_rows.append(dict(image=name, filter=lab, lsh_distance=d, accepted_at_tau3=int(d <= fp.TAU)))
            print(f"  {name:<12}{lab:<12}distance {d:<4}{'accepted' if d <= fp.TAU else 'REJECTED'}")
    with open(fp.OUT / "median_lsh_distance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d_rows[0].keys())); w.writeheader(); w.writerows(d_rows)
    print("saved results/median_lsh_distance.csv")