# Project layout (run everything from this folder)

| Folder | Contents |
|---|---|
| `full_pipeline.py` | end-to-end pipeline (CBSS, IWT + Hamming, ECC, dual authentication, recovery) |
| `data/` | `gray_images/`, `color_images/`, `dataset.py`, convert and verify scripts; put held-out sets here too |
| `results/` | every CSV the experiments write |
| `cbss/` | counting-based secret sharing |
| `lsh/` | LSH hash, threshold experiments, held-out check, SHA-256 vs LSH |
| `ecc/` | error-correcting codes and their experiments |
| `eval/` | Hamming (7,4) and (15,11), their comparison, metrics, baseline tests |
| `iwt/` | integer wavelet transform, diagnostics, integrated experiment, median filtering |

Commands: `python full_pipeline.py quick`, `python lsh/lsh_experiment.py`, `python lsh/heldout_tau.py data/heldout_images`,
`python lsh/matched_sha_lsh.py quick`, `python ecc/ecc_experiment.py`, `python eval/hamming_compare.py data/heldout_images`,
`python iwt/median_experiment.py quick`, `python iwt/integrated_experiment.py`, `python iwt/diagnostics.py`.
