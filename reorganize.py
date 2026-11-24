import ast, re, shutil, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APPLY = "--apply" in sys.argv
FOLDERS = ("cbss", "data", "ecc", "eval", "iwt", "lsh")

# destination -> list of (candidate source names, new name or None)
PLAN = {
    "data": [(["dataset.py"], None), (["convert_gray.py"], None), (["convert_airplane.py", "convert_airp.py", "fix_airplane.py"], None),
             (["verify.py"], None)],
    "cbss": [(["cbss.py"], None)],
    "lsh": [(["lsh.py"], None), (["lsh_experiment.py"], None), (["lsh_block_experiment.py"], None),
            (["heldout_tau.py"], None), (["matched_sha_lsh.py"], None)],
    "ecc": [(["ecc.py"], None), (["ecc_experiment.py", "ecc.experiment.py"], "ecc_experiment.py")],
    "eval": [(["hamming_me.py"], None), (["hamming1511.py"], None), (["hamming_compare.py"], None), (["metrics.py"], None),
             (["test_baseline.py"], None), (["test_module.py"], None)],
    "iwt": [(["iwt.py"], None), (["diagnostics.py", "diagnostic.py"], "diagnostics.py"), (["integrated_experiment.py"], None),
            (["median_experiment.py"], None)],
    "_removed": [(["main.py"], None)],
    "results": [(["results.zip"], None)],
}
DATA_DIRS = ["color_images", "gray_images"]
MARK = "project layout bootstrap"
BOOT = '''# --- project layout bootstrap (added by reorganize.py) ---
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
'''


def add_bootstrap(path):
    src = path.read_text(encoding="utf-8")
    if MARK in src:
        return "already has bootstrap"
    lines = src.split("\n")
    at = 0
    try:
        tree = ast.parse(src)
        for node in tree.body:
            is_doc = isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and isinstance(node.value.value, str)
            is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
            if (is_doc and node is tree.body[0]) or is_future:
                at = max(at, node.end_lineno)
            else:
                break
    except SyntaxError:
        return "SYNTAX ERROR in file: bootstrap NOT added, check this file"
    new = "\n".join(lines[:at]) + ("\n" if at else "") + BOOT + "\n".join(lines[at:])
    if APPLY:
        path.write_text(new, encoding="utf-8")
    return "bootstrap added"


def patch_data_paths(path):
    src = path.read_text(encoding="utf-8", errors="replace")
    if "__file__" in src.replace(BOOT, ""):
        return "uses __file__ for its paths: left unchanged (check it if the images are not found)"
    new, n = re.subn(r'(["\'])(color_images|gray_images)(?=[/\\"\'])', r'\1data/\2', src)
    if n and APPLY:
        path.write_text(new, encoding="utf-8")
    return f"{n} path(s) rewritten to data/..." if n else "no image-folder paths found"


def main():
    print(f"Project root: {ROOT}\nMode: {'APPLY' if APPLY else 'DRY RUN (nothing is changed)'}\n")
    if not (ROOT / "full_pipeline.py").exists():
        sys.exit("full_pipeline.py not found here. Put reorganize.py in the project root (next to full_pipeline.py) and run it from there.")
    moves, missing = [], []
    for dest, items in PLAN.items():
        for cands, newname in items:
            src = next((ROOT / c for c in cands if (ROOT / c).exists()), None)
            if src is None:
                missing.append(f"{dest}/{newname or cands[0]}  (looked for: {', '.join(cands)})")
            else:
                moves.append((src, ROOT / dest / (newname or src.name)))
    dir_moves = [(ROOT / d, ROOT / "data" / d) for d in DATA_DIRS if (ROOT / d).is_dir()]

    print("Files to move:")
    for a, b in moves + dir_moves:
        print(f"  {a.name:<28}->  {b.relative_to(ROOT)}")
    if missing:
        print("\nNot found (skipped, fine if you never had them):")
        for m in missing:
            print("  -", m)
    listed = {a for a, _ in moves + dir_moves} | {ROOT / "full_pipeline.py", ROOT / "reorganize.py"}
    other = [p.name for p in sorted(ROOT.iterdir()) if p not in listed and p.name not in FOLDERS and p.suffix in {".py"}]
    if other:
        print("\nOther .py files left at the top, not touched:", ", ".join(other))
    if not APPLY:
        print("\nDry run only. Run  python reorganize.py --apply  to do it (a backup is made first).")
        return

    backup = ROOT.parent / f"{ROOT.name}_backup_{datetime.datetime.now():%Y%m%d_%H%M}"
    print(f"\nBacking up to {backup} ...")
    shutil.copytree(ROOT, backup, ignore=shutil.ignore_patterns("venv", ".venv", "__pycache__", ".git"))
    for d in (*PLAN, "results"):
        (ROOT / d).mkdir(exist_ok=True)
    for a, b in moves + dir_moves:
        if b.exists():
            print(f"  !! {b.relative_to(ROOT)} already exists, skipped"); continue
        shutil.move(str(a), str(b))
    print("\nFixing imports and paths:")
    for p in [ROOT / "full_pipeline.py"] + [b for _, b in moves if b.suffix == ".py" and b.parent.name != "_removed"]:
        if p.exists():
            print(f"  {str(p.relative_to(ROOT)):<34}{add_bootstrap(p)}")
    for _, b in moves:
        if b.parent.name == "data" and b.suffix == ".py":
            print(f"  {str(b.relative_to(ROOT)):<34}{patch_data_paths(b)}")
    (ROOT / "README.md").write_text(README, encoding="utf-8")
    print(f"\nDone. Backup kept at {backup}.\nNow run, from this folder:  python full_pipeline.py quick")
    print("Held-out image folders can live in data/, e.g.  python lsh/heldout_tau.py data/heldout_images")


README = """# Project layout (run everything from this folder)

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
"""

if __name__ == "__main__":
    main()
