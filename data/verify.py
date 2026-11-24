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
from pathlib import Path
import numpy as np
from PIL import Image

for f in sorted(Path("data/gray_images").glob("*.png")):
    img = Image.open(f)
    arr = np.array(img)
    print(f"{f.name:<25} mode={img.mode}  shape={arr.shape}  dtype={arr.dtype}  min={arr.min()}  max={arr.max()}")