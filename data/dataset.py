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

NAMES = ["Baboon", "Airplane", "Sailboat", "Peppers", "Splash"]

def load_dataset(folder="data/gray_images"):
    images = {}
    for name in NAMES:
        images[name] = np.array(Image.open(Path(folder) / f"{name}.png"))
    return images