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
from PIL import Image

img = Image.open("data/gray_images/Airplane.tiff")

# Convert to 8-bit grayscale
img = img.convert("L")

# Save as PNG
img.save("data/gray_images/Airplane.png")

print("Conversion successful!")
print("Image size:", img.size)
print("Image mode:", img.mode)