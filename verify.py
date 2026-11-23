from pathlib import Path
import numpy as np
from PIL import Image

for f in sorted(Path("gray_images").glob("*.png")):
    img = Image.open(f)
    arr = np.array(img)
    print(f"{f.name:<25} mode={img.mode}  shape={arr.shape}  dtype={arr.dtype}  min={arr.min()}  max={arr.max()}")