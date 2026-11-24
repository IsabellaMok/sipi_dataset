from pathlib import Path
from PIL import Image

folder = Path("heldout_images")
for p in folder.iterdir():
    if p.suffix.lower() in {".tif", ".tiff"}:
        Image.open(p).save(p.with_suffix(".png"))
        print("converted", p.name)