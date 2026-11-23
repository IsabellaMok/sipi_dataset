from pathlib import Path
from PIL import Image

INPUT_DIR = Path("color_images")
OUTPUT_DIR = Path("gray_images")
OUTPUT_DIR.mkdir(exist_ok=True)

files = [f for f in INPUT_DIR.iterdir()
         if f.suffix.lower() in (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp")]

if not files:
    print("No images found in color_images/")

for f in sorted(files):
    img = Image.open(f)
    print(f"{f.name}: mode={img.mode}, size={img.size}")

    gray = img.convert("L")          # "L" = 8-bit grayscale

    if gray.size != (512, 512):
        print(f"  WARNING: size is {gray.size}, not 512x512")

    out_path = OUTPUT_DIR / (f.stem + "_gray.png")
    gray.save(out_path)              # PNG is lossless
    print(f"  saved -> {out_path}")

print("Done.")