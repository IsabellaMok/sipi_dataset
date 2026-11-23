from pathlib import Path
import numpy as np
from PIL import Image

NAMES = ["Baboon", "Airplane", "Sailboat", "Peppers", "Splash"]

def load_dataset(folder="gray_images"):
    images = {}
    for name in NAMES:
        images[name] = np.array(Image.open(Path(folder) / f"{name}.png"))
    return images