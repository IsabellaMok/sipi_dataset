from dataset import load_dataset

images = load_dataset()

for name, img in images.items():
    print(name, img.shape)