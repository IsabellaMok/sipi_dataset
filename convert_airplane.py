from PIL import Image

img = Image.open("gray_images/Airplane.tiff")

# Convert to 8-bit grayscale
img = img.convert("L")

# Save as PNG
img.save("gray_images/Airplane.png")

print("Conversion successful!")
print("Image size:", img.size)
print("Image mode:", img.mode)