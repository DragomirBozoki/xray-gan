import os
from PIL import Image, ImageEnhance
import numpy as np
import random

input_folder = "data/resized_512"
output_folder = "data/resized_512_augmented"

os.makedirs(output_folder, exist_ok=True)

# Uzmi prvih 1500 slika
all_images = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:1500]

def apply_augmentations(img):
    # Random brightness
    brightness_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Random contrast
    contrast_factor = random.uniform(0.8, 1.2)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Gaussian noise
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 10, np_img.shape)  # std=10
    np_img += noise
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

for idx, filename in enumerate(all_images):
    path = os.path.join(input_folder, filename)
    img = Image.open(path).convert("L")  # grayscale

    aug_img = apply_augmentations(img)
    base_name = os.path.splitext(filename)[0]
    aug_img.save(os.path.join(output_folder, base_name + "_aug.png"))

    if idx % 100 == 0:
        print(f"✅ {idx}/1500 done...")

print("🎉 All 1500 augmented X-ray images saved to", output_folder)
