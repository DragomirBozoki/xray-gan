# augmentation.py
import tensorflow as tf
import os
from glob import glob

INPUT_FOLDER  = "data/data/resized_512"
OUTPUT_FOLDER = "data/data/resized_512"

# ⚠️  BEZ prostornih transformacija
def augment_img(img):
    img = tf.image.random_brightness(img, max_delta=0.08)          # ±8 % svetline
    img = tf.image.random_contrast(img, lower=0.92, upper=1.08)    # ±8 % kontrasta
    return img

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)            # [0,1]
    img = tf.image.resize(img, [512, 512])                         # osiguraj dimenzije
    return img

def save_image(tensor, out_path):
    uint8 = tf.image.convert_image_dtype(tensor, tf.uint8, saturate=True)
    tf.io.write_file(out_path, tf.image.encode_jpeg(uint8))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
paths = glob(os.path.join(INPUT_FOLDER, "*.jpeg"))

for p in paths:
    base, ext = os.path.splitext(os.path.basename(p))
    out_name  = f"{base}_aug{ext}"
    aug_img   = augment_img(load_image(p))
    save_image(aug_img, os.path.join(OUTPUT_FOLDER, out_name))

print(f"Saved {len(paths)} augmented images → '{OUTPUT_FOLDER}' (suffix _aug)")
