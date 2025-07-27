import os
import tensorflow as tf
from glob import glob

# === Konfiguracija ===
FOLDER = "data/data/resized_512"
PATTERN = os.path.join(FOLDER, "*.jpeg")

def normalize_img(image):
    return tf.cast(image, tf.float32) / 127.5 - 1.0

def decode_and_process_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [512, 512])
    return normalize_img(img)

def augment(image):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_jpeg_quality(image, 90, 100)
    return image

def build_dataset():
    files = tf.io.gfile.glob(PATTERN)
    
    if not files:
        raise FileNotFoundError(f"[!] Nisu pronađene slike u: {PATTERN}")
    
    print(f"[✓] Pronađeno {len(files)} slika.")
    print("[INFO] Primeri fajlova:")
    for f in files[:5]:
        print(" -", f)

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(decode_and_process_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1300)
    ds = ds.batch(8, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

