import matplotlib.pyplot as plt
import ipywidgets as widgets  
import numpy as np
from PIL import Image
import os
import tensorflow as tf

input_folder = "data/images"
output_folder = "data/resized_1024"
target_size = (1024, 1024)

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith('.png'):
            input_path = os.path.join(root, filename)

            # Ispravno: relativna putanja
            relative_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)

            # Kreiraj izlazni podfolder ako ne postoji
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with Image.open(input_path) as img:
                resized_img = img.resize(target_size)
                resized_img.save(output_path)

            print(f'Resized: {relative_path}')

print("Done!")
