
# 🧠 Chest X-ray Generation using GANs

This project implements a **Generative Adversarial Network (GAN)** for generating realistic chest X-ray images. The goal is to synthetically augment medical imaging datasets, especially when labeled data is limited or expensive to acquire.

---

## 🩻 Project Summary

- **Type**: Deep Learning / Medical Imaging
- **Model**: DCGAN (Deep Convolutional GAN)
- **Input**: Random noise vector `z ~ N(0, 1)`
- **Output**: 256×256 grayscale chest X-ray images
- **Tools**: Python, TensorFlow/Keras, NumPy, Matplotlib

---

## 🎯 Objectives

- Learn the distribution of real chest X-ray images
- Generate new, high-quality synthetic images
- Evaluate performance qualitatively (visual inspection) and quantitatively (pixel histograms, FID-like metrics)
- Explore data augmentation and anomaly simulation use cases

---

## 📁 Dataset

- **Source**: [NIH Chest X-ray Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) *(used preprocessed subset)*
- **Resolution**: Resized to 256×256 grayscale
- **Preprocessing**:
  - Histogram equalization
  - Normalization to [−1, 1]
  - Binarization of labels removed (unsupervised generation)

---

## 🏗️ Architecture

### Generator

- Dense layer projecting latent vector to 8×8×256
- Transposed convolutions with BatchNorm and ReLU activations
- Output: 256×256 grayscale image with `tanh` activation

### Discriminator

- Convolutional layers with LeakyReLU and Dropout
- Output: Probability of image being real (`sigmoid`)

### Loss Functions

- **Discriminator**: Binary Cross-Entropy
- **Generator**: Binary Cross-Entropy (flipped labels)
- Optional: Label smoothing, noise injection for stability

---

## ⚙️ Training

- **Epochs**: 3500+
- **Batch size**: 64
- **Latent dimension (z)**: 100
- **Optimizers**: Adam (lr=0.0002, beta1=0.5)
- **Stabilization techniques**:
  - Label smoothing
  - Label flipping (10%)
  - Noise injection to discriminator input
  - Conditional training steps (more generator updates)

---

## 📈 Results

### ✅ Qualitative

- Generated images closely resemble real chest X-rays in terms of structure and contrast.
- Visual realism improves significantly after ~2000 epochs.

### 📊 Quantitative

- Pixel intensity histograms show similar distributions between real and synthetic data.
- Euclidean and Manhattan distance metrics used to compare real vs. generated images (early-stage analysis).

---

## 📦 Folder Structure


---

## 📌 Sample Results

### Epoch 500 vs. Epoch 3000
| Epoch 500                | Epoch 3000               |
|--------------------------|--------------------------|
| ![<img width="512" height="512" alt="epoch_500_img_1" src="https://github.com/user-attachments/assets/b406a004-bc0b-466e-bfc3-5587b32da93f" />
]() | ![]()<img width="512" height="512" alt="epoch_3040_img_5" src="https://github.com/user-attachments/assets/a6140547-912f-4e96-b059-3000bb82fc40" />
 |

---

## 🚀 Future Work

- Add **Conditional GAN (cGAN)** based on disease class labels (e.g., pneumonia)
- Train on higher resolution (512×512)
- Apply **FID** and **SSIM** metrics for better image evaluation
- Explore integration into medical diagnosis pipeline (data augmentation)

---

## 🧠 Skills Demonstrated

- GAN architecture implementation from scratch
- Deep learning for medical imaging
- Image preprocessing and dataset curation
- Training stabilization techniques
- Evaluation of generative model quality

---


---


