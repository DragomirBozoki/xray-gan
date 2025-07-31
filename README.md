
<img width="512" height="512" alt="epoch_2830_img_4" src="https://github.com/user-attachments/assets/01ea7890-822a-4697-b580-a83f0b1182bd" />



🧠 Chest X-ray Image Generation using GAN

A project focused on generating realistic chest X-ray images using a Generative Adversarial Network (GAN).

Author: Dragomir Božoki
Year: 2025
Faculty: FTN Novi Sad
📌 Project Overview

This project implements a GAN-based model that generates synthetic chest X-ray images. The GAN architecture consists of:

    A Generator that learns to produce realistic images

    A Discriminator that learns to distinguish real images from generated ones

Through adversarial training, the generator becomes capable of producing highly convincing medical images resembling real X-rays.
🧰 Technical Details
📁 Data Preparation

    Initial dataset: ~1700 X-ray images

    Image resolution: 512x512

    Data augmentation (brightness, contrast, JPEG quality) → expanded to ~3000 images

    Batch size: 8

    Images loaded via dataloader.py

🧠 Architecture

    Generator uses:

        UpSampling layers

        More filters in deeper layers

    Discriminator outputs binary classification (real/fake)

    Skip-connection layers included for preserving fine details

⚙️ Training

    Trained for 3500 epochs

    Additional Fine-Tuning phase (~500 epochs) with reduced learning rate

    Useful training techniques:

        Label smoothing

        Label swapping

        Training the generator multiple times per discriminator update

✅ Results

The trained model successfully generates images with:

    Coarse anatomical structure: lungs, ribs, chest outline

    Fine details: clavicles, soft tissue, tonal variation

📈 Key Takeaways

    Balance between generator and discriminator is essential

    Larger models ≠ better performance (can lead to instability)

    Data augmentation helps, but must be used with caution

    Data volume matters more than architecture complexity

    Hyperparameter tuning (e.g., learning rate, update ratio G/D) has major impact

🚀 Future Improvements

    Increase dataset size and diversity

    Enhance architecture:

        Add more filters in deeper layers

        Use skip connections to retain visual details

    More advanced fine-tuning strategies with adaptive learning rate

    Implement Conditional GANs for better output control

    Explore other GAN architectures:

        StyleGAN

        Pix2Pix

        CycleGAN

🧑‍⚕️ Potential Applications

    Augmenting limited medical datasets

    Supporting diagnostic AI model training

    Educational use in medical AI


