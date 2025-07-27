import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from dataloader import build_dataset
from generator import build_generator
from discriminator import build_discriminator

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ========== Dataset ==========
ds = build_dataset()

# ========== Model ==========
generator = build_generator()
discriminator = build_discriminator()

# ========== Optimizatori i gubici ==========
g_opt = Adam(1e-4)
d_opt = Adam(1e-4)
bce   = BinaryCrossentropy(from_logits=False)

# ========== Utility ==========
def smooth_labels(y, smooth=0.1):
    return y * (1.0 - smooth) + 0.5 * smooth

def flip_labels(y, prob=0.05):
    mask = tf.random.uniform(tf.shape(y)) < prob
    return tf.where(mask, 1 - y, y)

# ========== GAN model ==========
class XRayGAN(Model):
    def __init__(self, gen, disc, zdim=256):
        super().__init__()
        self.G, self.D, self.zdim = gen, disc, zdim

    def compile(self, g_opt, d_opt, loss_fn):
        super().compile()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.loss_fn = loss_fn

    def train_step(self, real_imgs):
        bs = tf.shape(real_imgs)[0]
        z = tf.random.normal((bs, self.zdim))
        fake_imgs = self.G(z, training=True)

        # Discriminator
        with tf.GradientTape() as tape:
            y_real = self.D(real_imgs, training=True)
            y_fake = self.D(fake_imgs, training=True)
            real_lbl = flip_labels(smooth_labels(tf.ones_like(y_real)))
            fake_lbl = flip_labels(smooth_labels(tf.zeros_like(y_fake)))
            d_loss = self.loss_fn(real_lbl, y_real) + self.loss_fn(fake_lbl, y_fake)

        grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables))

        # Generator
        z = tf.random.normal((bs, self.zdim))
        with tf.GradientTape() as tape:
            gen_imgs = self.G(z, training=True)
            y_fake = self.D(gen_imgs, training=False)
            g_loss = self.loss_fn(tf.ones_like(y_fake), y_fake)

        grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.G.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# ========== Callback ==========
class ModelMonitor(Callback):
    def __init__(self, G, D, zdim=256, save_dir="generated", start=0):
        super().__init__()
        self.G = G
        self.D = D
        self.zdim = zdim
        self.start = start
        self.dir = save_dir
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        global_epoch = self.start + epoch + 1

        if global_epoch <= 103 and global_epoch % 3 == 0:
            n = 4
        elif global_epoch > 103 and global_epoch % 10 == 0:
            n = 5
        else:
            return  
        z = tf.random.normal((n, self.zdim))
        img = (self.G(z, training=False) + 1) / 2.0
        for i in range(n):
            fp = os.path.join(self.dir, f"epoch_{global_epoch}_img_{i+1}.png")
            plt.imsave(fp, tf.squeeze(img[i]).numpy(), cmap="gray")
        print(f"[Monitor] Saved {n} samples for epoch {global_epoch}")

        # Snimanje modela na svakih 500 epoha
        if global_epoch % 500 == 0:
            g_path = os.path.join("checkpoints", f"generator_epoch_{global_epoch}.h5")
            d_path = os.path.join("checkpoints", f"discriminator_epoch_{global_epoch}.h5")
            self.G.save(g_path)
            self.D.save(d_path)
            print(f"[Checkpoint] Saved generator to {g_path}")
            print(f"[Checkpoint] Saved discriminator to {d_path}")

# ========== Treniranje ==========
gan = XRayGAN(generator, discriminator)
gan.compile(g_opt, d_opt, bce)

monitor = ModelMonitor(generator, discriminator, start=0)
gan.fit(ds,
        initial_epoch=0,
        epochs=10000,
        callbacks=[monitor],
        verbose=1)

# ========== Finalni modeli ==========
generator.save("checkpoints/generator_final.h5")
discriminator.save("checkpoints/discriminator_final.h5")
print("[✓] Final models saved after training.")
