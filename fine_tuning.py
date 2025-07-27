import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from dataloader import build_dataset

# ========== GPU Memory Growth ==========
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[✓] Memory growth enabled for GPU")
    except RuntimeError as e:
        print(e)

# ========== Mixed Precision ==========
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("[✓] Mixed precision enabled")

# ========== Config ==========
START_EPOCH = 3000
TOTAL_EPOCHS = 5000    # nastavi još 2000 epoha
Z_DIM = 256
SAVE_IMG_EVERY = 5
CHECKPOINT_DIR = "checkpoints"
GEN_CKPT = os.path.join(CHECKPOINT_DIR, f"generator_epoch_{START_EPOCH}.h5")
DISC_CKPT = os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{START_EPOCH}.h5")

LR_G = 5e-5
LR_D = 1e-4
BETA1, BETA2 = 0.0, 0.99
EMA_DECAY = 0.999

os.makedirs("generated_ft", exist_ok=True)
os.makedirs("checkpoints_ft", exist_ok=True)

# ========== Dataset ==========
ds = build_dataset()

# ========== Učitaj modele ==========
generator = load_model(GEN_CKPT, compile=False)
discriminator = load_model(DISC_CKPT, compile=False)
print(f"[✓] Loaded checkpoints @ epoch {START_EPOCH}")

# ========== Optimizatori i gubici ==========
g_opt = Adam(LR_G, beta_1=BETA1, beta_2=BETA2)
d_opt = Adam(LR_D, beta_1=BETA1, beta_2=BETA2)
bce   = BinaryCrossentropy(from_logits=False)

# ========== EMA Helper ==========
def clone_weights_like(model):
    clone = tf.keras.models.clone_model(model)
    clone.build(model.inputs[0].shape)
    clone.set_weights(model.get_weights())
    return clone

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = clone_weights_like(model)
    def update(self, model):
        for w_ema, w in zip(self.ema_model.weights, model.weights):
            w_ema.assign(self.decay * w_ema + (1. - self.decay) * w)

# ========== Label Tricks ==========
def smooth_labels(y, smooth=0.1):
    return y * (1.0 - smooth) + 0.5 * smooth
def flip_labels(y, prob=0.05):
    mask = tf.random.uniform(tf.shape(y)) < prob
    return tf.where(mask, 1 - y, y)

# ========== GAN Model ==========
class XRayGAN(Model):
    def __init__(self, gen, disc, zdim=256, ema_decay=0.999):
        super().__init__()
        self.G, self.D, self.zdim = gen, disc, zdim
        self.ema = EMA(self.G, ema_decay) if ema_decay is not None else None
    def compile(self, g_opt, d_opt, loss_fn):
        super().compile()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.loss_fn = loss_fn
    def train_step(self, real_imgs):
        bs = tf.shape(real_imgs)[0]
        z = tf.random.normal((bs, self.zdim))
        fake_imgs = self.G(z, training=True)

        # --- Discriminator ---
        with tf.GradientTape() as tape:
            y_real = self.D(real_imgs, training=True)
            y_fake = self.D(fake_imgs, training=True)
            real_lbl = flip_labels(smooth_labels(tf.ones_like(y_real)))
            fake_lbl = flip_labels(smooth_labels(tf.zeros_like(y_fake)))
            d_loss = self.loss_fn(real_lbl, y_real) + self.loss_fn(fake_lbl, y_fake)
        grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_opt.apply_gradients(zip(grads, self.D.trainable_variables))

        # --- Generator ---
        z = tf.random.normal((bs, self.zdim))
        with tf.GradientTape() as tape:
            gen_imgs = self.G(z, training=True)
            y_fake = self.D(gen_imgs, training=False)
            g_loss = self.loss_fn(tf.ones_like(y_fake), y_fake)
        grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.G.trainable_variables))

        if self.ema is not None:
            self.ema.update(self.G)
        return {"d_loss": d_loss, "g_loss": g_loss}

# ========== Callback ==========
class FineTuneMonitor(Callback):
    def __init__(self, model: XRayGAN, zdim=256, start_epoch=0,
                 save_dir="generated_ft", save_every=5, n_samples=10):
        super().__init__()
        self.model = model
        self.zdim = zdim
        self.start = start_epoch
        self.dir = save_dir
        self.save_every = save_every
        self.n = n_samples
    def on_epoch_end(self, epoch, logs=None):
        global_epoch = self.start + epoch + 1
        if global_epoch % self.save_every != 0:
            return
        G_to_use = self.model.ema.ema_model if self.model.ema is not None else self.model.G
        z = tf.random.normal((self.n, self.zdim))
        img = (G_to_use(z, training=False) + 1) / 2.0
        for i in range(self.n):
            fp = os.path.join(self.dir, f"epoch_{global_epoch}_img_{i+1}.png")
            plt.imsave(fp, tf.squeeze(img[i]).numpy(), cmap="gray")
        print(f"[Monitor] Saved {self.n} samples for epoch {global_epoch}")
        if global_epoch % 100 == 0:
            g_path = os.path.join("checkpoints_ft", f"generator_epoch_{global_epoch}.h5")
            d_path = os.path.join("checkpoints_ft", f"discriminator_epoch_{global_epoch}.h5")
            g_ema_path = os.path.join("checkpoints_ft", f"generatorEMA_epoch_{global_epoch}.h5")
            self.model.G.save(g_path)
            self.model.D.save(d_path)
            if self.model.ema is not None:
                self.model.ema.ema_model.save(g_ema_path)
            print(f"[Checkpoint] Saved G -> {g_path}")
            print(f"[Checkpoint] Saved D -> {d_path}")
            if self.model.ema is not None:
                print(f"[Checkpoint] Saved G_EMA -> {g_ema_path}")

# ========== Training ==========
gan = XRayGAN(generator, discriminator, zdim=Z_DIM, ema_decay=EMA_DECAY)
gan.compile(g_opt, d_opt, bce)
monitor = FineTuneMonitor(gan, zdim=Z_DIM, start_epoch=START_EPOCH,
                          save_dir="generated_ft", save_every=SAVE_IMG_EVERY, n_samples=10)
gan.fit(ds, initial_epoch=START_EPOCH, epochs=TOTAL_EPOCHS, callbacks=[monitor], verbose=1)

# ========== Final save ==========
generator.save("checkpoints_ft/generator_final_ft.h5")
discriminator.save("checkpoints_ft/discriminator_final_ft.h5")
if gan.ema is not None:
    gan.ema.ema_model.save("checkpoints_ft/generator_final_ft_EMA.h5")
print("[✓] Final fine-tuned models saved.")
