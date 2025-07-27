from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dropout

def build_generator():
    model = Sequential()

    # Latent dim = 256 → 4x4x1024
    model.add(Dense(4 * 4 * 1024, input_dim=256))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4, 4, 1024)))

    # 8x8
    model.add(UpSampling2D())
    model.add(Conv2D(512, kernel_size=7, padding='same'))  # širi kernel
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1, noise_shape=(None, 1, 1, 1)))

    # 16x16
    model.add(UpSampling2D())
    model.add(Conv2D(512, kernel_size=7, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1, noise_shape=(None, 1, 1, 1)))

    # 32x32
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=5, padding='same'))  # prelazni nivo
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1, noise_shape=(None, 1, 1, 1)))

    # 64x64
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1, noise_shape=(None, 1, 1, 1)))

    # 128x128
    model.add(UpSampling2D())
    model.add(Conv2D(192, kernel_size=5, padding='same'))  # više kanala
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1, noise_shape=(None, 1, 1, 1)))

    # 256x256
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1, noise_shape=(None, 1, 1, 1)))

    # 512x512
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding='same'))  # manji kernel za fine detalje
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.1, noise_shape=(None, 1, 1, 1)))

    # Output layer
    model.add(Conv2D(1, kernel_size=3, padding='same', activation='tanh'))

    return model
