from keras.models import Sequential
from keras.layers import Reshape, Dense, Conv2D, LeakyReLU, Flatten, Dropout, Conv2DTranspose
from keras.optimizers import Adam

import imageio
import glob

class Models():

    def generator(self, latent_dim):
        model = Sequential()
        # 4x4 image
        nodes = 256 * 4 * 4
        model.add(Dense(nodes, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        # upsample 8x8 image
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 16x16 image
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 32x32 image
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # output layer
        model.add(Conv2D(3, (3, 3), activation="tanh", padding="same"))
        return model

    def discriminator(self, input_shape=(32, 32, 3)):
        model = Sequential()
        # Normal
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.2))
        # Downsample
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # Downsample
        model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # Downsample
        model.add(Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        # Classifier (Normal NN)
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation="sigmoid"))
        # Compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        return model

    def gan(self, generator, discriminator):
        model = Sequential()
        discriminator.trainable = False
        # Add generator
        model.add(generator)
        # Add discriminator
        model.add(discriminator)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        return model
