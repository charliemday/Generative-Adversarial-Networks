from keras.models import Sequential
from keras.layers import Dense
import numpy as np


class Models():
    def __init__(self):
        return

    # --- Discriminator Model --- #
    def discriminator(self, n=2):
        model = Sequential()
        model.add(Dense(25, activation="relu", input_dim=n))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy",
                      optimizer="adam", metrics=["accuracy"])

        return model

    # --- Generator Model --- #
    def generator(self, latent_dim, n=2):
        model = Sequential()
        model.add(Dense(15, activation="relu", input_dim=latent_dim))
        model.add(Dense(n, activation="linear"))
        return model

    # --- GAN model --- #
    def gan(self, generator, discriminator):
        discriminator.trainable = False
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss="binary_crossentropy", optimizer="adam")
        return model
