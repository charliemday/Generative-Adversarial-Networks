from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, Dense
from keras.layers import LeakyReLU, Input, Dropout
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import Adam
import numpy as np

class Models():

    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 1000

    def discriminator(self):
        model = Sequential()
        # model.add(CuDNNLSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        # model.add(Bidirectional(CuDNNLSTM(512)))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        # Compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
        return model

    def generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))
        # model.summary()
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
