from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LeakyReLU, Dropout, Reshape, Conv2DTranspose
from keras.optimizers import Adam

class Models():

    def __init__(self):
        return

    def generator(self):
        model = Sequential()
        model.add(Dense(128 * 7 * 7, input_dim=100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        # Upsample to 14 x 14
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        # This is reported as best practice for GAN models
        model.add(LeakyReLU(alpha=0.2))
        # Upsample to 28 x 28
        model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
        return model

    def discriminator(self, input_shape=(28, 28, 1)):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), strides=(2, 2),
                         padding='same', input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3, 3), strides=(2, 2),
                         padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # Compile the model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy',
                      optimizer=opt, metrics=['accuracy'])
        return model

    def gan(self, generator, discriminator):
        discriminator.trainable = False
        model = Sequential()
        # Add the generator
        model.add(generator)
        # Add the discriminator
        model.add(discriminator)
        # Compile
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
