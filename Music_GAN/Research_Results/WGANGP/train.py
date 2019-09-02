# WGAN model

# Import necessary libraries
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, Dense
from keras.layers import LeakyReLU, Input, Dropout
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import RMSprop
from keras import backend

import numpy as np
import datetime
from music21 import stream, note, instrument, chord, duration
import matplotlib.pyplot as plt
import pickle
import time
import math

class Models():

    def __init__(self, w_loss):
        self.seq_length = 100
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 100
        self.critic_loss_function = w_loss
        self.gan_loss_function = w_loss

    def critic(self):
        model = Sequential()

        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))

        model.add(Dense(512))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        opt = RMSprop(lr=0.00005)
        model.compile(loss=self.critic_loss_function,
                      optimizer=opt, metrics=['accuracy'])
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

        model.summary()

        return model

    def gan(self, generator, critic):
        model = Sequential()
        critic.trainable = False
        # Add generator
        model.add(generator)
        # Add discriminator
        model.add(critic)
        model.summary()
        opt = RMSprop(lr=0.00005)
        model.compile(loss=self.gan_loss_function, optimizer=opt)
        return model

# Generate REAL samples for the Discriminator


def generate_real_samples(notes, batch_size):
    # Random n samples
    index = np.random.randint(0, notes.shape[0], batch_size)
    # Get indexed image
    x = notes[index]
    # Ground truth
    y = -np.ones((batch_size, 1))
    return x, y

# Generate latent samples for the input of the Generator


def generate_latent_points(latent_dim, batch_size):
    # Random points
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    # Reshape (may not be needed)
    noise = noise.reshape(batch_size, latent_dim)
    return noise

# Use the generator to create some FAKE samples


def generate_fake_samples(generator, latent_dim, n):
    # Random latent space samples
    x = generate_latent_points(latent_dim, n)
    # Input into generator
    x = generator.predict(x)
    # Ground truth
    y = np.ones((n, 1))
    return x, y

# Create some samples


def create_midi(prediction_output, filename):

    # Initiate the offset
    offset = 0
    output_notes = []

    # Create note and chord objects
    for item in prediction_output:

        try:
            pattern = item.split()[0]
            pattern_duration = item.split()[1]
        except:
            print("There was an error!")

        # Check if the pattern duration is a float (e.g. 0.5)
        try:
            pattern_duration = float(pattern_duration)
        # Else deal with fractions (e.g. 1/3 becomes 0.3)
        except:
            fraction = pattern_duration.split("/")
            numerator = float(fraction[0])
            denominator = float(fraction[1])
            pattern_duration = numerator / denominator

        # If pattern is a chord
        if ('.' in pattern) or pattern.isdigit():

            # Turn the duration into a Duration Object
            dur_object = duration.Duration(pattern_duration)
            # dur_object.quarterLength = pattern_duration

            # Split the chord up into notes
            notes_in_chord = pattern.split('.')
            notes = []
            # Loop through chord notes
            for current_note in notes_in_chord:
                # Turn integer into note
                new_note = note.Note(int(current_note))
                new_note.sortedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes, duration=dur_object)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Else if pattern is a note
        else:
            new_note = note.Note(pattern, quarterLength=pattern_duration)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration so that notes don't stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write("midi", fp="samples/WGANGP/{}.mid".format(filename))

# Main training function
def train(gan, generator, critic, dataset, notes, batch_size=128, epochs=1000, sample_interval=50):

    # Define clip value
    clip_value = 0.1

    # Disc and Gen loss lists for graph plot
    critic_loss_real = []
    critic_loss_fake = []
    gen_loss = []

    # Disc accuracy lists for graph plot
    critic_acc_real = []
    critic_acc_fake = []

    # Ground truths
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    dummy = np.zeros((batch_size, 1))  # This is for the Gradient Penalty

    latent_dim = 100

    # Start a timer
    start = time.time()

    # --- WGANGP additions --- #

    # Construct weighted average between real and fake images
    interpolated_img = RandomWeightedAverage()([real_img, fake_img])

    # Loop through the epochs
    for epoch in range(epochs):
        print("[INFO] Running Epoch %s" % str(epoch))

        # Generate latent points
        noise = generate_latent_points(latent_dim, batch_size)

        # -------------------------- #
        # Critic Training
        # -------------------------- #

        # Get real samples
        x_real, y_real = generate_real_samples(dataset, batch_size)

        # Update discriminator relative to REAL
        # c_loss_real, c_acc_real = critic.train_on_batch(x_real, y_real)
        c_loss_real, c_acc_real = critic.train_on_batch(
            [x_real, y_real, noise], [real, fake, dummy])

        # Get fake samples
        x_fake, y_fake = generate_fake_samples(
            generator, latent_dim, batch_size)

        # Update discriminator relative to FAKE
        c_loss_fake, c_acc_fake = critic.train_on_batch(x_fake, y_fake)

        # Get the total discriminator loss
        c_loss_total = 0.5 * np.add(c_loss_fake, c_loss_real)

        # Clip the critic weights
        for layer in critic.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            layer.set_weights(weights)

        # -------------------------- #
        # Generator Training
        # -------------------------- #

        # Train GAN model (only generator)
        g_loss = gan.train_on_batch(noise, y_real)

        # Print progress
        print('>%d, d_real=%.3f, d_fake=%.3f g_loss=%.3f' %
              (epoch + 1, c_loss_real, c_loss_fake, g_loss))

        epoch_time = time.time() - start
        if epoch == 0:
            time_per_epoch = epoch_time
        else:
            time_per_epoch = epoch_time / epoch

        # Print time of epoch
        print("[INFO] Epoch time: {:1.2f} seconds".format(epoch_time))

        # Estimate completion time
        print("[INFO] Estimated completion time: %s seconds" %
              str(datetime.timedelta(seconds=math.floor(time_per_epoch * (epochs - epoch)))))

        critic_loss_real.append(c_loss_real)
        critic_loss_fake.append(c_loss_fake)
        gen_loss.append(g_loss)

        critic_acc_real.append(c_acc_real)
        critic_acc_fake.append(c_acc_fake)

        if epoch % sample_interval == 0:
            summarize_performance(epoch, notes, generator)

    print("-------------------------")
    print("[INFO] Training complete.")
    print("[INFO] Total time: %s" %
          str(datetime.timedelta(seconds=math.floor(time.time() - start))))

    print("Dis Loss Real", critic_loss_real)
    print("Dis Loss Fake", critic_loss_fake)
    print("Gen Loss", gen_loss)
    print("Dis Acc Real", critic_acc_real)
    print("Dis Acc Fake", critic_acc_fake)

    plot_progress(critic_loss_real, critic_loss_fake, gen_loss,
                  critic_acc_real, critic_acc_fake, epochs)


class WGANGP():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # -------------------------------
        # Construct Computational Graph
        #       for the Critic
        # -------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=optimizer,
                                  loss_weights=[1, 1, 10])
        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(
            loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
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

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):
        model = Sequential()

        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))

        model.add(Dense(512))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        opt = RMSprop(lr=0.00005)
        model.compile(loss=self.critic_loss_function,
                      optimizer=opt, metrics=['accuracy'])

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def plot_progress(self, disc_loss_real, disc_loss_fake, gen_loss, disc_acc_real, disc_acc_fake, epochs=100):

        # plot loss
        plt.figure()
        plt.plot(disc_loss_real, label='d-real')
        plt.plot(disc_loss_fake, label='d-fake')
        plt.plot(gen_loss, label='gen')
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("GAN_Loss_per_Epoch.png")
        plt.show()
        plt.close()

        # plot accuracy
        plt.figure()
        plt.plot(disc_acc_real, label='acc-real')
        plt.plot(disc_acc_fake, label='acc-fake')
        plt.legend()
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig('GAN_Accuracy_per_Epoch.png')
        plt.show()
        plt.close()

    def summarize_performance(epoch, input_notes, generator, latent_dim=100):

        # Get notes
        notes = input_notes

        # Sort the notes into unique elements
        pitchnames = sorted(set(item for item in notes))

        # Enumerate the notes into a dictionary
        int_to_note = dict((number, note)
                           for number, note in enumerate(pitchnames))

        # Create random noise for the Generator input
        noise = np.random.normal(0, 1, (1, latent_dim))

        # Predict using the Generator
        predictions = generator.predict(noise)

        # Get the normalization range for the results between -1 and 1
        norm_range = len(int_to_note) / 2

        # Convert predicted notes to index (normalize) - Some problem here...?
        try:
            print("Predictions (Working)", predictions)
            pred_notes = [(x * norm_range + norm_range) for x in predictions[0]]
            pred_notes = [int_to_note[int(x)] for x in pred_notes]
            print("Processed Pred Notes", pred_notes)

            # Save the generator weights
            filename = "weights/WGANGP/generator_model_%03d.h5" % (epoch)
            generator.save(filename)

            filename = "sample_epoch_%s" % str(epoch)

            # Create a midi file
            create_midi(pred_notes, filename)

        except:
            print("There was an error in the pred notes!")
            print("Predictions (Error)", predictions)
            print("Pred Notes", pred_notes)
            filename = "weights/WGANGP/generator_model(ERROR)_%03d.h5" % (epoch)
            generator.save(filename)

    def train(self, dataset, notes, epochs, batch_size, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                          [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)


dataset_file = open("../pickled_file_data.pkl", "rb")
dataset = pickle.load(dataset_file)
dataset_file.close()

notes_file = open("../pickled_file_notes.pkl", "rb")
notes = pickle.load(notes_file)
notes_file.close()

models = Models(wasserstein_loss)
critic = models.critic()
generator = models.generator()
gan = models.gan(generator, critic)
train(gan, generator, critic, dataset, notes)
