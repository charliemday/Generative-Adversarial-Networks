# Copy and paste into Google Colab Cell
# Import necessary libraries
from keras.models import Sequential, Model
from keras.layers import LSTM, Bidirectional, Dense
from keras.layers import LeakyReLU, Input, Dropout
from keras.layers import BatchNormalization, Reshape
from keras.optimizers import Adam

import numpy as np
import datetime
from music21 import stream
import matplotlib.pyplot as plt
import pickle
import time
import math

class Models():

    def __init__(self):
        self.seq_length = 100
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 100
        self.dis_loss_function = "binary_crossentropy"
        self.gan_loss_function = "binary_crossentropy"

    def discriminator(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(512, input_shape=self.seq_shape)))

        model.add(Dense(512))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(64))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=self.dis_loss_function, optimizer=opt, metrics=['accuracy'])
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

        return model

    def gan(self, generator, discriminator):
        model = Sequential()
        discriminator.trainable = False
        # Add generator
        model.add(generator)
        # Add discriminator
        model.add(discriminator)
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=self.gan_loss_function, optimizer=opt)
        return model

def normalize(n, max_n=1.0, min_n=-1.0, max_z=114):
    # Normalize between 0 and 1
    z = (n - min_n / (max_n - min_n))
    z = z * max_z
    return z

def load_notes(filepath="data"):
    # Load dataset
    notes = get_notes(filepath)
    # Preprocess notes
    x_train, y_train = prepare_sequences(notes)
    return x_train, notes


def generate_real_samples(notes, batch_size):
    # Random n samples
    index = np.random.randint(0, notes.shape[0], batch_size)
    # Get indexed image
    x = notes[index]
    # Ground truth
    y = np.ones((batch_size, 1))
    return x, y


def generate_latent_points(latent_dim, batch_size):
    # Random points
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    # Reshape (may not be needed)
    noise = noise.reshape(batch_size, latent_dim)
    return noise


def generate_fake_samples(generator, latent_dim, n):
    # Random latent space samples
    x = generate_latent_points(latent_dim, n)
    # Input into generator
    x = generator.predict(x)
    # Ground truth
    y = np.zeros((n, 1))
    return x, y


def create_midi(prediction_output, filename):
    offset = 0
    output_notes = []

    # Create note and chord objects
    for item in prediction_output:
        pattern = str(item[0])
        pattern = item.split()[0]
        pattern_duration = item.split()[1]
        # If pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.sortedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Else if pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration so that notes don't stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="samples/21_Aug/{}.mid".format(filename))


def summarize_performance(epoch, input_notes, generator, latent_dim=100):
    # Get notes
    notes = input_notes

    # Sort the notes
    pitchnames = sorted(set(item for item in notes))

    # Integer to note (Note to integer?)
    int_to_note = dict((number, note)
                       for number, note in enumerate(pitchnames))

    # Create random noise
    noise = np.random.normal(0, 1, (1, latent_dim))

    # Predict using generator
    predictions = generator.predict(noise)

    # Get the normalization range for between -1 and 1
    norm_range = len(int_to_note) / 2

    # Convert predicted notes to index (normalize)
    try:
      pred_notes = [(x * norm_range + norm_range) for x in predictions[0]]

      pred_notes = [int_to_note[int(x)] for x in pred_notes]
      # pred_notes = [x for x in pred_notes]
    except:
      print("There was an error in the pred notes!")
      exit()

    # Save the generator weights
    filename = "weights/21_Aug/generator_model_%03d.h5" % (epoch)
    generator.save(filename)

    filename = "sample_epoch_%s" % str(epoch)

    # Create a midi file
    create_midi(pred_notes, filename)


def plot_progress(disc_loss_real, disc_loss_fake, gen_loss, epochs = 100):
    plt.plot(disc_loss_real, c='red')
    plt.plot(disc_loss_fake, c='orange')
    plt.plot(gen_loss, c='blue')
    plt.title("GAN Loss per Epoch")
    plt.legend(['Discriminator (Real)', 'Discriminator (Fake)', 'Generator'])
    plt.xlim(0, epochs)
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('GAN_Loss_per_Epoch_21_Aug.png', transparent=True)
    plt.show()
    plt.close()


#     1000 epochs and generator/discriminator appear to be in equilibrium (~20 mins)
def train(gan, generator, discriminator, dataset, notes, batch_size=128, epochs=1000, sample_interval=50):

    # Disc and Gen loss lists for graph plot
    disc_loss_real = []
    disc_loss_fake = []
    gen_loss = []

    # Ground truths
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # latent_dim = 1000
    latent_dim = 100

    # Start a timer
    start = time.time()


    # Loop through the epochs
    for epoch in range(epochs):
        print("[INFO] Running Epoch %s" % str(epoch))

        # Get real samples
        x_real, y_real = generate_real_samples(dataset, batch_size)

        # Update discriminator relative to REAL
        d_loss_real, _ = discriminator.train_on_batch(x_real, y_real)

        # Get fake samples
        x_fake, y_fake = generate_fake_samples(
            generator, latent_dim, batch_size)

        # Update discriminator relative to FAKE
        d_loss_fake, _ = discriminator.train_on_batch(x_fake, y_fake)

        # Get the total discriminator loss
        d_loss_total = 0.5 * np.add(d_loss_fake, d_loss_real)

        # Generate latent points
        noise = generate_latent_points(latent_dim, batch_size)

        # Train GAN model (only generator)
        g_loss = gan.train_on_batch(noise, y_real)

        # Print progress
        print('>%d, d_real=%.3f, d_fake=%.3f g_loss=%.3f' %
              (epoch + 1, d_loss_real, d_loss_fake, g_loss))

        epoch_time = time.time() - start
        if epoch == 0:
            time_per_epoch = epoch_time
        else:
            time_per_epoch = epoch_time / epoch

        # Print time of epoch
        print("[INFO] Epoch time: {:1.2f} seconds".format(epoch_time))

        # Estimate completion time
        print("[INFO] Estimated completion time: %s seconds" %
              str(datetime.timedelta(seconds = math.floor(time_per_epoch * (epochs - epoch)))))

        disc_loss_real.append(d_loss_real)
        disc_loss_fake.append(d_loss_fake)
        gen_loss.append(g_loss)

        if epoch % sample_interval == 0:
            summarize_performance(epoch, notes, generator)

    print("-------------------------")
    print("[INFO] Training complete.")
    print("[INFO] Total time: %s" % str(datetime.timedelta(seconds = math.floor(time.time() - start))))


    plot_progress(disc_loss_real, disc_loss_fake, gen_loss, epochs)





# dataset_file = open("preprocessed_data(note_duration).pkl", "rb")
# dataset = pickle.load(dataset_file)
# dataset_file.close()
#
# notes_file = open("preprocessed_notes(note_duration).pkl", "rb")
# notes = pickle.load(notes_file)
# notes_file.close()

models = Models()
discriminator = models.discriminator()
generator = models.generator()
gan = models.gan(generator, discriminator)

from keras.utils.vis_utils import plot_model

plot_model(discriminator, to_file="discriminator.png")

exit()


# train(gan, generator, discriminator, dataset, notes)
