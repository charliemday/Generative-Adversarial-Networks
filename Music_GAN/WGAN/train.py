import time
import math
import datetime
import numpy as np
from music21 import stream
import matplotlib.pyplot as plt

from keras.datasets.mnist import load_data
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from matplotlib import pyplot


# clip model weights to a given hypercube
class ClipConstraint():
	# set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

	# clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

class Models():

    def __init__(self, wasserstein_loss):
        self.w_loss = wasserstein_loss

    # define the standalone critic model
    def critic(self, in_shape=(28,28,1)):
    	# weight initialization
    	init = RandomNormal(stddev=0.02)
    	# weight constraint
    	const = ClipConstraint(0.01)
    	# define model
    	model = Sequential()
    	# downsample to 14x14
    	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=in_shape))
    	model.add(BatchNormalization())
    	model.add(LeakyReLU(alpha=0.2))
    	# downsample to 7x7
    	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
    	model.add(BatchNormalization())
    	model.add(LeakyReLU(alpha=0.2))
    	# scoring, linear activation
    	model.add(Flatten())
    	model.add(Dense(1))
    	# compile model
    	opt = RMSprop(lr=0.00005)
    	model.compile(loss=self.w_loss, optimizer=opt)
    	return model

    # define the standalone generator model
    def generator(self, latent_dim = 100):
    	# weight initialization
    	init = RandomNormal(stddev=0.02)
    	# define model
    	model = Sequential()
    	# foundation for 7x7 image
    	n_nodes = 128 * 7 * 7
    	model.add(Dense(n_nodes, kernel_initializer=init, input_dim=latent_dim))
    	model.add(LeakyReLU(alpha=0.2))
    	model.add(Reshape((7, 7, 128)))
    	# upsample to 14x14
    	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    	model.add(BatchNormalization())
    	model.add(LeakyReLU(alpha=0.2))
    	# upsample to 28x28
    	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    	model.add(BatchNormalization())
    	model.add(LeakyReLU(alpha=0.2))
    	# output 28x28x1
    	model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
    	return model

    def gan(self, generator, critic):
        model = Sequential()
        critic.trainable = False
        # Add generator
        model.add(generator)
        # Add discriminator
        model.add(critic)
        # Use RMSprop
        opt = RMSprop(lr=0.00005)
        model.compile(loss=self.w_loss, optimizer=opt)
        return model


def normalize(n, max_n=1.0, min_n=-1.0, max_z=114):
    # Scale between -1 and +1
    x = (x - 127.5) / 127.5
    return z


def load_notes(filepath="data"):
    # Load dataset
    notes = get_notes(filepath)
    # Preprocess notes (this is an imported function)
    x_train, _ = prepare_sequences(notes)
    return x_train, notes


def generate_real_samples(notes, batch_size):
    # Random n samples
    index = np.random.randint(0, notes.shape[0], batch_size)
    # Get indexed image
    x = notes[index]
    # Ground truth
    y = -np.ones((batch_size, 1))
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
    y = np.ones((n, 1))
    return x, y


def create_midi(prediction_output, filename):
    offset = 0
    output_notes = []

    # Create note and chord objects
    for item in prediction_output:
        pattern = str(item[0])
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
    midi_stream.write("midi", fp="samples/{}.mid".format(filename))


def summarize_performance(epoch, input_notes, generator, latent_dim=1000):
    # Get notes
    notes = input_notes

    # Get pitchnames
    # Could this be an ordered list?
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
    pred_notes = [(x * norm_range + norm_range) for x in predictions[0]]

    pred_notes = [int_to_note[int(x)] for x in pred_notes]
    # pred_notes = [x for x in pred_notes]

    # Save the generator weights
    filename = "weights/generator_model_%03d.h5" % (epoch)
    generator.save(filename)

    filename = "sample_epoch_%s" % str(epoch)

    # Create a midi file
    create_midi(pred_notes, filename)


def plot_progress(disc_loss_real, disc_loss_fake, gen_loss, epochs=100):
    plt.plot(disc_loss_real, c='red')
    plt.plot(disc_loss_fake, c='orange')
    plt.plot(gen_loss, c='blue')
    plt.title("GAN Loss per Epoch")
    plt.legend(['critic (Real)', 'critic (Fake)', 'Generator'])
    plt.xlim(0, epochs)
    plt.ylim(0, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
    plt.show()
    plt.close()


def train(self, epochs, batch_size=64, sample_interval=50, n_critic=5):

    # Disc and Gen loss lists for graph plot
    disc_loss_real = []
    disc_loss_fake = []
    gen_loss = []

    # Get dataset
    dataset = self.get_dataset()

    batch_per_epoch = int(dataset.shape[0] / batch_size)

    # Number of training iterations
    n_steps = batch_per_epoch * epochs

    n_steps = bat_per_epo * n_epochs

    # Get half batch
    half_batch = int(batch_size / 2)

    print("[INFO] Starting WGAN training...")

    for epoch in range(n_steps):
        print("--- Epoch", epoch,"---")

        # Update the critic n times more than the generator
        for _ in range(n_critic):

            # --- Train the Critic (Discriminator) --- #

            # Generate real samples
            x_real, y_real = self.real_samples(dataset, half_batch)

            # Generate some fake samples
            x_fake, y_fake = self.generate_fake_samples(half_batch)

            # Train critic
            d_loss_real = self.critic.train_on_batch(x_real, y_real)
            d_loss_fake = self.critic.train_on_batch(x_fake, y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # --- Train the Generator --- #

        # Generator latent samples
        x_latent, y_latent = self.generate_latent_points(self.latent_dim, batch_size)

        # Update Generator
        g_loss = self.gan.train_on_batch(x_latent, y_latent)

        if epoch % sample_interval == 0:
            self.summarize_performance(epoch, dataset)

        disc_loss_real.append(d_loss_real)
        disc_loss_fake.append(d_loss_fake)
        gen_loss.append(g_loss)

    print("--- [INFO] Training Complete ---")
    self.plot_progress(disc_loss_real, disc_loss_fake, gen_loss, epochs)


dataset, notes = load_notes("data_2.0")
models = Models(wasserstein_loss)

critic = models.critic()
generator = models.generator()
gan = models.gan(generator, critic)

train(gan, generator, critic, dataset, notes)
