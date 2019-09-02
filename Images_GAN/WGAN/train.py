# from models import Models
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras import backend
from keras.models import Model, Sequential
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

class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

	# clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

	# get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

class Models():

    def critic(self, wasserstein_loss, img_shape = (28,28,1)):
    	# weight initialization
    	init = RandomNormal(stddev=0.02)
    	# weight constraint
    	const = ClipConstraint(0.01)
    	# define model
    	model = Sequential()
    	# downsample to 14x14
    	model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=img_shape))
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
    	model.compile(loss=wasserstein_loss, optimizer=opt)
    	return model

    def generator(self, latent_dim, channels):
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

    def gan(self, generator, critic, wasserstein_loss):
    	# make weights in the critic not trainable
    	critic.trainable = False
    	# connect them
    	model = Sequential()
    	# add generator
    	model.add(generator)
    	# add the critic
    	model.add(critic)
    	# compile model
    	opt = RMSprop(lr=0.00005)
    	model.compile(loss=wasserstein_loss, optimizer=opt)
    	return model

class WGAN():

    def __init__(self):

        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.n_critic = 5
        self.clip_value = 0.01

        self.latent_dim = 100

        self.critic = Models().critic(self.wasserstein_loss, self.img_shape)
        self.generator = Models().generator(self.latent_dim, self.channels)
        self.gan = Models().gan(self.generator, self.critic, self.wasserstein_loss)


    def wasserstein_loss(self, y_true, y_pred):
        return backend.mean(y_true * y_pred)

    def get_dataset(self):
        # Get MNIST dataset
        (train_X, _), (_, _) = mnist.load_data()
        # Expand for grayscale channel
        x = np.expand_dims(train_X, axis=-1)
        # Change to floats
        x = x.astype('float32')
        # Scale between -1 and +1
        x = (x - 127.5) / 127.5
        return x

    def real_samples(self, dataset, n):
        # Random real samples
        index = np.random.randint(0, dataset.shape[0], n)
        # Get selected images
        x = dataset[index]
        # Ground truth (-1 for real)
        y = -np.ones((n, 1))
        return x, y

    def generate_latent_points(self, latent_dim, n):
        # Random points in the latent space
        x = np.random.randn(self.latent_dim * n)
        # Reshape
        x = x.reshape(n, self.latent_dim)
        # Ground truth
        y = -np.ones((n, 1))
        return x, y

    def generate_fake_samples(self, n):
        # Generate latent points
        x, y = self.generate_latent_points(self.latent_dim, n)
        # Generate fake samples using the generator
        x = self.generator.predict(x)
        # Ground truth (+1 for fake)
        y = np.ones((n, 1))
        return x, y

    def generate_plot(self, examples, epoch, n = 4):
        for i in range(n * 4):
            plt.subplot(4, 4, 1 + i)
            plt.axis('off')
            plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        filename = 'samples/generated_plot_e%03d.png' % (epoch)
        plt.savefig(filename)
        plt.close()

    def plot_progress(self, disc_loss_real, disc_loss_fake, gen_loss, epochs = 4000):
        plt.plot(disc_loss_real, c='red')
        plt.plot(disc_loss_fake, c='orange')
        plt.plot(gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Critic (Real)', 'Critic (Fake)', 'Generator'])
        plt.xlim(0, epochs)
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.show()
        plt.close()

    def summarize_performance(self, epoch, dataset, n = 100):
        # Generate fake samples
        x_fake, y_fake = self.generate_fake_samples(n)
        # Rescale back to [0,1]
        x_fake = (x_fake + 1) / 2.0
        # Generate some samples
        self.generate_plot(x_fake, epoch)
        # Save the generator model
        filename = 'weights/critic_model_%03d.h5' % (epoch)
        self.critic.save(filename)

    def train(self, epochs, batch_size=64, sample_interval=50):

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
            for _ in range(self.n_critic):

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

wgan = WGAN()
wgan.train(epochs = 1000)
