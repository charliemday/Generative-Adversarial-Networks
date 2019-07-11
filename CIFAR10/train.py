from models import Models

from keras.datasets.cifar10 import load_data

import numpy as np
import matplotlib.pyplot as plt


def load_cifar():
    # Load dataset
    (train_x, _), (_, _) = load_data()
    # Convert to floats
    x = train_x.astype('float32')
    # Scale between -1 and 1
    x = (x - 127.5) / 127.5
    return x


def generate_real_samples(dataset, n):
    # Random n samples
    index = np.random.randint(0, dataset.shape[0], n)
    # Get indexed image
    x = dataset[index]
    # Ground truth
    y = np.ones((n, 1))
    return x, y

def generate_latent_points(latent_dim, n):
    # Random points (Gaussian Normal Distribution)
    x = np.random.randn(latent_dim * n)
    # Reshape
    x = x.reshape(n, latent_dim)
    return x


def generate_fake_samples(generator, latent_dim, n):
    # Random latent space samples
    x = generate_latent_points(latent_dim, n)
    # Input into generator
    x = generator.predict(x)
    # Ground truth
    y = np.zeros((n, 1))
    return x, y

def save_plot(examples, epoch, n = 4):
    # Scale samples
    examples = (examples + 1) / 2.0
    # Plot images
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    # Save the plot
    filename = "images/generated_plot_epoch_%03d.png" % (epoch + 1)
    plt.savefig(filename)
    plt.close()


def summarise_performance(epoch, generator, discriminator, dataset, latent_dim, n=150):
    # Get real samples
    x_real, y_real = generate_real_samples(dataset, n)
    # Evaluate discriminator
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # Get fake samples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # Evaluate discriminator
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    # Summarise performance
    print("Accuracy Real: %.0f%%, Accuracy Fake: %0.f%%" %
          (acc_real * 100, acc_fake * 100))
    # Save plot
    save_plot(x_fake, epoch)
    # Save generator
    filename = 'weights/generator_model_%03d.h5' % (epoch + 1)
    generator.save(filename)



def train(generator, discriminator, gan, dataset, latent_dim, epochs=200, batch_size=128):
    # Split batches up
    batch_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    print("[INFO] Training starting...")
    # Loop through epochs
    for epoch in range(epochs):
        for batch in range(batch_per_epoch):
            # Get real samples
            x_real, y_real = generate_real_samples(dataset, half_batch)
            # Update discriminator relative to REAL
            d_loss_real, _ = discriminator.train_on_batch(x_real, y_real)
            # Get fake samples
            x_fake, y_fake = generate_fake_samples(
                generator, latent_dim, half_batch)
            # Update discriminator relative to FAKE
            d_loss_fake, _ = discriminator.train_on_batch(x_fake, y_fake)
            # Generate latent points
            x_latent = generate_latent_points(latent_dim, batch_size)
            # Ground truth
            y_latent = np.ones((batch_size, 1))
            # Train GAN model (only generator)
            g_loss = gan.train_on_batch(x_latent, y_latent)
            # Summarise the loss
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (epoch + 1, batch +
                                                           1, batch_per_epoch, d_loss_real, d_loss_fake, g_loss))

        if epoch % 10 == 0:
            summarise_performance(
                epoch, generator, discriminator, dataset, latent_dim)


dataset = load_cifar()
models = Models()
latent_dim = 100
discriminator = models.discriminator()
generator = models.generator(latent_dim)
gan = models.gan(generator, discriminator)

train(generator, discriminator, gan, dataset, latent_dim)
