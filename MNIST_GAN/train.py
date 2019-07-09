import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist

from models import Models

model = Models()
discriminator = model.discriminator()
generator = model.generator()
gan = model.gan(generator, discriminator)


def get_dataset():
    # Get MNIST dataset
    (train_X, _), (_, _) = mnist.load_data()
    # Expand for grayscale channel
    x = np.expand_dims(train_X, axis=-1)
    # Change to floats
    x = x.astype('float32')
    # Scale
    x = x / 255.0
    return x


def real_samples(dataset, n):
    # Random samples
    index = np.random.randint(0, dataset.shape[0], n)
    # Get selected images
    x = dataset[index]
    # Ground truth
    y = np.ones((n, 1))
    return x, y

def generate_latent_points(latent_dim, n):
    # Random points in the latent space
    x = np.random.randn(latent_dim * n)
    # Reshape
    x = x.reshape(n, latent_dim)
    return x


def generate_fake_samples(generator, latent_dim, n):
    # Generate latent points
    x = generate_latent_points(latent_dim, n)
    # Generate fake samples using the generator
    x = generator.predict(x)
    # Ground truth
    y = np.zeros((n, 1))
    return x, y

def save_plot(examples, epoch, n = 4):
    for i in range(n * 4):
        plt.subplot(4, 4, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    filename = 'images/generated_plot_e%03d.png' % (epoch)
    plt.savefig(filename)
    plt.close()


def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n=100):
    # Get real samples
    x_real, y_real = real_samples(dataset, n)
    # Evaluate discriminator on real samples
    _, acc_real = discriminator.evaluate(x_real, y_real)
    # Generate fake samples
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # Evaluate discriminator on fake samples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake)
    # Print results
    print("Accuracy Real: %.0f%%, Accuracy Fake: %0.f%%" %
          (acc_real * 100, acc_fake * 100))
    # Save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model/generator_model_%03d.h5' % (epoch)
    generator.save(filename)


def train(generator, discriminator, gan, dataset, latent_dim, epochs=100, batch_size=256):
    batch_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    for epoch in range(epochs):
        for batch in range(batch_per_epoch):
            # Get REAL samples
            x_real, y_real = real_samples(dataset, half_batch)
            # Get FAKE samples
            x_fake, y_fake = generate_fake_samples(
                generator, latent_dim, half_batch)
            # Combine training set for discriminator
            x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            # Get discriminator loss
            d_loss, _ = discriminator.train_on_batch(x, y)
            # Get LATENT samples
            x_latent = generate_latent_points(latent_dim, batch_size)
            # Get LATENT labels
            y_latent = np.ones((batch_size, 1))
            # Update generator
            g_loss = gan.train_on_batch(x_latent, y_latent)
        if epoch % 10 == 0:
            summarize_performance(epoch, generator, discriminator, dataset, latent_dim)

latent_dim = 100
dataset = get_dataset()
train(generator, discriminator, gan, dataset, latent_dim)
