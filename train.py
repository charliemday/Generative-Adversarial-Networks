# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from models import Models
import math

# Define the latent space
latent_dim = 5

# This gets all the models from the models import
models = Models()
discriminator = models.discriminator()
generator = models.generator(latent_dim)
gan = models.gan(generator, discriminator)


# This creates the initial dataset
def generate_real_samples(n):
    inputs = np.random.rand(n) - 0.5
    outputs = inputs ** 2
    inputs = inputs.reshape(n, 1)
    outputs = outputs.reshape(n, 1)
    x = np.hstack((inputs, outputs))
    y = np.ones((n, 1))
    return x, y

# Generate the latent points for the generator 
def generate_latent_points(latent_dim, n):
    # Use the normal distribution to randomly create points
    x = np.random.randn(latent_dim * n)
    # Reshape the points to be dimensionally correct
    x = x.reshape(n, latent_dim)
    return x

# Use the generator to make n fake samples
def generate_fake_samples(generator, latent_dim, n):
    # Generate points
    x = generate_latent_points(latent_dim, n)
    # Predict outputs
    x = generator.predict(x)
    # Ground truth
    y = np.ones((n, 1))
    return x, y


def evaluate_gan(epoch, gen, dis, latent_dim, n=100):
    # Get real random samples
    x_real, y_real = generate_real_samples(n)
    # Get real accuracy (from discriminator)
    _, acc_real = dis.evaluate(x_real, y_real, verbose=0)
    # Get fake samples (from generator)
    x_fake, y_fake = generate_fake_samples(gen, latent_dim, n)
    # Get fake accuracy (from discriminator)
    _, acc_fake = dis.evaluate(x_fake, y_fake, verbose=0)
    print("[INFO] Epoch : %s" % str(epoch))
    print("[INFO] Real accuracy: %s percent" % str(acc_real * 100))
    print("[INFO] Fake accuracy: %s percent" % str(acc_fake * 100))
    print("[INFO] Saving graph...")
    # Save the graph
    plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    plt.savefig("images/epoch_%s" % str(epoch))
    plt.close()


def train(gen, dis, gan, latent_dim, epochs=30000, batch_size=128, eval=1000):
    # Split the batch in two (half for generator, half for discriminator)
    half_batch = int(batch_size / 2)
    # Loop through epochs
    for i in range(epochs):
        # Get real random samples
        x_real, y_real = generate_real_samples(half_batch)
        # Get fake samples (from generator)
        x_fake, y_fake = generate_fake_samples(gen, latent_dim, half_batch)
        # Train discriminator on real
        dis.train_on_batch(x_real, y_real)
        # Train discriminator on fake
        dis.train_on_batch(x_fake, y_fake)
        # Generate some random latent points
        x_gan = generate_latent_points(latent_dim, batch_size)
        # Ground truth
        y_gan = np.ones((batch_size, 1))
        # Train the GAN, update the generator via the discriminator
        gan.train_on_batch(x_gan, y_gan)
        # Evaluate GAN progress
        if i % eval == 0:
            evaluate_gan(i, gen, dis, latent_dim)


train(generator, discriminator, gan, latent_dim)
