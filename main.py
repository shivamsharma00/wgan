import os
os.chdir("..")

# Importing Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Generator import Generator
from Discriminator import Discriminator
from util import split_indices, initialize_weights
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from IPython.display import Image
from torchvision.utils import save_image


def test(batch_size, in_channels, image_height, image_width, noise_dim):
  """
  This function tests the Discriminator and Generator models.

  Args:
    batch_size (int): The batch size.
    in_channels (int): The number of input channels.
    image_height (int): The height of the images.
    image_width (int): The width of the images.
    noise_dim (int): The dimension of the noise vector.

  Returns:
    None.
  """

  # Generate a random batch of images.
  x = torch.randn((batch_size, in_channels, image_height, image_width))

  # Create a Discriminator model.
  discriminator = Discriminator(in_channels, 8)

  # Check that the output of the Discriminator has the correct shape.
  assert discriminator(x).shape == (batch_size, 1, 1, 1), "Discriminator test failed"

  # Generate a random batch of noise.
  z = torch.randn((batch_size, noise_dim, 1, 1))

  # Create a Generator model.
  generator = Generator(noise_dim, in_channels, 8)

  # Check that the output of the Generator has the correct shape.
  assert generator(z).shape == (batch_size, in_channels, image_height, image_width), "Generator test failed"

  print("Success")


# HYPERPARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 5e-5  
batch_size = 64
image_size = 64
channels_img = 3
noise_dim = 100
num_epochs = 100
features_disc = 64
features_gen = 64
critic_iterations = 5
weight_clip = 0.02
image_path = '../input/celeba-dataset/img_align_celeba'

# Define image transforms
transforms = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for i in range(channels_img)], [0.5 for i in range(channels_img)])
])

# Load the dataset and create train/validation splits
dataset = datasets.ImageFolder(image_path, transform = transforms)
validation_percentage = 0.4
random_seed = 42
train_indices, val_indices = split_indices(len(dataset), validation_percentage, random_seed)

# Create the dataloader for the training set
train_sampler = SubsetRandomSampler(train_indices)
loader = DataLoader(dataset, batch_size, sampler = train_sampler)

# Load the dataset.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Define the generator and discriminator models
generator = Generator(z_dim=noise_dim, channels_img=channels_img, features_g=features_gen).to(device)
discriminator = Discriminator(channels_img, features_disc).to(device)

# Initialize the weights of the models.
initialize_weights(generator)
initialize_weights(discriminator)

# Define the optimizers for the generator and discriminator
optimizer_generator = optim.RMSprop(generator.parameters(), lr = learning_rate)
optimizer_discriminator = optim.RMSprop(discriminator.parameters(), lr = learning_rate)

# Define a helper function to reset the gradients of the optimizers
def reset_grad():
    optimizer_discriminator.zero_grad()
    optimizer_generator.zero_grad()


def train_discriminator(images):
    # create labels, for real image label is 1, for fake 
    # loss for real images
    
    for _ in range(critic_iterations):
        
        disc_real = discriminator(images).reshape(-1)
        #d_loss_real = torch.mean(disc_real)
        real_score = disc_real
        
        
        z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
        fake_images = generator(z)
        disc_fake = discriminator(fake_images).reshape(-1)
        #d_loss_fake = torch.mean(disc_fake)
        fake_score = disc_fake
        
        loss_disc = - (torch.mean(disc_real) - torch.mean(disc_fake))
        
        reset_grad()
        
        loss_disc.backward()
        
        optimizer_discriminator.step()
        
        for p in discriminator.parameters():
                p.data.clamp_(-weight_clip, weight_clip)
        
        return loss_disc, real_score, fake_score 
    

def train_generator():
    # Generate fake images and calculate loss
    z = torch.randn(batch_size, noise_dim, 1, 1).to(device)
    fake_images = generator(z)
    labels = torch.ones(batch_size, 1).to(device)
    output = discriminator(fake_images).reshape(-1)
    g_loss = - torch.mean(output)

    # Backprop and optimize
    reset_grad()
    g_loss.backward()
    optimizer_generator.step()
    return g_loss, fake_images


sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

def denorm(x):
  out = (x + 1) / 2
  return out.clamp(0, 1)

sample_vectors = torch.randn(batch_size, noise_dim, 1, 1).to(device)

def save_fake_images(index):
    fake_images = generator(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 3, 64, 64)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=16)

import tqdm
total_step = len(loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

for epoch in tqdm.tqdm(range(num_epochs)):
    for i, (images, _) in enumerate(loader):
        # Load a batch & transform to vectors
        images = images.to(device)
        
        # Train the discriminator and generator
        d_loss, real_score, fake_score = train_discriminator(images)
        g_loss, fake_images = train_generator()
        
        # Inspect the losses
        if (i+1) / 100 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, D(x): {real_score.mean().item()}, D(G(z)): {fake_score.mean().item()}]')
    # Sample and save images
    save_fake_images(epoch+1)