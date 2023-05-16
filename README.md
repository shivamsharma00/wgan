
# WGAN (Wasserstein GAN) - README

## Introduction
This is the PyTorch implementation of Wasserstein GAN (WGAN), a generative adversarial network (GAN) variant that uses Wasserstein distance instead of Jensen-Shannon divergence to measure the similarity between the generated and real data distributions. This code generates new images that resemble the images in a given dataset.

![Artificially generated Images](https://github.com/shivamsharma00/wgan/blob/master/gan_gif.gif)

## Requirements
* Python 3.x
* PyTorch 1.7 or higher
* torchvision
* numpy
* matplotlib

## Getting Started
The `test()` function in the code can be used to check if the Discriminator and Generator models have the correct output shapes. To run the code for generating new images using WGAN, first download the dataset. The code uses the CelebA dataset in this implementation. The dataset can be downloaded from [here](https://www.kaggle.com/jessicali9530/celeba-dataset). 

After downloading and extracting the dataset, update the `image_path` variable in the code with the path to the directory containing the image files. 

Next, adjust the hyperparameters, such as learning rate, batch size, number of epochs, etc., to desired values.

Run the code. The Generator model will generate new images during the training process, and the Discriminator model will evaluate how similar the generated images are to the real images. The generated images will be saved in the `samples` directory.

## Code Description
### `test(batch_size, in_channels, image_height, image_width, noise_dim)`
* This function tests the Discriminator and Generator models by generating a random batch of images and a random batch of noise and checking that the output of the models has the correct shape.

### Hyperparameters
* `device`: device to use for training, either "cuda" if available, otherwise "cpu"
* `learning_rate`: learning rate for the optimizer
* `batch_size`: batch size for training
* `image_size`: size of the image
* `channels_img`: number of channels in the image
* `noise_dim`: dimension of the noise vector
* `num_epochs`: number of epochs to train for
* `features_disc`: number of features in the Discriminator
* `features_gen`: number of features in the Generator
* `critic_iterations`: number of times to train the Discriminator before training the Generator
* `weight_clip`: maximum absolute value of weights in the Discriminator
* `image_path`: path to the directory containing the image files

### Training
* `loader`: dataloader for the training set
* `generator`: the Generator model
* `discriminator`: the Discriminator model
* `initialize_weights`: initializes the weights of the models
* `optimizer_generator`: optimizer for the Generator model
* `optimizer_discriminator`: optimizer for the Discriminator model
* `reset_grad()`: helper function to reset the gradients of the optimizers
* `train_discriminator(images)`: trains the Discriminator model on a batch of real and fake images
* `train_generator()`: trains the Generator model using the output of the Discriminator
* `sample_vectors`: random noise vectors for generating new images
* `denorm(x)`: helper function to denormalize the image tensor
* `save_fake_images(index)`: saves a grid of generated images to the `samples` directory every 500 iterations

## Conclusion
WGAN is a powerful technique for generating new images that resemble a given dataset. The Wasserstein distance helps to overcome some of the issues associated with the original GAN loss function, such as mode collapse and vanishing gradients. By adjusting the hyperparameters, this code can be used to generate images from any dataset.
