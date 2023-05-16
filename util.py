import numpy as np
import torch.nn as nn

def split_indices(num_samples, validation_percentage, seed=0):
  """
  Splits a list of indices into two parts, one for training and one for validation.

  Args:
    num_samples: The total number of samples.
    validation_percentage: The percentage of samples to use for validation.
    seed: The random seed to use.

  Returns:
    A tuple of two lists, one containing the training indices and one containing the validation indices.
  """

  # Calculate the number of validation samples.
  num_validation_samples = int(num_samples * validation_percentage)

  # Set the random seed.
  np.random.seed(seed)

  # Generate a random permutation of the indices.
  indices = np.random.permutation(num_samples)

  # Split the indices into two parts.
  training_indices = indices[:-num_validation_samples]
  validation_indices = indices[-num_validation_samples:]

  # Return the two lists of indices.
  return training_indices, validation_indices

def initialize_weights(model):
    """Initializes weights according to the DCGAN paper.

    Args:
        model (nn.Module): The model to initialize the weights of.
    """

    # Iterate over the model's modules.
    for m in model.modules():
        # Check if the module is a convolutional or batch normalization layer.
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            # Initialize the weights of the layer to a normal distribution with mean 0.0 and standard deviation 0.02.
            nn.init.normal_(m.weight.data, 0.0, 0.02)
