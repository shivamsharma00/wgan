
import torch.nn as nn
   
class Generator(nn.Module):
    """A generative network that takes a latent vector as input and outputs an image."""

    def __init__(self, z_dim, channels_img, features_g):
        """Initialize the generator.

        Args:
            z_dim (int): The dimensionality of the latent vector.
            channels_img (int): The number of channels in the output image.
            features_g (int): The number of features in the generator's hidden layers.
        """
        super(Generator, self).__init__()

        # The generator consists of a series of transposed convolutional layers,
        # followed by batch normalization and ReLU activation layers.
        self.net = nn.Sequential(
            # First layer:
            #   Input: z_dim
            #   Output: features_g * 16, size = 4 * 4
            nn.ConvTranspose2d(z_dim, features_g * 16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(),

            # Second layer:
            #   Input: features_g * 16, size = 4 * 4
            #   Output: features_g * 8, size = 8 * 8
            nn.ConvTranspose2d(features_g * 16, features_g * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(),

            # Third layer:
            #   Input: features_g * 8, size = 8 * 8
            #   Output: features_g * 4, size = 16 * 16
            nn.ConvTranspose2d(features_g * 8, features_g * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(),

            # Fourth layer:
            #   Input: features_g * 4, size = 16 * 16
            #   Output: features_g * 2, size = 32 * 32
            nn.ConvTranspose2d(features_g * 4, features_g * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(),

            # Fifth layer:
            #   Input: features_g * 2, size = 32 * 32
            #   Output: channels_img, size = 64 * 64
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, x):
        """Generate an image from a latent vector.

        Args:
            x (torch.Tensor): The latent vector.

        Returns:
            torch.Tensor: The generated image.
        """
        return self.net(x)