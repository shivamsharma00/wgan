import torch.nn as nn

class Discriminator(nn.Module):
    """
    A discriminator network for generative adversarial networks (GANs).

    Args:
        channels_img (int): The number of channels in the input image.
        features_d (int): The number of features in the discriminator.
    """

    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()

        # The discriminator consists of a sequence of convolutional layers,
        # batch normalization layers, and leaky ReLU activation layers.

        self.disc = nn.Sequential(
            # The first layer reduces the size of the input image by half.
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # The next two layers reduce the size of the input image by half again.
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),

            # The last layer reduces the size of the input image by half one last time.
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2),

            # The final layer outputs a single value, which represents the probability
            # that the input image is real.
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The output of the discriminator.
        """
        return self.disc(x)
    
