"""
patchgan_discriminator.py
----------------------------------
70x70 PatchGAN discriminator for conditional GAN.
Input: pose heatmaps + RGB image
Output: patch-level realism map
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN Discriminator
    """

    def __init__(self, in_channels=17, image_channels=3):
        """
        Args:
            in_channels: number of pose heatmap channels
            image_channels: number of image channels (RGB = 3)
        """
        super().__init__()
        input_channels = in_channels + image_channels

        self.layer1 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(input_channels, 64,
                          kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(64, 128,
                          kernel_size=4, stride=2, padding=1)
            ),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(128, 256,
                          kernel_size=4, stride=2, padding=1)
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(256, 512,
                          kernel_size=4, stride=2, padding=1)
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output patch map
        self.output_layer = spectral_norm(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, heatmap, image):
        """
        Args:
            heatmap: pose heatmaps (B, K, H, W)
            image: RGB image (B, 3, H, W)
        Returns:
            patch_map: (B, 1, H', W')
        """
        x = torch.cat([heatmap, image], dim=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        patch_map = self.output_layer(x)
        return patch_map
