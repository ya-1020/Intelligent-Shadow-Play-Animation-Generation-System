"""
unet_generator.py
----------------------------------
U-Net generator for conditional GAN.
Input: pose heatmaps (K channels)
Output: RGB shadow puppet image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# -----------------------------
# Basic convolutional blocks
# -----------------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super().__init__()
        layers = [
            spectral_norm(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=4, stride=2, padding=1)
            )
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True):
        super().__init__()
        layers = [
            spectral_norm(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=4, stride=2, padding=1)
            )
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# -----------------------------
# U-Net Generator
# -----------------------------
class UNetGenerator(nn.Module):
    """
    U-Net Generator
    Input: pose heatmaps (B, K, H, W)
    Output: RGB image (B, 3, H, W)
    """

    def __init__(self, in_channels=17, out_channels=3):
        """
        Args:
            in_channels: number of input heatmap channels (default: 17)
            out_channels: output image channels (RGB = 3)
        """
        super().__init__()

        # Encoder
        self.down1 = DownBlock(in_channels, 64, use_norm=False)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            spectral_norm(
                nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
            ),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up4 = UpBlock(512, 512)
        self.up3 = UpBlock(1024, 256)
        self.up2 = UpBlock(512, 128)
        self.up1 = UpBlock(256, 64)

        # Output layer
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)     # (B, 64, H/2, W/2)
        d2 = self.down2(d1)    # (B, 128, H/4, W/4)
        d3 = self.down3(d2)    # (B, 256, H/8, W/8)
        d4 = self.down4(d3)    # (B, 512, H/16, W/16)

        # Bottleneck
        bn = self.bottleneck(d4)  # (B, 512, H/32, W/32)

        # Decoder with skip connections
        u4 = self.up4(bn)                 # (B, 512, H/16, W/16)
        u4 = torch.cat([u4, d4], dim=1)   # (B, 1024, ...)

        u3 = self.up3(u4)                 # (B, 256, H/8, W/8)
        u3 = torch.cat([u3, d3], dim=1)   # (B, 512, ...)

        u2 = self.up2(u3)                 # (B, 128, H/4, W/4)
        u2 = torch.cat([u2, d2], dim=1)   # (B, 256, ...)

        u1 = self.up1(u2)                 # (B, 64, H/2, W/2)
        u1 = torch.cat([u1, d1], dim=1)   # (B, 128, ...)

        out = self.out_conv(u1)           # (B, 3, H, W)
        return out
