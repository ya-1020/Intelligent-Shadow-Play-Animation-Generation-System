"""
hrnet_aam.py
----------------------------------
HRNet-W32 backbone with Attention Augmentation Module (AAM)
for shadow puppet action recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Attention Augmentation Module (AAM)
# -------------------------------------------------
class AAM(nn.Module):
    """
    Lightweight channel attention module.
    """

    def __init__(self, channels, reduction=16):
        super(AAM, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()

        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)

        return x * y


# -------------------------------------------------
# Simplified HRNet Stage Block
# -------------------------------------------------
class HRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HRBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# -------------------------------------------------
# HRNet-W32 + AAM for Action Recognition
# -------------------------------------------------
class HRNetAAM(nn.Module):
    def __init__(self, num_classes):
        super(HRNetAAM, self).__init__()

        # HRNet-W32 like multi-stage structure
        self.stage1 = HRBlock(25, 32)
        self.aam1 = AAM(32)

        self.stage2 = HRBlock(32, 64)
        self.aam2 = AAM(64)

        self.stage3 = HRBlock(64, 128)
        self.aam3 = AAM(128)

        self.stage4 = HRBlock(128, 256)
        self.aam4 = AAM(256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        x: [B, K, H, W]  (K = number of keypoints)
        """

        x = self.stage1(x)
        x = self.aam1(x)

        x = self.stage2(x)
        x = self.aam2(x)

        x = self.stage3(x)
        x = self.aam3(x)

        x = self.stage4(x)
        x = self.aam4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        out = self.classifier(x)
        return out
