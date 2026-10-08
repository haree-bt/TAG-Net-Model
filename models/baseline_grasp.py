"""
Baseline Geometric Grasp Synthesis Network (GR-ConvNet / GG-CNN Style)
Predicts grasp quality (Q), angle (sin 2θ, cos 2θ), and gripper width (W)
purely based on 3D geometry without semantic task awareness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual convolutional block with batch normalization and ReLU."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class BaselineGraspNet(nn.Module):
    """
    Standard Generative Residual Grasp Network (GR-ConvNet style).
    Input: RGB-D [B, 4, H, W]
    Outputs:
        - quality: [B, 1, H, W] in [0, 1]
        - sin_2theta: [B, 1, H, W] in [-1, 1]
        - cos_2theta: [B, 1, H, W] in [-1, 1]
        - width: [B, 1, H, W] in [0, 1]
    """
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()

        # Encoder (Downsampling)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # Bottleneck Residual Blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )

        # Decoder (Upsampling)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Prediction Heads
        self.pos_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.sin_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.cos_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.width_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.down2(x)

        # Bottleneck
        features = self.res_blocks(x)

        # Decoder
        x = self.up1(features)
        x = self.up2(x)

        # Heads
        quality = torch.sigmoid(self.pos_head(x))
        sin_2theta = torch.tanh(self.sin_head(x))
        cos_2theta = torch.tanh(self.cos_head(x))
        width = torch.sigmoid(self.width_head(x))

        return {
            "quality": quality,
            "sin_2theta": sin_2theta,
            "cos_2theta": cos_2theta,
            "width": width
        }
