"""
TAG-Net: Task-Aware Affordance-Gated Network for Function-Specific Robotic Grasping
Novel Paper Contribution:
Dual-stream feature extraction with an Affordance-Gated Attention Module (AGAM)
that modulates geometric grasp candidates with semantic task functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseline_grasp import ResidualBlock


class AffordanceGatedAttentionModule(nn.Module):
    """
    Novel Contribution (AGAM):
    Differentiable soft-gating mechanism that conditions geometric grasp quality
    on semantic task affordance features.
    
    Formula:
        A_gate = Sigmoid( Conv1x1( [F_geom, F_aff] ) )
        Q_task = Q_geom * A_gate
    """
    def __init__(self, in_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, f_geom, f_aff, q_geom):
        concat_features = torch.cat([f_geom, f_aff], dim=1)
        soft_gate = self.gate_conv(concat_features)
        q_task = q_geom * soft_gate
        return q_task, soft_gate


class TAGNet(nn.Module):
    """
    Task-Aware Grasp Network.
    Inputs:
        - x: RGB-D image [B, 4, H, W]
    Outputs:
        - affordance_mask: Predicted functional zone [B, 1, H, W]
        - quality_geom: Unconstrained geometric grasp quality [B, 1, H, W]
        - quality_task: Task-conditioned affordance-gated grasp quality [B, 1, H, W]
        - sin_2theta: Grasp angle sine component [B, 1, H, W]
        - cos_2theta: Grasp angle cosine component [B, 1, H, W]
        - width: Gripper opening width [B, 1, H, W]
        - soft_gate: Learned attention gating map [B, 1, H, W]
    """
    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()

        # 1. Shared Feature Encoder
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

        # 2. Shared Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4),
            ResidualBlock(base_channels * 4)
        )

        # 3. Stream A: Affordance Segmentation Decoder
        self.aff_up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.aff_up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.affordance_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

        # 4. Stream B: Geometric Grasp Decoder
        self.geom_up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.geom_up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.geom_pos_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.sin_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.cos_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        self.width_head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

        # 5. Novelty: Affordance-Gated Attention Module
        self.agam = AffordanceGatedAttentionModule(base_channels)

    def forward(self, x):
        # Shared Encoder
        feat = self.in_conv(x)
        feat = self.down1(feat)
        feat = self.down2(feat)
        bottleneck = self.bottleneck(feat)

        # Affordance Stream
        f_aff = self.aff_up1(bottleneck)
        f_aff = self.aff_up2(f_aff)
        affordance_mask = torch.sigmoid(self.affordance_head(f_aff))

        # Geometric Grasp Stream
        f_geom = self.geom_up1(bottleneck)
        f_geom = self.geom_up2(f_geom)
        quality_geom = torch.sigmoid(self.geom_pos_head(f_geom))
        sin_2theta = torch.tanh(self.sin_head(f_geom))
        cos_2theta = torch.tanh(self.cos_head(f_geom))
        width = torch.sigmoid(self.width_head(f_geom))

        # Affordance-Gated Modulation
        quality_task, soft_gate = self.agam(f_geom, f_aff, quality_geom)

        return {
            "affordance_mask": affordance_mask,
            "quality_geom": quality_geom,
            "quality_task": quality_task,
            "sin_2theta": sin_2theta,
            "cos_2theta": cos_2theta,
            "width": width,
            "soft_gate": soft_gate
        }
