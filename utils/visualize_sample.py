"""
Script to visualize a dataset sample:
Displays RGB, Depth, Affordance Mask, and Ground Truth Grasp Heatmaps.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import SyntheticToolAffordanceDataset

def main():
    dataset = SyntheticToolAffordanceDataset(num_samples=10, img_size=224, seed=123)
    sample = dataset[0]  # First sample

    rgb = sample["rgbd"][:3].numpy().transpose(1, 2, 0)
    depth = sample["rgbd"][3].numpy()
    affordance = sample["affordance_mask"][0].numpy()
    quality_geom = sample["quality_geom"][0].numpy()
    quality_task = sample["quality_task"][0].numpy()
    tool_type = sample["tool_type"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    
    axes[0].imshow(rgb)
    axes[0].set_title(f"Input RGB ({tool_type.upper()})", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(depth, cmap='inferno')
    axes[1].set_title("Input Depth (Sensor)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(quality_geom, cmap='jet')
    axes[2].set_title("Geometric Grasp Q\n(Includes dangerous blade/rim)", fontsize=10, color='red')
    axes[2].axis('off')

    axes[3].imshow(affordance, cmap='Greens')
    axes[3].set_title("Semantic Affordance\n(Safe functional zone only)", fontsize=10, color='green')
    axes[3].axis('off')

    axes[4].imshow(quality_task, cmap='jet')
    axes[4].set_title("Task Grasp Q\n(Affordance-Gated Grasp)", fontsize=10, color='blue', fontweight='bold')
    axes[4].axis('off')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "sample_visualization.png")
    plt.savefig(output_path, dpi=150)
    print(f"Sample visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
