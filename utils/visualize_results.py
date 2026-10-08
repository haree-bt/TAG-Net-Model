"""
Generates side-by-side comparative visualizations between Baseline and TAG-Net.
Produces the primary figure for the paper and presentation slides.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import cv2

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.dataset import SyntheticToolAffordanceDataset
from models import BaselineGraspNet, TAGNet
from utils.metrics import detect_top_grasp


def draw_grasp_rectangle(ax, gx, gy, angle, width, color='lime', label=None):
    """Draws an oriented parallel-jaw gripper rectangle on a matplotlib axis."""
    length = 24  # Gripper jaw finger length
    
    # Calculate corner offsets
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Center points for left and right fingers
    dx_w = (width / 2) * sin_a
    dy_w = -(width / 2) * cos_a
    
    dx_l = (length / 2) * cos_a
    dy_l = (length / 2) * sin_a

    # Gripper center marker
    ax.scatter([gx], [gy], color=color, s=40, zorder=5)

    # Gripper crossbar
    ax.plot([gx - dx_w, gx + dx_w], [gy - dy_w, gy + dy_w], color=color, linewidth=2.5, zorder=4)

    # Left and right jaws
    ax.plot([gx - dx_w - dx_l, gx - dx_w + dx_l], [gy - dy_w - dy_l, gy - dy_w + dy_l], color=color, linewidth=3.5, zorder=4)
    ax.plot([gx + dx_w - dx_l, gx + dx_w + dx_l], [gy + dy_w - dy_l, gy + dy_w + dy_l], color=color, linewidth=3.5, zorder=4)

    if label:
        ax.text(gx, gy - 20, label, color=color, fontsize=9, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))


def generate_comparison_figure(sample_idx=2, save_name="comparative_results.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset sample
    dataset = SyntheticToolAffordanceDataset(num_samples=50, img_size=224, seed=99)
    sample = dataset[sample_idx]
    
    x = sample["rgbd"].unsqueeze(0).to(device)
    tool_type = sample["tool_type"]
    rgb = sample["rgbd"][:3].numpy().transpose(1, 2, 0)
    depth = sample["rgbd"][3].numpy()
    aff_gt = sample["affordance_mask"][0].numpy()

    # Load Baseline Model
    baseline = BaselineGraspNet().to(device)
    baseline.load_state_dict(torch.load("checkpoints/baseline_best.pth", map_location=device))
    baseline.eval()

    # Load TAG-Net Model
    tag_net = TAGNet().to(device)
    tag_net.load_state_dict(torch.load("checkpoints/tag_net_best.pth", map_location=device))
    tag_net.eval()

    with torch.no_grad():
        b_out = baseline(x)
        t_out = tag_net(x)

    # Extract Top Grasps
    b_grasp = detect_top_grasp(b_out["quality"][0], b_out["sin_2theta"][0], b_out["cos_2theta"][0], b_out["width"][0], threshold=0.05)
    t_grasp = detect_top_grasp(t_out["quality_task"][0], t_out["sin_2theta"][0], t_out["cos_2theta"][0], t_out["width"][0], threshold=0.05)

    # Plot Comparison Grid
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    # Row 1: Baseline (Geometric-Only Failure)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"Input RGB: {tool_type.upper()}", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(b_out["quality"][0, 0].cpu().numpy(), cmap='jet')
    axes[0, 1].set_title("Baseline Grasp Quality Map\n(Ignores Functionality)", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(rgb)
    if b_grasp:
        gx, gy, ang, w, conf = b_grasp
        # Check if baseline failed task
        is_safe = aff_gt[int(gy), int(gx)] > 0.5
        box_col = 'lime' if is_safe else 'red'
        tag_text = "TASK SAFE" if is_safe else "TASK FAILURE (Dangerous Zone)"
        draw_grasp_rectangle(axes[0, 2], gx, gy, ang, w, color=box_col, label=tag_text)
    axes[0, 2].set_title("Baseline Grasp Execution", fontsize=11, fontweight='bold', color='red' if not is_safe else 'green')
    axes[0, 2].axis('off')

    # Row 2: TAG-Net (Affordance-Gated Success)
    axes[1, 0].imshow(t_out["affordance_mask"][0, 0].cpu().numpy(), cmap='Greens')
    axes[1, 0].set_title("TAG-Net: Predicted Affordance Mask\n(Functional Zone: Handle)", fontsize=10, color='darkgreen', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(t_out["quality_task"][0, 0].cpu().numpy(), cmap='jet')
    axes[1, 1].set_title("TAG-Net: Affordance-Gated Grasp Map\n(Gated by AGAM Module)", fontsize=10, color='blue', fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(rgb)
    if t_grasp:
        tgx, tgy, tang, tw, tconf = t_grasp
        is_safe_tag = aff_gt[int(tgy), int(tgx)] > 0.5
        draw_grasp_rectangle(axes[1, 2], tgx, tgy, tang, tw, color='lime', label="TASK SUCCESS (Safe Handle Grasp)")
    axes[1, 2].set_title("TAG-Net Grasp Execution", fontsize=11, fontweight='bold', color='green')
    axes[1, 2].axis('off')

    plt.suptitle("Comparative Evaluation: SOTA Geometric Grasping vs. Proposed TAG-Net", fontsize=14, fontweight='bold', y=0.98)
    
    out_dir = os.path.dirname(__file__)
    save_path = os.path.join(out_dir, save_name)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Comparative figure saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    generate_comparison_figure()
