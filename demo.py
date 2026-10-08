"""
Interactive Demo Script for TAG-Net:
Runs real-time inference on sample objects and prints execution summary.
"""

import os
import sys
import torch
import cv2
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.visualize_results import generate_comparison_figure
from simulation.robot_sim import save_simulation_video


def run_demo():
    print("\n" + "=" * 65)
    print("      TAG-Net: Task-Aware Robotic Grasping Demonstration      ")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/3] Using Acceleration Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    print("[2/3] Generating Comparative Heatmaps and Grasp Predictions...")
    fig_path = generate_comparison_figure(sample_idx=2, save_name="comparative_results.png")
    print(f"      -> Saved visual result to: {fig_path}")

    print("[3/3] Simulating Robotic Gripper Trajectory Execution...")
    save_simulation_video("simulation/grasp_execution.mp4", "simulation/grasp_execution.gif")
    print("      -> Saved animation GIF to: simulation/grasp_execution.gif")

    print("\n" + "=" * 65)
    print("DEMO COMPLETE! All figures and simulation files are ready.")
    print("=" * 65)


if __name__ == "__main__":
    run_demo()
