"""
Training Pipeline for Robotic Grasp Models:
Trains both the Baseline (Geometric-only) model and TAG-Net (Affordance-gated),
and generates comparative benchmark metrics.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data.dataset import SyntheticToolAffordanceDataset
from models import BaselineGraspNet, TAGNet
from utils.metrics import detect_top_grasp, evaluate_sample_grasp


def train_baseline(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """Trains the standard geometric grasp baseline network."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_smooth_l1 = nn.SmoothL1Loss()
    criterion_mse = nn.MSELoss()

    print("\n" + "=" * 50)
    print(">>> TRAINING BASELINE MODEL (Geometric-Only)")
    print("=" * 50)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["rgbd"].to(device)
            q_gt = batch["quality_geom"].to(device)
            sin_gt = batch["sin_2theta"].to(device)
            cos_gt = batch["cos_2theta"].to(device)
            w_gt = batch["width"].to(device)

            optimizer.zero_grad()
            out = model(x)

            loss_q = criterion_smooth_l1(out["quality"], q_gt)
            loss_sin = criterion_mse(out["sin_2theta"], sin_gt)
            loss_cos = criterion_mse(out["cos_2theta"], cos_gt)
            loss_w = criterion_smooth_l1(out["width"], w_gt)

            loss = 2.0 * loss_q + loss_sin + loss_cos + loss_w
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Baseline Loss: {avg_loss:.4f}")

    return model


def train_tag_net(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    """Trains the proposed TAG-Net with Affordance-Gated Attention."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_bce = nn.BCELoss()
    criterion_smooth_l1 = nn.SmoothL1Loss()
    criterion_mse = nn.MSELoss()

    print("\n" + "=" * 50)
    print(">>> TRAINING PROPOSED TAG-Net (Affordance-Gated)")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x = batch["rgbd"].to(device)
            aff_gt = batch["affordance_mask"].to(device)
            q_geom_gt = batch["quality_geom"].to(device)
            q_task_gt = batch["quality_task"].to(device)
            sin_gt = batch["sin_2theta"].to(device)
            cos_gt = batch["cos_2theta"].to(device)
            w_gt = batch["width"].to(device)

            optimizer.zero_grad()
            out = model(x)

            loss_aff = criterion_bce(out["affordance_mask"], aff_gt)
            loss_q_geom = criterion_smooth_l1(out["quality_geom"], q_geom_gt)
            loss_q_task = criterion_smooth_l1(out["quality_task"], q_task_gt)
            loss_sin = criterion_mse(out["sin_2theta"], sin_gt)
            loss_cos = criterion_mse(out["cos_2theta"], cos_gt)
            loss_w = criterion_smooth_l1(out["width"], w_gt)

            # Novel Compound Loss
            loss = (3.0 * loss_aff + 
                    1.5 * loss_q_geom + 
                    3.0 * loss_q_task + 
                    loss_sin + loss_cos + loss_w)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if epoch % 2 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - TAG-Net Compound Loss: {avg_loss:.4f}")

    return model


def evaluate_model(model, test_loader, device, is_tag_net=False):
    """Evaluates Geometric Accuracy, Task Success Rate, and Inference Latency."""
    model.eval()
    geom_successes = 0
    task_successes = 0
    total = 0
    inference_times = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["rgbd"].to(device)
            aff_gt = batch["affordance_mask"]
            q_geom_gt = batch["quality_geom"]

            batch_size = x.size(0)
            for i in range(batch_size):
                sample_x = x[i:i+1]
                
                # Measure latency
                t0 = time.perf_counter()
                out = model(sample_x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                inference_times.append(t1 - t0)

                # Quality map to use for top grasp detection
                if is_tag_net:
                    q_map = out["quality_task"][0]
                else:
                    q_map = out["quality"][0]

                grasp = detect_top_grasp(
                    q_map,
                    out["sin_2theta"][0],
                    out["cos_2theta"][0],
                    out["width"][0],
                    threshold=0.1
                )

                eval_res = evaluate_sample_grasp(
                    grasp,
                    aff_gt[i],
                    q_geom_gt[i]
                )

                if eval_res["geometric_success"]:
                    geom_successes += 1
                if eval_res["task_success"]:
                    task_successes += 1
                total += 1

    avg_latency_ms = np.mean(inference_times[5:]) * 1000  # Discard warmup
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0

    return {
        "geometric_accuracy": (geom_successes / total) * 100.0,
        "task_success_rate": (task_successes / total) * 100.0,
        "latency_ms": avg_latency_ms,
        "fps": fps
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Dataset preparation
    full_dataset = SyntheticToolAffordanceDataset(num_samples=1000, img_size=224, seed=42)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

    # 2. Initialize Models
    baseline_model = BaselineGraspNet().to(device)
    tag_net_model = TAGNet().to(device)

    # 3. Train Both Models
    baseline_model = train_baseline(baseline_model, train_loader, test_loader, device, epochs=10)
    tag_net_model = train_tag_net(tag_net_model, train_loader, test_loader, device, epochs=10)

    # 4. Save Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(baseline_model.state_dict(), "checkpoints/baseline_best.pth")
    torch.save(tag_net_model.state_dict(), "checkpoints/tag_net_best.pth")
    print("\nSaved trained model checkpoints to checkpoints/ directory.")

    # 5. Evaluate and Compare
    print("\n" + "=" * 50)
    print(">>> RUNNING RIGOROUS BENCHMARK EVALUATION")
    print("=" * 50)
    baseline_metrics = evaluate_model(baseline_model, test_loader, device, is_tag_net=False)
    tag_net_metrics = evaluate_model(tag_net_model, test_loader, device, is_tag_net=True)

    print("\n" + "=" * 70)
    print(f"{'METRIC':<30} | {'BASELINE (Geometric)':<20} | {'TAG-Net (Proposed)':<15}")
    print("-" * 70)
    print(f"{'Geometric Success Rate (%)':<30} | {baseline_metrics['geometric_accuracy']:<20.1f} | {tag_net_metrics['geometric_accuracy']:<15.1f}")
    print(f"{'Task Success Rate (TSR) (%)':<30} | {baseline_metrics['task_success_rate']:<20.1f} | {tag_net_metrics['task_success_rate']:<15.1f}")
    print(f"{'Inference Latency (ms)':<30} | {baseline_metrics['latency_ms']:<20.2f} | {tag_net_metrics['latency_ms']:<15.2f}")
    print(f"{'Inference Speed (FPS)':<30} | {baseline_metrics['fps']:<20.1f} | {tag_net_metrics['fps']:<15.1f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
