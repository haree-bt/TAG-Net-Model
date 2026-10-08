"""
Benchmark Evaluation Script:
Loads trained Baseline and TAG-Net checkpoints, evaluates them on the test set,
and prints the publication-ready benchmark table.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from data.dataset import SyntheticToolAffordanceDataset
from models import BaselineGraspNet, TAGNet
from utils.metrics import detect_top_grasp, evaluate_sample_grasp


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
    print(f"Evaluation Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load test split
    full_dataset = SyntheticToolAffordanceDataset(num_samples=1000, img_size=224, seed=42)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_set = random_split(full_dataset, [train_size, test_size])
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

    # Load Baseline
    baseline = BaselineGraspNet().to(device)
    baseline.load_state_dict(torch.load("checkpoints/baseline_best.pth", map_location=device))
    baseline.eval()

    # Load TAG-Net
    tag_net = TAGNet().to(device)
    tag_net.load_state_dict(torch.load("checkpoints/tag_net_best.pth", map_location=device))
    tag_net.eval()

    print("\n" + "=" * 70)
    print(">>> RUNNING RIGOROUS BENCHMARK EVALUATION (200 TEST SAMPLES)")
    print("=" * 70)
    baseline_metrics = evaluate_model(baseline, test_loader, device, is_tag_net=False)
    tag_net_metrics = evaluate_model(tag_net, test_loader, device, is_tag_net=True)

    print("\n" + "=" * 75)
    print(f"{'METRIC':<30} | {'BASELINE (Geometric)':<20} | {'TAG-Net (Proposed)':<18}")
    print("-" * 75)
    print(f"{'Geometric Success Rate (%)':<30} | {baseline_metrics['geometric_accuracy']:<20.1f} | {tag_net_metrics['geometric_accuracy']:<18.1f}")
    print(f"{'Task Success Rate (TSR) (%)':<30} | {baseline_metrics['task_success_rate']:<20.1f} | {tag_net_metrics['task_success_rate']:<18.1f}")
    print(f"{'Inference Latency (ms)':<30} | {baseline_metrics['latency_ms']:<20.2f} | {tag_net_metrics['latency_ms']:<18.2f}")
    print(f"{'Inference Speed (FPS)':<30} | {baseline_metrics['fps']:<20.1f} | {tag_net_metrics['fps']:<18.1f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
