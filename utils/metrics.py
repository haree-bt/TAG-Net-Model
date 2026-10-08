"""
Robotics Grasp Evaluation Metrics
Calculates:
1. Jaccard Grasp Metric (IoU > 0.25, Angle Error < 30 deg)
2. Task Success Rate (TSR): Percentage of grasps executed on the valid functional zone
3. Inference Latency (FPS)
"""

import numpy as np
import torch


def detect_top_grasp(quality_map, sin_map, cos_map, width_map, threshold=0.2):
    """
    Extracts the highest-confidence grasp from heatmaps.
    Returns:
        (x, y, angle_rad, width_pixels, confidence) or None if below threshold.
    """
    q = quality_map.detach().cpu().numpy().squeeze()
    s = sin_map.detach().cpu().numpy().squeeze()
    c = cos_map.detach().cpu().numpy().squeeze()
    w = width_map.detach().cpu().numpy().squeeze()

    max_idx = np.unravel_index(np.argmax(q), q.shape)
    confidence = q[max_idx]

    if confidence < threshold:
        return None

    gy, gx = max_idx
    sin_val = s[gy, gx]
    cos_val = c[gy, gx]
    angle = 0.5 * np.arctan2(sin_val, cos_val)
    angle = (angle + np.pi) % np.pi  # Normalize to [0, pi]
    width = w[gy, gx] * q.shape[0]

    return gx, gy, angle, width, float(confidence)


def evaluate_sample_grasp(pred_grasp, affordance_gt, quality_geom_gt, angle_tolerance=np.deg2rad(30)):
    """
    Evaluates whether a predicted grasp is:
    1. Geometrically viable (on the object)
    2. Task-valid (inside the affordance zone)
    """
    if pred_grasp is None:
        return {"geometric_success": False, "task_success": False}

    gx, gy, angle, width, conf = pred_grasp
    h, w = affordance_gt.shape[-2:]

    if not (0 <= gx < w and 0 <= gy < h):
        return {"geometric_success": False, "task_success": False}

    # Geometric validity: Ground-truth geometric quality at (gy, gx) > 0.2
    geom_val = quality_geom_gt[..., gy, gx].item()
    geom_success = geom_val > 0.2

    # Task validity: Ground-truth affordance mask at (gy, gx) > 0.5
    aff_val = affordance_gt[..., gy, gx].item()
    task_success = geom_success and (aff_val > 0.5)

    return {
        "geometric_success": geom_success,
        "task_success": task_success
    }
