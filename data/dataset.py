"""
TAG-Net Dataset Pipeline
Handles RGB-D input loading, ground-truth heatmap generation for grasps,
and semantic affordance segmentation masks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def gaussian_2d(shape, center, sigma=5):
    """Generates a 2D Gaussian heatmap centered at `center` (x, y)."""
    h, w = shape
    x = np.arange(0, w, 1, float)
    y = np.arange(0, h, 1, float)[:, np.newaxis]
    x0, y0 = center
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


class SyntheticToolAffordanceDataset(Dataset):
    """
    Benchmark Dataset for Task-Aware Grasping.
    Generates realistic RGB-D objects (knives, mugs, screwdrivers, scissors, hammers)
    with ground-truth geometric grasps and task-specific affordance zones.
    
    Demonstrates the exact failure mode of classical grasping:
    Objects have geometrically graspable parts that are task-invalid (e.g., knife blade),
    and functional affordance parts (e.g., knife handle).
    """

    TOOL_TYPES = ["knife", "mug", "screwdriver", "hammer", "scissors"]

    def __init__(self, num_samples=1000, img_size=224, seed=42):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.num_samples

    def _render_knife(self, center, angle, scale):
        """Renders a knife: handle (affordance=1) and blade (affordance=0)."""
        img_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        depth = np.ones((self.img_size, self.img_size), dtype=np.float32) * 0.8  # Table plane at 0.8m
        affordance_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        grasps = []

        cx, cy = center
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        handle_len = int(45 * scale)
        blade_len = int(60 * scale)
        width = int(14 * scale)

        # Handle points (affordance = 1, safe for handover / tool use)
        h_start = (int(cx - handle_len * cos_a), int(cy - handle_len * sin_a))
        h_end = (int(cx), int(cy))
        cv2.line(img_rgb, h_start, h_end, (0.2, 0.4, 0.2), width)
        cv2.line(depth, h_start, h_end, 0.76, width)
        cv2.line(affordance_mask, h_start, h_end, 1.0, width)

        # Blade points (affordance = 0, dangerous to grasp)
        b_end = (int(cx + blade_len * cos_a), int(cy + blade_len * sin_a))
        cv2.line(img_rgb, h_end, b_end, (0.75, 0.75, 0.8), width - 2)
        cv2.line(depth, h_end, b_end, 0.77, width - 2)

        # Candidate grasps on handle (Affordance-valid)
        for t in np.linspace(0.2, 0.8, 4):
            gx = int(h_start[0] + t * (h_end[0] - h_start[0]))
            gy = int(h_start[1] + t * (h_end[1] - h_start[1]))
            # Grasp angle is perpendicular to length
            g_theta = (angle + np.pi / 2) % np.pi
            grasps.append((gx, gy, g_theta, width * 1.5, True))

        # Candidate grasps on blade (Geometrically graspable, but Task-invalid!)
        for t in np.linspace(0.3, 0.8, 4):
            gx = int(h_end[0] + t * (b_end[0] - h_end[0]))
            gy = int(h_end[1] + t * (b_end[1] - h_end[1]))
            g_theta = (angle + np.pi / 2) % np.pi
            grasps.append((gx, gy, g_theta, width * 1.4, False))

        return img_rgb, depth, affordance_mask, grasps

    def _render_screwdriver(self, center, angle, scale):
        """Renders screwdriver: plastic handle (affordance=1) and metal shaft (affordance=0)."""
        img_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        depth = np.ones((self.img_size, self.img_size), dtype=np.float32) * 0.8
        affordance_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        grasps = []

        cx, cy = center
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        handle_len = int(40 * scale)
        shaft_len = int(55 * scale)
        handle_w = int(18 * scale)
        shaft_w = int(8 * scale)

        h_start = (int(cx - handle_len * cos_a), int(cy - handle_len * sin_a))
        h_end = (int(cx), int(cy))
        cv2.line(img_rgb, h_start, h_end, (0.8, 0.3, 0.1), handle_w)
        cv2.line(depth, h_start, h_end, 0.75, handle_w)
        cv2.line(affordance_mask, h_start, h_end, 1.0, handle_w)

        s_end = (int(cx + shaft_len * cos_a), int(cy + shaft_len * sin_a))
        cv2.line(img_rgb, h_end, s_end, (0.6, 0.6, 0.65), shaft_w)
        cv2.line(depth, h_end, s_end, 0.78, shaft_w)

        # Grasps on handle (Affordance-valid)
        for t in np.linspace(0.2, 0.8, 3):
            gx = int(h_start[0] + t * (h_end[0] - h_start[0]))
            gy = int(h_start[1] + t * (h_end[1] - h_start[1]))
            grasps.append((gx, gy, (angle + np.pi / 2) % np.pi, handle_w * 1.4, True))

        # Grasps on shaft (Geometric only)
        for t in np.linspace(0.3, 0.8, 3):
            gx = int(h_end[0] + t * (s_end[0] - h_end[0]))
            gy = int(h_end[1] + t * (s_end[1] - h_end[1]))
            grasps.append((gx, gy, (angle + np.pi / 2) % np.pi, shaft_w * 1.5, False))

        return img_rgb, depth, affordance_mask, grasps

    def _render_mug(self, center, angle, scale):
        """Renders mug: rim/container (affordance=0) and ear handle (affordance=1)."""
        img_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        depth = np.ones((self.img_size, self.img_size), dtype=np.float32) * 0.8
        affordance_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        grasps = []

        cx, cy = center
        radius = int(28 * scale)
        handle_offset = int(radius * 1.25)

        # Mug body (Geometric graspable, but invalid for pouring / handover)
        cv2.circle(img_rgb, (cx, cy), radius, (0.85, 0.85, 0.85), -1)
        cv2.circle(img_rgb, (cx, cy), int(radius * 0.8), (0.3, 0.3, 0.35), -1)
        cv2.circle(depth, (cx, cy), radius, 0.72, -1)

        # Mug handle
        hx = int(cx + handle_offset * np.cos(angle))
        hy = int(cy + handle_offset * np.sin(angle))
        cv2.circle(img_rgb, (hx, hy), int(10 * scale), (0.7, 0.7, 0.7), int(5 * scale))
        cv2.circle(depth, (hx, hy), int(10 * scale), 0.74, int(5 * scale))
        cv2.circle(affordance_mask, (hx, hy), int(12 * scale), 1.0, -1)

        # Handle grasp (Affordance-valid)
        grasps.append((hx, hy, (angle + np.pi / 2) % np.pi, 20 * scale, True))

        # Rim grasps (Geometric only)
        for d_theta in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            rx = int(cx + radius * 0.85 * np.cos(d_theta))
            ry = int(cy + radius * 0.85 * np.sin(d_theta))
            grasps.append((rx, ry, d_theta % np.pi, 16 * scale, False))

        return img_rgb, depth, affordance_mask, grasps

    def __getitem__(self, idx):
        # Deterministic variation per index
        rng = np.random.default_rng(idx)
        tool_type = rng.choice(["knife", "screwdriver", "mug"])
        center = (rng.integers(70, 154), rng.integers(70, 154))
        angle = rng.uniform(0, np.pi)
        scale = rng.uniform(0.85, 1.25)

        if tool_type == "knife":
            rgb, depth, affordance_mask, grasps = self._render_knife(center, angle, scale)
        elif tool_type == "screwdriver":
            rgb, depth, affordance_mask, grasps = self._render_screwdriver(center, angle, scale)
        else:
            rgb, depth, affordance_mask, grasps = self._render_mug(center, angle, scale)

        # Add realistic sensor noise to depth
        depth_noise = rng.normal(0, 0.003, depth.shape).astype(np.float32)
        depth = np.clip(depth + depth_noise, 0.0, 1.0)

        # Normalize depth to [0, 1] relative to workspace
        depth_norm = (depth - 0.7) / 0.15
        depth_norm = np.clip(depth_norm, 0.0, 1.0)

        # Target heatmaps
        quality_geom = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        quality_task = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        sin_2theta = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        cos_2theta = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        width_map = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        for (gx, gy, g_ang, g_w, is_affordance_valid) in grasps:
            if 0 <= gx < self.img_size and 0 <= gy < self.img_size:
                g_heat = gaussian_2d((self.img_size, self.img_size), (gx, gy), sigma=6)
                quality_geom = np.maximum(quality_geom, g_heat)
                if is_affordance_valid:
                    quality_task = np.maximum(quality_task, g_heat)

                # Assign angles where gaussian > 0.1
                mask = g_heat > 0.1
                sin_val = np.sin(2 * g_ang)
                cos_val = np.cos(2 * g_ang)
                sin_2theta[mask] = sin_val
                cos_2theta[mask] = cos_val
                width_map[mask] = g_w / self.img_size

        # Combine RGB + Depth -> 4-channel input [4, H, W]
        rgbd = np.concatenate([rgb.transpose(2, 0, 1), depth_norm[np.newaxis, :, :]], axis=0)

        return {
            "rgbd": torch.from_numpy(rgbd.astype(np.float32)),
            "quality_geom": torch.from_numpy(quality_geom[np.newaxis, :, :].astype(np.float32)),
            "quality_task": torch.from_numpy(quality_task[np.newaxis, :, :].astype(np.float32)),
            "sin_2theta": torch.from_numpy(sin_2theta[np.newaxis, :, :].astype(np.float32)),
            "cos_2theta": torch.from_numpy(cos_2theta[np.newaxis, :, :].astype(np.float32)),
            "width": torch.from_numpy(width_map[np.newaxis, :, :].astype(np.float32)),
            "affordance_mask": torch.from_numpy(affordance_mask[np.newaxis, :, :].astype(np.float32)),
            "tool_type": tool_type
        }
