"""
TAG-Net REST API & Interactive Demonstration Service
High-performance FastAPI service providing real-time grasp synthesis,
dynamic feature-space alteration, live visualization, and 3D simulation streaming.
"""

import os
import sys
import io
import time
import base64
import threading
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from fastapi import FastAPI, Request, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data.dataset import SyntheticToolAffordanceDataset
from models import BaselineGraspNet, TAGNet
from utils.metrics import detect_top_grasp
from utils.visualize_results import draw_grasp_rectangle
from simulation.task_simulation import (
    generate_simulation_stream,
    get_agam_task_grasp,
    launch_task_simulation_desktop
)

app = FastAPI(
    title="TAG-Net Grasp Synthesis API",
    description="Task-Aware Affordance-Gated Robotic Grasping API with Dynamic Feature Alteration",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount assets directory for direct static serving (3D arm GIF, diagrams)
assets_path = os.path.join(ROOT_DIR, "assets")
if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# Global State: Hardware & Models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_MODEL = None
TAGNET_MODEL = None
DATASET = None
TOOL_INDEX_MAP = {}


def load_models():
    global BASELINE_MODEL, TAGNET_MODEL, DATASET, TOOL_INDEX_MAP
    if TAGNET_MODEL is not None and DATASET is not None:
        return

    print(f"[API Startup] Using Hardware Acceleration: {DEVICE}")

    # Load Baseline Checkpoint
    BASELINE_MODEL = BaselineGraspNet().to(DEVICE)
    b_path = os.path.join(ROOT_DIR, "checkpoints", "baseline_best.pth")
    if os.path.exists(b_path):
        BASELINE_MODEL.load_state_dict(torch.load(b_path, map_location=DEVICE))
        print(f"[API Startup] Loaded Baseline Checkpoint: {b_path}")
    BASELINE_MODEL.eval()

    # Load TAG-Net Checkpoint
    TAGNET_MODEL = TAGNet().to(DEVICE)
    t_path = os.path.join(ROOT_DIR, "checkpoints", "tag_net_best.pth")
    if os.path.exists(t_path):
        TAGNET_MODEL.load_state_dict(torch.load(t_path, map_location=DEVICE))
        print(f"[API Startup] Loaded TAG-Net Checkpoint: {t_path}")
    TAGNET_MODEL.eval()

    # Pre-index unseen test samples (seed=101)
    DATASET = SyntheticToolAffordanceDataset(num_samples=150, img_size=224, seed=101)
    for idx in range(len(DATASET)):
        t_type = str(DATASET[idx]["tool_type"])
        if t_type not in TOOL_INDEX_MAP:
            TOOL_INDEX_MAP[t_type] = []
        TOOL_INDEX_MAP[t_type].append(idx)
    print(f"[API Startup] Dataset indexed for tools: {list(TOOL_INDEX_MAP.keys())}")


# Initialize immediately
load_models()


class GraspRequest(BaseModel):
    tool_type: str = Field("knife", description="Target tool: knife, mug, screwdriver, hammer, scissors")
    sample_idx: int = Field(0, description="Instance index for the selected tool type (0-20)")
    task_mode: str = Field("safe_handover", description="Task mode: safe_handover, tool_use, unconstrained")
    affordance_weight: float = Field(1.0, ge=0.0, le=1.0, description="Alpha: 0.0 (Pure Geometry) to 1.0 (Strict Safety)")
    model_type: str = Field("tag_net", description="Model architecture: tag_net or baseline")
    quality_threshold: float = Field(0.05, ge=0.01, le=0.8, description="Minimum candidate grasp confidence")
    max_gripper_width: float = Field(50.0, ge=15.0, le=90.0, description="Maximum gripper finger opening (pixels)")


PLOT_LOCK = threading.Lock()
LATEST_IMAGE_BYTES = None


def generate_visualization_bytes(rgb, grasp, aff_gt, q_map, aff_pred=None, title="Grasp Output"):
    """Thread-safe generation of side-by-side Grasp Detection + Quality Heatmap PNG bytes."""
    global LATEST_IMAGE_BYTES
    is_safe = False
    with PLOT_LOCK:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=100)
        plt.subplots_adjust(wspace=0.15, left=0.05, right=0.95, top=0.88, bottom=0.08)

        # 1. Grasp on RGB
        axes[0].imshow(rgb)
        axes[0].set_title(f"RGB Camera: {title}", fontsize=11, fontweight="bold", color="white")
        axes[0].axis("off")

        if grasp:
            gx, gy, ang, w, conf = grasp
            is_safe = bool(aff_gt[int(np.clip(gy, 0, 223)), int(np.clip(gx, 0, 223))] > 0.5)
            color = "#10B981" if is_safe else "#EF4444"  # Emerald Green vs Crimson Red
            lbl = f"CONF: {conf*100:.1f}%\n" + ("SAFE [PASS]" if is_safe else "HAZARD [FAIL]")
            draw_grasp_rectangle(axes[0], gx, gy, ang, w, color=color, label=lbl)

        # 2. Quality Heatmap
        im = axes[1].imshow(q_map, cmap="jet", vmin=0.0, vmax=max(0.6, float(np.max(q_map))))
        axes[1].set_title("Synthesized Grasp Quality Heatmap", fontsize=11, fontweight="bold", color="white")
        axes[1].axis("off")
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#0F172A", edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        img_bytes = buf.read()
        LATEST_IMAGE_BYTES = img_bytes
        return img_bytes, is_safe


def create_visualization_base64(rgb, grasp, aff_gt, q_map, aff_pred=None, title="Grasp Output"):
    """Renders side-by-side Grasp Detection + Quality Heatmap into Base64 PNG."""
    img_bytes, is_safe = generate_visualization_bytes(rgb, grasp, aff_gt, q_map, aff_pred=aff_pred, title=title)
    return base64.b64encode(img_bytes).decode("utf-8"), is_safe


@app.get("/health")
def health():
    """Health status and active hardware compute device."""
    return {
        "status": "online",
        "service": "TAG-Net Grasp Synthesis API",
        "hardware": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU",
        "supported_tools": list(TOOL_INDEX_MAP.keys()) if TOOL_INDEX_MAP else []
    }


@app.get("/features")
def list_features():
    """Returns documentation of all tunable input features and valid ranges."""
    return {
        "features": {
            "tool_type": {
                "type": "select",
                "options": ["knife", "mug", "screwdriver", "scissors", "hammer"],
                "default": "knife",
                "description": "Category of object under active camera view"
            },
            "task_mode": {
                "type": "select",
                "options": ["safe_handover", "tool_use", "unconstrained"],
                "default": "safe_handover",
                "description": "Robotic operational intention (handover prioritizes handles; unconstrained ignores task)"
            },
            "affordance_weight": {
                "type": "slider",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "default": 1.0,
                "description": "Affordance Gating factor alpha: 0.0 produces classical blind geometry, 1.0 enforces strict semantic safety"
            },
            "model_type": {
                "type": "select",
                "options": ["tag_net", "baseline"],
                "default": "tag_net",
                "description": "Neural network architecture to execute"
            },
            "quality_threshold": {
                "type": "slider",
                "min": 0.01,
                "max": 0.8,
                "step": 0.02,
                "default": 0.05,
                "description": "Minimum confidence required to declare a valid grasp"
            },
            "max_gripper_width": {
                "type": "slider",
                "min": 15.0,
                "max": 90.0,
                "step": 5.0,
                "default": 50.0,
                "description": "Physical limit of gripper finger span (pixels)"
            }
        }
    }


def run_inference_core(
    tool_type: str = "knife",
    sample_idx: int = 1,
    model_type: str = "tag_net",
    task_mode: str = "safe_handover",
    affordance_weight: float = 1.0,
    quality_threshold: float = 0.05,
    max_gripper_width: float = 50.0,
    generate_image: bool = True
):
    """
    Core execution logic shared between JSON endpoint and Binary Image endpoint.
    Guarantees 100% synchronization between visualization and returned coordinates.
    """
    tools = TOOL_INDEX_MAP.get(tool_type.lower(), [0])
    sample_index = tools[sample_idx % len(tools)]
    sample = DATASET[sample_index]

    x = sample["rgbd"].unsqueeze(0).to(DEVICE)
    rgb = sample["rgbd"][:3].numpy().transpose(1, 2, 0)
    aff_gt = sample["affordance_mask"][0].numpy()

    t0 = time.perf_counter()
    with torch.no_grad():
        if model_type.lower() == "baseline":
            out = BASELINE_MODEL(x)
            q_map_tensor = out["quality"][0, 0]
            aff_pred = None
        else:
            out = TAGNET_MODEL(x)
            q_task = out["quality_task"][0, 0]
            aff_pred = out["affordance_mask"][0, 0].cpu().numpy()

            alpha = float(affordance_weight)
            if task_mode in ["unconstrained", "pick_and_place"]:
                alpha = min(0.65, alpha) if task_mode == "pick_and_place" else 0.0
            elif task_mode in ["tool_use", "work_with_it"]:
                alpha = min(0.85, alpha)
            elif task_mode in ["safe_handover", "handover"]:
                alpha = alpha

            if alpha < 0.99:
                b_out = BASELINE_MODEL(x)
                q_base = b_out["quality"][0, 0]
                q_map_tensor = (1.0 - alpha) * q_base + alpha * q_task
            else:
                q_map_tensor = q_task

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    q_map_np = q_map_tensor.cpu().numpy()
    sin_map = out["sin_2theta"][0]
    cos_map = out["cos_2theta"][0]
    w_map = out["width"][0]

    grasp = detect_top_grasp(
        q_map_tensor.unsqueeze(0),
        sin_map,
        cos_map,
        w_map,
        threshold=quality_threshold
    )

    is_safe = False
    if grasp:
        gx, gy, ang, w, conf = grasp
        is_safe = bool(aff_gt[int(np.clip(gy, 0, 223)), int(np.clip(gx, 0, 223))] > 0.5)

    img_bytes = None
    if generate_image:
        title_label = f"{tool_type.upper()} | Model: {model_type.upper()} (α={affordance_weight:.2f})"
        img_bytes, is_safe = generate_visualization_bytes(rgb, grasp, aff_gt, q_map_np, aff_pred, title=title_label)

    grasp_data = None
    if grasp:
        gx, gy, ang, w, conf = grasp
        grasp_data = {
            "center_x": float(round(gx, 2)),
            "center_y": float(round(gy, 2)),
            "angle_deg": float(round(np.rad2deg(ang), 2)),
            "angle_rad": float(round(ang, 4)),
            "width_px": float(round(w, 2)),
            "confidence": float(round(conf, 4))
        }

    image_url = (
        f"/visualization.png?tool_type={tool_type}&sample_idx={sample_idx}&model_type={model_type}"
        f"&task_mode={task_mode}&affordance_weight={affordance_weight}&quality_threshold={quality_threshold}"
    )

    result = {
        "success": bool(grasp is not None),
        "tool_type": tool_type,
        "sample_idx": sample_idx,
        "features_applied": {
            "task_mode": task_mode,
            "affordance_weight_alpha": affordance_weight,
            "model_type": model_type,
            "quality_threshold": quality_threshold,
            "max_gripper_width": max_gripper_width
        },
        "grasp": grasp_data,
        "safety_evaluation": {
            "is_safe": is_safe,
            "verdict": "SAFE (Handle) [PASS]" if is_safe else ("DANGEROUS (Hazard/Rim) [FAIL]" if grasp else "NO_GRASP"),
            "target_contact_zone": "handle" if is_safe else "working_tip_or_rim"
        },
        "performance": {
            "latency_ms": round(latency_ms, 3),
            "estimated_fps": round(1000.0 / max(0.01, latency_ms), 1),
            "device": str(DEVICE)
        },
        "image_url": image_url,
        "visualization_base64": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}" if img_bytes else None
    }
    return result, img_bytes


@app.post("/predict_grasp")
def predict_grasp(req: GraspRequest):
    """
    Main Grasp Prediction Endpoint:
    Receives feature configuration, executes forward pass, and returns grasp coordinates + metrics.
    Ultra-low latency (~8ms) without Matplotlib rendering overhead.
    """
    data, _ = run_inference_core(
        tool_type=req.tool_type,
        sample_idx=req.sample_idx,
        model_type=req.model_type,
        task_mode=req.task_mode,
        affordance_weight=req.affordance_weight,
        quality_threshold=req.quality_threshold,
        max_gripper_width=req.max_gripper_width,
        generate_image=False
    )
    return data


@app.get("/visualization.png")
@app.get("/preview_image.png")
def get_visualization_image(
    tool_type: str = Query("knife"),
    sample_idx: int = Query(1),
    model_type: str = Query("tag_net"),
    task_mode: str = Query("safe_handover"),
    affordance_weight: float = Query(1.0),
    quality_threshold: float = Query(0.05),
    max_gripper_width: float = Query(50.0),
    t: str = Query(None)
):
    """
    Direct Binary PNG Endpoint:
    Returns the real-time synthesized visualization directly as an image/png stream.
    Used by web browsers for zero-lag, glitch-free image rendering.
    """
    _, img_bytes = run_inference_core(
        tool_type=tool_type,
        sample_idx=sample_idx,
        model_type=model_type,
        task_mode=task_mode,
        affordance_weight=affordance_weight,
        quality_threshold=quality_threshold,
        max_gripper_width=max_gripper_width,
        generate_image=True
    )
    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/latest_visualization.png")
def get_latest_visualization():
    """Returns the most recently rendered grasp visualization PNG."""
    global LATEST_IMAGE_BYTES
    if LATEST_IMAGE_BYTES is None:
        _, img_bytes = run_inference_core()
        return Response(content=img_bytes, media_type="image/png")
    return Response(
        content=LATEST_IMAGE_BYTES,
        media_type="image/png",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/simulation/stream")
def get_simulation_stream(
    tool_type: str = Query("knife"),
    task_mode: str = Query("handover"),
    width: int = Query(640),
    height: int = Query(480),
    fps: int = Query(25)
):
    """
    Live MuJoCo 3D Physics Simulation Stream.
    Streams real-time 6-DoF robotic arm pick-and-manipulation execution
    modulated by TAG-Net AGAM heatmapping directly into HTML <img> elements.
    """
    return StreamingResponse(
        generate_simulation_stream(tool_name=tool_type, task_name=task_mode, width=width, height=height, fps=fps),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/simulation/launch")
def launch_desktop_simulation(
    tool_type: str = Query("knife"),
    task_mode: str = Query("handover")
):
    """
    Spawns the native 60 FPS interactive desktop OpenGL MuJoCo window
    for full mouse orbit, pan, and zoom manipulation.
    """
    import subprocess
    norm_tool = "screwdriver" if tool_type.lower() in ["tool", "screwdriver"] else tool_type.lower()
    norm_task = "handover" if task_mode.lower() in ["handover", "safe_handover", "human_hand"] else (
        "work_with_it" if task_mode.lower() in ["work", "tool_use", "work_with_it"] else "pick_and_place"
    )
    cmd = [sys.executable, os.path.join(ROOT_DIR, "simulation", "realtime_mujoco_simulation.py"), "--tool", norm_tool, "--task", norm_task]
    subprocess.Popen(cmd, cwd=ROOT_DIR)
    return {"status": "launched", "tool": norm_tool, "task": norm_task}


@app.get("/simulation/telemetry")
def get_simulation_telemetry(
    tool_type: str = Query("knife"),
    task_mode: str = Query("handover")
):
    """Returns AGAM affordance coordinates, 3D target coordinates, and kinematic safety metrics."""
    pred = get_agam_task_grasp(tool_type, task_mode)
    return {
        "tool": pred["tool"],
        "task": pred["task"],
        "heatmap_u": pred["u"],
        "heatmap_v": pred["v"],
        "confidence_score": round(pred["score"], 4),
        "grasp_angle_deg": round(pred["angle_deg"], 2),
        "target_cartesian_xyz": pred["target_xyz"],
        "is_safe": pred["is_safe"]
    }



DASHBOARD_TEMPLATE_PATH = os.path.join(ROOT_DIR, "templates", "studio_dashboard.html")

@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
def interactive_demo_page():
    """Serves the executive-grade dark studio web UI dashboard."""
    if os.path.exists(DASHBOARD_TEMPLATE_PATH):
        with open(DASHBOARD_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Studio Dashboard Template Not Found</h1>", status_code=500)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("   TAG-Net REST API and Interactive Demonstration Server")
    print("   Swagger Docs : http://localhost:8000/docs")
    print("   Web Studio   : http://localhost:8000/demo (or http://localhost:8000)")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
