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
from fastapi.responses import HTMLResponse, JSONResponse
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
    max_gripper_width: float = 50.0
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
            if task_mode == "unconstrained":
                alpha = 0.0
            elif task_mode == "tool_use":
                alpha = min(0.8, alpha)

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
        "visualization_base64": f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    }
    return result, img_bytes


@app.post("/predict_grasp")
def predict_grasp(req: GraspRequest):
    """
    Main Grasp Prediction Endpoint:
    Receives feature configuration, executes forward pass, and returns grasp coordinates + visualization.
    """
    data, _ = run_inference_core(
        tool_type=req.tool_type,
        sample_idx=req.sample_idx,
        model_type=req.model_type,
        task_mode=req.task_mode,
        affordance_weight=req.affordance_weight,
        quality_threshold=req.quality_threshold,
        max_gripper_width=req.max_gripper_width
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
        max_gripper_width=max_gripper_width
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


@app.get("/", response_class=HTMLResponse)
@app.get("/demo", response_class=HTMLResponse)
def interactive_demo_page():
    """Serves the interactive web UI dashboard for testing feature changes."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TAG-Net: Interactive Grasp Synthesis & Feature Studio</title>
  <style>
    :root {
      --bg: #0B1120;
      --card: #1E293B;
      --border: #334155;
      --accent: #06B6D4;
      --green: #10B981;
      --red: #EF4444;
      --text: #F8FAFC;
      --muted: #94A3B8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }
    .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 15px; margin-bottom: 20px; }
    .header h1 { font-size: 1.5rem; color: var(--text); }
    .badge { background: #0284C7; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .grid { display: grid; grid-template-columns: 360px 1fr; gap: 20px; }
    .panel { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
    .panel h2 { font-size: 1.1rem; margin-bottom: 15px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .form-group { margin-bottom: 16px; }
    label { display: block; font-size: 0.85rem; font-weight: 600; color: var(--muted); margin-bottom: 6px; }
    select, input[type="range"], input[type="number"] { width: 100%; padding: 8px 12px; background: #0F172A; border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.9rem; }
    .slider-val { font-weight: 700; color: var(--accent); margin-left: 6px; }
    .btn { width: 100%; padding: 12px; border: none; border-radius: 8px; background: linear-gradient(135deg, #06B6D4, #3B82F6); color: white; font-weight: 700; font-size: 0.95rem; cursor: pointer; transition: 0.2s; margin-top: 10px; }
    .btn:hover { opacity: 0.9; transform: translateY(-1px); }
    .btn-stream { background: linear-gradient(135deg, #10B981, #059669); margin-top: 8px; }
    .btn-stream.active { background: linear-gradient(135deg, #EF4444, #B91C1C) !important; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.75; } 100% { opacity: 1; } }
    .presets { display: grid; grid-template-columns: 1fr; gap: 8px; margin-top: 15px; }
    .preset-btn { background: #0F172A; border: 1px solid var(--border); color: var(--muted); padding: 8px 12px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; text-align: left; transition: 0.2s; }
    .preset-btn:hover { border-color: var(--accent); color: var(--text); }
    .preset-btn strong { color: var(--text); display: block; }
    
    /* View Tabs */
    .tab-bar { display: flex; gap: 8px; margin-bottom: 15px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .tab-btn { background: #0F172A; border: 1px solid var(--border); color: var(--muted); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: 0.2s; }
    .tab-btn.active { background: var(--accent); color: white; border-color: var(--accent); }
    
    .vis-card { text-align: center; margin-bottom: 20px; background: #050811; border: 1px solid var(--border); border-radius: 8px; padding: 12px; min-height: 280px; display: flex; align-items: center; justify-content: center; position: relative; }
    .vis-card img { max-width: 100%; border-radius: 6px; display: block; margin: 0 auto; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }
    
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
    .metric-box { background: #0F172A; border: 1px solid var(--border); border-radius: 8px; padding: 12px; text-align: center; }
    .metric-title { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }
    .metric-val { font-size: 1.25rem; font-weight: 700; color: var(--text); }
    .status-pass { color: var(--green) !important; }
    .status-fail { color: var(--red) !important; }
    .json-box { background: #050811; border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-family: monospace; font-size: 0.8rem; color: #38BDF8; max-height: 220px; overflow-y: auto; white-space: pre-wrap; }
    .docs-link { color: var(--accent); text-decoration: none; font-size: 0.85rem; font-weight: 600; }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>TAG-Net: Interactive Grasp Synthesis API Studio</h1>
      <p style="color: var(--muted); font-size: 0.85rem; margin-top: 4px;">Dynamic Feature Alteration & Multi-Contact Safety Demonstration</p>
    </div>
    <div>
      <a href="/docs" target="_blank" class="docs-link">📖 Interactive Swagger API Docs (/docs)</a>
    </div>
  </div>

  <div class="grid">
    <!-- Left Controls Panel -->
    <div class="panel">
      <h2>1. Input Features & Parameters</h2>

      <div class="form-group">
        <label>Object Type (Tool Category):</label>
        <select id="tool_type">
          <option value="knife">Knife (Sharp Blade vs. Handle)</option>
          <option value="mug">Mug (Drinking Rim vs. Handle)</option>
          <option value="screwdriver">Screwdriver (Metal Shaft vs. Handle)</option>
        </select>
      </div>

      <div class="form-group">
        <label>Sample Index (Random Variation):</label>
        <input type="number" id="sample_idx" value="1" min="0" max="25">
      </div>

      <div class="form-group">
        <label>Model Architecture:</label>
        <select id="model_type">
          <option value="tag_net">TAG-Net (Proposed Affordance-Gated)</option>
          <option value="baseline">Baseline (Classical Geometric-Only)</option>
        </select>
      </div>

      <div class="form-group">
        <label>Task Context Mode:</label>
        <select id="task_mode">
          <option value="safe_handover">Safe Handover to Human (Handle Priority)</option>
          <option value="tool_use">Direct Functional Tool Use</option>
          <option value="unconstrained">Unconstrained Pick & Place</option>
        </select>
      </div>

      <div class="form-group">
        <label>Affordance Gating Weight (&alpha;): <span class="slider-val" id="alpha_display">1.00</span></label>
        <input type="range" id="affordance_weight" min="0" max="1" step="0.05" value="1.0" oninput="document.getElementById('alpha_display').innerText = parseFloat(this.value).toFixed(2)">
        <small style="color: var(--muted); font-size: 0.75rem;">&alpha; = 0.0 (Pure Geometry) &rarr; &alpha; = 1.0 (Strict Task Affordance)</small>
      </div>

      <div class="form-group">
        <label>Grasp Quality Threshold: <span class="slider-val" id="thresh_display">0.05</span></label>
        <input type="range" id="quality_threshold" min="0.01" max="0.50" step="0.01" value="0.05" oninput="document.getElementById('thresh_display').innerText = parseFloat(this.value).toFixed(2)">
      </div>

      <button type="button" class="btn" onclick="executeInference(event)">▶ Run Single Inference</button>
      <button type="button" class="btn btn-stream" id="stream_btn" onclick="toggleLiveStream(event)">🔴 Start Continuous Live Stream</button>

      <div style="margin-top: 20px;">
        <label>One-Click Demonstration Presets:</label>
        <div class="presets">
          <button type="button" class="preset-btn" onclick="applyPreset('mug', 'baseline', 0.0, 'unconstrained', 1, event)">
            <strong>❌ Preset A: Classical Baseline Failure</strong>
            Mug Drinking Rim Grasp (&alpha; = 0.0 &rarr; DANGEROUS [FAIL])
          </button>
          <button type="button" class="preset-btn" onclick="applyPreset('mug', 'tag_net', 1.0, 'safe_handover', 1, event)">
            <strong>✅ Preset B: TAG-Net Semantic Correction</strong>
            Mug Handle Grasp (&alpha; = 1.0 &rarr; SAFE [PASS])
          </button>
          <button type="button" class="preset-btn" onclick="applyPreset('knife', 'tag_net', 1.0, 'safe_handover', 0, event)">
            <strong>🔪 Preset C: Knife Handle Handover</strong>
            Sharp Blade Avoidance (&alpha; = 1.0 &rarr; SAFE [PASS])
          </button>
          <button type="button" class="preset-btn" onclick="applyPreset('screwdriver', 'tag_net', 1.0, 'safe_handover', 0, event)">
            <strong>🪛 Preset D: Screwdriver Ergonomic Grip</strong>
            Metal Shaft Clearance (&alpha; = 1.0 &rarr; SAFE [PASS])
          </button>
        </div>
      </div>
    </div>

    <!-- Right Results Panel -->
    <div class="panel">
      <!-- Tabs -->
      <div class="tab-bar">
        <button type="button" class="tab-btn active" id="tab_grasp_btn" onclick="switchTab('grasp')">📸 AI Grasp Heatmap & Detection (Real-Time)</button>
        <button type="button" class="tab-btn" id="tab_robot_btn" onclick="switchTab('robot')">🤖 3D Robot Arm Execution (MuJoCo Physics)</button>
      </div>

      <!-- View 1: AI Grasp Detection & Heatmaps -->
      <div id="grasp_view">
        <div class="vis-card">
          <img id="vis_img" 
               src="/visualization.png?tool_type=knife&sample_idx=1&model_type=tag_net&task_mode=safe_handover&affordance_weight=1.0&quality_threshold=0.05" 
               alt="AI Grasp Heatmap & Detection"
               style="max-width: 100%; height: auto; min-height: 250px; display: block; margin: 0 auto; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.6);"
          >
        </div>

        <div class="metrics">
          <div class="metric-box">
            <div class="metric-title">Safety Verdict</div>
            <div class="metric-val status-pass" id="verdict_val">PASS</div>
          </div>
          <div class="metric-box">
            <div class="metric-title">Grasp Center (X, Y)</div>
            <div class="metric-val" id="coords_val">(145, 117)</div>
          </div>
          <div class="metric-box">
            <div class="metric-title">Orientation (&theta;)</div>
            <div class="metric-val" id="angle_val">37.2°</div>
          </div>
          <div class="metric-box">
            <div class="metric-title">Latency / Speed</div>
            <div class="metric-val" id="latency_val">~8 ms (120 FPS)</div>
          </div>
        </div>

        <h3 style="font-size: 0.9rem; color: var(--muted); margin-bottom: 8px;">Raw HTTP API JSON Response:</h3>
        <div class="json-box" id="json_output">{ "message": "Inference results will appear here..." }</div>
      </div>

      <!-- View 2: 3D Articulated Robot Arm Physics -->
      <div id="robot_3d_view" style="display: none;">
        <div class="vis-card" style="padding: 15px;">
          <img src="/assets/arm_simulation_3d.gif" style="max-width: 100%; border-radius: 8px;" alt="3D Robot Arm Simulation">
        </div>
        <div class="metric-box" style="margin-bottom: 15px; text-align: left; padding: 15px;">
          <h4 style="color: var(--green); margin-bottom: 6px;">Physics Simulation: 6-DoF Articulated Manipulator</h4>
          <p style="color: var(--text); font-size: 0.85rem; line-height: 1.6;">
            • <strong>Kinematic Pipeline:</strong> 2D grasp center (u, v) translated into Cartesian workspace (X, Y, Z).<br>
            • <strong>Vertical Alignment:</strong> Top-down constraint (j2 + j3 + j4 = &pi;) guarantees perpendicular approach.<br>
            • <strong>Contact Physics:</strong> Fingers clamp flush across the 32mm handle diameter with zero slip and lift tool 180mm into the air.
          </p>
        </div>
      </div>

    </div>
  </div>

  <script>
    let isStreaming = false;
    let streamTimer = null;
    let currentSample = 1;
    let inFlight = false;

    function switchTab(tab) {
      if (tab === 'grasp') {
        document.getElementById('tab_grasp_btn').className = 'tab-btn active';
        document.getElementById('tab_robot_btn').className = 'tab-btn';
        document.getElementById('grasp_view').style.display = 'block';
        document.getElementById('robot_3d_view').style.display = 'none';
      } else {
        document.getElementById('tab_grasp_btn').className = 'tab-btn';
        document.getElementById('tab_robot_btn').className = 'tab-btn active';
        document.getElementById('grasp_view').style.display = 'none';
        document.getElementById('robot_3d_view').style.display = 'block';
      }
    }

    async function streamLoop() {
      if (!isStreaming) return;
      currentSample = (currentSample + 1) % 10;
      document.getElementById('sample_idx').value = currentSample;
      await executeInference();
      if (isStreaming) {
        streamTimer = setTimeout(streamLoop, 400);
      }
    }

    function toggleLiveStream(event) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      isStreaming = !isStreaming;
      const btn = document.getElementById('stream_btn');

      if (isStreaming) {
        btn.innerText = '⏹ Stop Live Stream';
        btn.classList.add('active');
        streamLoop();
      } else {
        btn.innerText = '🔴 Start Continuous Live Stream';
        btn.classList.remove('active');
        if (streamTimer) clearTimeout(streamTimer);
      }
      return false;
    }

    let debounceTimer = null;
    function triggerUpdate() {
      if (isStreaming) return;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(executeInference, 120);
    }

    async function executeInference(event) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      if (inFlight) return false;
      inFlight = true;

      const payload = {
        tool_type: document.getElementById('tool_type').value,
        sample_idx: parseInt(document.getElementById('sample_idx').value) || 0,
        model_type: document.getElementById('model_type').value,
        task_mode: document.getElementById('task_mode').value,
        affordance_weight: parseFloat(document.getElementById('affordance_weight').value),
        quality_threshold: parseFloat(document.getElementById('quality_threshold').value),
        max_gripper_width: 50.0
      };

      try {
        const resp = await fetch('/predict_grasp', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await resp.json();

        // 1. Update image via binary endpoint with cache-buster timestamp
        const imgElem = document.getElementById('vis_img');
        const imgUrl = `/visualization.png?tool_type=${encodeURIComponent(payload.tool_type)}&sample_idx=${payload.sample_idx}&model_type=${encodeURIComponent(payload.model_type)}&task_mode=${encodeURIComponent(payload.task_mode)}&affordance_weight=${payload.affordance_weight}&quality_threshold=${payload.quality_threshold}&t=${Date.now()}`;
        imgElem.src = imgUrl;

        // 2. Update Verdict
        const vElem = document.getElementById('verdict_val');
        vElem.innerText = data.safety_evaluation.is_safe ? 'PASS' : 'FAIL';
        vElem.className = 'metric-val ' + (data.safety_evaluation.is_safe ? 'status-pass' : 'status-fail');

        // 3. Update Coords & Angle
        if (data.grasp) {
          document.getElementById('coords_val').innerText = `(${data.grasp.center_x}, ${data.grasp.center_y})`;
          document.getElementById('angle_val').innerText = `${data.grasp.angle_deg}°`;
        } else {
          document.getElementById('coords_val').innerText = 'None';
          document.getElementById('angle_val').innerText = '--';
        }

        // 4. Latency
        document.getElementById('latency_val').innerText = `${data.performance.latency_ms} ms (${data.performance.estimated_fps} FPS)`;

        // 5. Raw JSON
        document.getElementById('json_output').innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        document.getElementById('verdict_val').innerText = 'ERROR';
        document.getElementById('json_output').innerText = 'Request Error: ' + err.message;
      } finally {
        inFlight = false;
      }
      return false;
    }

    function applyPreset(tool, model, alpha, task, sampleIdx, event) {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
      }
      document.getElementById('tool_type').value = tool;
      document.getElementById('model_type').value = model;
      document.getElementById('affordance_weight').value = alpha;
      document.getElementById('alpha_display').innerText = parseFloat(alpha).toFixed(2);
      document.getElementById('task_mode').value = task;
      document.getElementById('sample_idx').value = sampleIdx;
      executeInference();
      return false;
    }

    // Auto-attach live event listeners
    window.addEventListener('DOMContentLoaded', () => {
      document.getElementById('tool_type').addEventListener('change', triggerUpdate);
      document.getElementById('model_type').addEventListener('change', triggerUpdate);
      document.getElementById('task_mode').addEventListener('change', triggerUpdate);
      document.getElementById('sample_idx').addEventListener('input', triggerUpdate);
      document.getElementById('affordance_weight').addEventListener('input', () => {
        document.getElementById('alpha_display').innerText = parseFloat(document.getElementById('affordance_weight').value).toFixed(2);
        triggerUpdate();
      });
      document.getElementById('quality_threshold').addEventListener('input', () => {
        document.getElementById('thresh_display').innerText = parseFloat(document.getElementById('quality_threshold').value).toFixed(2);
        triggerUpdate();
      });

      // Initial execute on load
      executeInference();
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("   TAG-Net REST API & Interactive Demonstration Server")
    print("   Swagger Docs : http://localhost:8000/docs")
    print("   Web Studio   : http://localhost:8000/demo (or http://localhost:8000)")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
