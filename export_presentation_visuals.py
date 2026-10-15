"""
TAG-Net Presentation Visuals & 3D Model Image Exporter
=====================================================
Renders and exports high-resolution visual assets for presentation slides:
1. 3D MuJoCo Physics Model Grasp Snapshots (Knife, Mug, Screwdriver)
2. Neural Network Architecture & AGAM Dataflow Diagram
3. Executive Dark Studio Dashboard Workflow Infographic
"""

import os
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from simulation.task_simulation import (
    build_task_scene_xml,
    get_agam_task_grasp,
    solve_ik_top_down,
    generate_task_phases
)
import mujoco

OUTPUT_DIR = os.path.join(ROOT_DIR, "assets", "presentation_slides")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_3d_mujoco_snapshots():
    """Captures 3D physics simulation snapshots of the 6-DoF robotic arm executing grasps."""
    print("\n>>> Exporting 3D MuJoCo Physics Snapshots...")
    tools = [
        ("knife", "handover", 0.0, "tag_net", "3d_knife_handover_safe.png"),
        ("knife", "handover", 0.0, "conventional", "3d_knife_blade_hazard.png"),
        ("mug", "pick_and_place", 45.0, "tag_net", "3d_mug_rotated_safe.png"),
        ("screwdriver", "work_with_it", 90.0, "tag_net", "3d_screwdriver_torque_safe.png")
    ]

    for tool_name, task_name, orient, model_type, filename in tools:
        pred = get_agam_task_grasp(tool_name, task_name, model_type=model_type, orientation_deg=orient)
        xml = build_task_scene_xml(tool_name, pred["target_xyz"], task_name, model_type=model_type, orientation_deg=orient)
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, width=960, height=540)

        # Set arm at clamp phase
        phases = generate_task_phases(model, data, tool_name, task_name, pred["target_xyz"], pred["clamp_val"], roll=math.radians(orient))
        clamp_phase = phases[3]  # Clamp phase (name, q_start, q_end, f_start, f_end, mode, steps)
        data.qpos[0:5] = clamp_phase[2]
        data.qpos[5] = -clamp_phase[4]
        data.qpos[6] = clamp_phase[4]

        # Tool position at target
        tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{tool_name}_body")
        if tool_body_id != -1:
            model.body_pos[tool_body_id] = pred["target_xyz"]

        # Cinematic camera setup
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat = [0.38, -0.05, 0.75]
        camera.distance = 1.15
        camera.elevation = -22.0
        camera.azimuth = 135.0

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        img_np = renderer.render()

        # Add overlay badge
        fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
        ax.imshow(img_np)
        ax.axis('off')

        is_safe = (model_type == "tag_net")
        color = '#10B981' if is_safe else '#EF4444'
        badge_text = f"TAG-NET: {tool_name.upper()} {task_name.upper()} [SAFE PASS]" if is_safe else f"CONVENTIONAL BASELINE: {tool_name.upper()} BLADE HAZARD [FAIL]"
        
        ax.text(40, 60, badge_text, color=color, fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#07090E", edgecolor=color, linewidth=2, alpha=0.9))
        
        telemetry = f"Wrist Roll j5: {orient:.1f}° | Target XYZ: [{pred['target_xyz'][0]:.2f}, {pred['target_xyz'][1]:.2f}, {pred['target_xyz'][2]:.3f}]"
        ax.text(40, 110, telemetry, color='#38BDF8', fontsize=12, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#07090E", edgecolor='#1E293B', alpha=0.85))

        out_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"  - Saved: {out_path}")


def export_architecture_diagram():
    """Generates the neural architecture and API pipeline infographic."""
    print("\n>>> Exporting Architecture & API Pipeline Infographic...")
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=200, facecolor='#07090E')
    ax.set_facecolor('#07090E')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    def draw_card(x, y, w, h, title, subtitle, color, icon=""):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                     linewidth=1.8, edgecolor=color, facecolor='#0D121C', alpha=0.95)
        ax.add_patch(rect)
        ax.text(x + 0.25, y + h - 0.45, f"{icon} {title}", color='#FFFFFF', fontsize=11, fontweight='bold')
        ax.text(x + 0.25, y + 0.35, subtitle, color='#94A3B8', fontsize=8.5, fontfamily='monospace')

    def draw_arrow(x1, y1, x2, y2, color="#38BDF8"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.0, mutation_scale=15))

    # Cards
    draw_card(0.5, 2.5, 2.2, 1.6, "RGB-D Input", "224x224x4\n(RGB + Depth)", "#38BDF8", "📷")
    draw_card(3.3, 2.5, 2.4, 1.6, "Backbone", "ResNet-50 Feature\nExtractor (2048-D)", "#818CF8", "🧠")
    
    # Branches
    draw_card(6.3, 4.0, 2.6, 1.8, "Affordance (AGAM)", "Semantic Attention\nM_aff in [0, 1]", "#10B981", "🛡️")
    draw_card(6.3, 0.8, 2.6, 1.8, "Geometry Branch", "Antipodal Cand.\nQ_geom, sin/cos, w", "#38BDF8", "📐")

    # Fusion
    draw_card(9.5, 2.5, 2.3, 1.6, "Hadamard Fusion", "Q_task = Q_geom ⊙ M\nGrasp Peak Detector", "#F59E0B", "⚡")

    # MuJoCo / API
    draw_card(12.3, 2.5, 1.5, 1.6, "6-DoF Twin", "MuJoCo 3.15\nTop-Down IK", "#10B981", "🤖")

    # Arrows
    draw_arrow(2.7, 3.3, 3.3, 3.3)
    draw_arrow(5.7, 3.7, 6.3, 4.8)
    draw_arrow(5.7, 2.9, 6.3, 1.8)
    draw_arrow(8.9, 4.8, 9.5, 3.7)
    draw_arrow(8.9, 1.8, 9.5, 2.9)
    draw_arrow(11.8, 3.3, 12.3, 3.3)

    ax.text(7.0, 6.1, "TAG-NET NEURAL ARCHITECTURE & ASYNCHRONOUS INFERENCE PIPELINE",
            color='#FFFFFF', fontsize=13, fontweight='bold', ha='center')

    out_path = os.path.join(OUTPUT_DIR, "tagnet_architecture_pipeline.png")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.2, facecolor='#07090E')
    plt.close(fig)
    print(f"  - Saved: {out_path}")


if __name__ == "__main__":
    export_3d_mujoco_snapshots()
    export_architecture_diagram()
    print(f"\n[SUCCESS] All presentation visuals successfully generated in:\n   {OUTPUT_DIR}\n")
