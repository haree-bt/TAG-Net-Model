"""
Generate TAG-Net Presentation Images PDF & Studio Mockup
========================================================
Compiles all high-resolution visual assets into a single clean PDF:
`TAG_Net_Presentation_Images.pdf` with slide mapping labels.
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(ROOT_DIR, "assets", "presentation_slides")
COLAB_DIR = os.path.join(ROOT_DIR, "colab_eda_results")
PDF_PATH = os.path.join(ROOT_DIR, "TAG_Net_Presentation_Images.pdf")

os.makedirs(ASSETS_DIR, exist_ok=True)


def create_studio_dashboard_mockup():
    """Renders a visual graphic of the Web Studio Dashboard interface."""
    out_path = os.path.join(ASSETS_DIR, "studio_dashboard_interface.png")
    
    # Load visualization image if available, else generate
    vis_path = os.path.join(COLAB_DIR, "plot5_qualitative_explainability_matrix.png")
    sim_path = os.path.join(ASSETS_DIR, "3d_knife_handover_safe.png")

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor='#07090E')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#07090E')
    ax.axis('off')

    # Header bar
    ax.text(0.04, 0.94, "⚡ TAG-NET STUDIO // 6-DoF ROBOTIC MANIPULATION DIGITAL TWIN",
            color='#38BDF8', fontsize=16, fontweight='bold', transform=ax.transAxes)
    ax.text(0.04, 0.90, "FASTAPI ASYNC BACKEND • 17.69 ms INFERENCE • 56.5 FPS • MUJOCO 3.15 HD",
            color='#94A3B8', fontsize=10, fontfamily='monospace', transform=ax.transAxes)

    # Status pills
    ax.text(0.72, 0.93, "STATUS: ONLINE (56.5 FPS)", color='#10B981', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#0D121C', edgecolor='#10B981', lw=1.5), transform=ax.transAxes)
    ax.text(0.88, 0.93, "ENGINE: MUJOCO HD", color='#38BDF8', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='#0D121C', edgecolor='#38BDF8', lw=1.5), transform=ax.transAxes)

    # Draw two main screen cards
    # Left: Perception
    ax.text(0.04, 0.83, "VIEWPORT 1: AGAM NEURAL AFFORDANCE PERCEPTION", color='#F8FAFC', fontsize=12, fontweight='bold', transform=ax.transAxes)
    if os.path.exists(vis_path):
        img_vis = Image.open(vis_path)
        ax_in1 = fig.add_axes([0.04, 0.16, 0.44, 0.64])
        ax_in1.imshow(img_vis)
        ax_in1.axis('off')

    # Right: MuJoCo Live Stream
    ax.text(0.52, 0.83, "VIEWPORT 2: 6-DoF MUJOCO PHYSICAL EXECUTION (16:9 HD)", color='#F8FAFC', fontsize=12, fontweight='bold', transform=ax.transAxes)
    if os.path.exists(sim_path):
        img_sim = Image.open(sim_path)
        ax_in2 = fig.add_axes([0.52, 0.16, 0.44, 0.64])
        ax_in2.imshow(img_sim)
        ax_in2.axis('off')

    # Bottom control indicators
    controls_text = "INTERACTIVE DOCK: [ MODEL: TAG-Net (AGAM) ]   [ TOOL: Chef Knife ]   [ TASK: Safe Handover ]   [ ORIENTATION: 0° (Wrist Roll j5) ]   [ α: 1.00 ]"
    ax.text(0.50, 0.06, controls_text, color='#38BDF8', fontsize=10, fontfamily='monospace', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#0D121C', edgecolor='#1E293B', lw=1), transform=ax.transAxes)

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.2, facecolor='#07090E')
    plt.close(fig)
    print(f"Created: {out_path}")
    return out_path


def build_images_pdf():
    mockup_path = create_studio_dashboard_mockup()

    # Image registry: (Title, Slide to Use On, Description, File Path)
    images_to_compile = [
        (
            "IMAGE 1: Conventional Baseline Failure (Blade Grasp)",
            "SLIDE 2: The Critical Failure of Conventional Grasping",
            "Shows conventional model (GG-CNN/Dex-Net philosophy) clamping the dangerous sharp cutting blade.",
            os.path.join(ASSETS_DIR, "3d_knife_blade_hazard.png")
        ),
        (
            "IMAGE 2: TAG-Net Architecture & AGAM Dataflow",
            "SLIDE 4: Proposed TAG-Net & AGAM Architecture",
            "Full neural dataflow: 4-channel RGB-D input, ResNet backbone, AGAM attention branch, Hadamard fusion, and 6-DoF twin.",
            os.path.join(ASSETS_DIR, "tagnet_architecture_pipeline.png")
        ),
        (
            "IMAGE 3: TAG-Net Safe Knife Handover",
            "SLIDE 6: Safe Tool Handover Demonstration (Chef Knife)",
            "MuJoCo robotic arm securely grasping the knife handle and presenting it directly to the human palm.",
            os.path.join(ASSETS_DIR, "3d_knife_handover_safe.png")
        ),
        (
            "IMAGE 4: Rotated Mug Handle Grasp (45° Wrist Roll)",
            "SLIDE 7: Handling Rotated Objects & Dynamic Wrist Alignment",
            "Robot wrist roll rotates 45° to match rotated mug, avoiding the sanitary drinking rim and grasping the loop.",
            os.path.join(ASSETS_DIR, "3d_mug_rotated_safe.png")
        ),
        (
            "IMAGE 5: Screwdriver Power Grip for Work/Torque",
            "SLIDE 8: Functional Tool Manipulation (Screwdriver)",
            "Grasping the textured rubber power grip instead of the metal shaft to apply clockwise fastening torque.",
            os.path.join(ASSETS_DIR, "3d_screwdriver_torque_safe.png")
        ),
        (
            "IMAGE 6: Side-by-Side Visual Proof Matrix",
            "SLIDE 9: Qualitative Comparison Matrix (Knife, Mug, Tool)",
            "Multi-tool comparison: RGB Camera vs GT Affordance Mask vs Conventional Failure (Red) vs TAG-Net Safe (Green).",
            os.path.join(COLAB_DIR, "plot5_qualitative_explainability_matrix.png")
        ),
        (
            "IMAGE 7: Executive Web Studio Dashboard Interface",
            "SLIDE 10: Real-Time Web Studio & API Architecture",
            "Live demonstration UI showing synchronized neural heatmap perception and 6-DoF MuJoCo simulation stream.",
            mockup_path
        )
    ]

    print(f"\n>>> Compiling {len(images_to_compile)} high-resolution images into PDF...")
    with PdfPages(PDF_PATH) as pdf:
        # Cover / Index Page
        fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=150, facecolor='#07090E') # A4 Landscape
        ax.set_facecolor('#07090E')
        ax.axis('off')

        ax.text(0.5, 0.90, "TAG-NET PRESENTATION VISUAL ASSETS CATALOG", color='#38BDF8', fontsize=18, fontweight='bold', ha='center')
        ax.text(0.5, 0.85, "High-Resolution Images for Slides • Exact Slide Mapping & Captions", color='#94A3B8', fontsize=11, fontfamily='monospace', ha='center')

        y = 0.75
        for idx, (title, slide_info, desc, path) in enumerate(images_to_compile):
            exists = "✓" if os.path.exists(path) else "✗"
            color = '#10B981' if exists == "✓" else '#EF4444'
            ax.text(0.08, y, f"[{exists}] {title}", color='#FFFFFF', fontsize=11, fontweight='bold')
            ax.text(0.12, y - 0.035, f"→ Target: {slide_info}", color='#38BDF8', fontsize=9.5, fontfamily='monospace')
            ax.text(0.12, y - 0.065, f"→ Details: {desc}", color='#94A3B8', fontsize=8.5)
            y -= 0.10

        ax.text(0.5, 0.05, "TAG-Net Project • Task-Aware Affordance-Gated Grasp Network", color='#64748B', fontsize=9, ha='center')
        pdf.savefig(fig, facecolor='#07090E')
        plt.close(fig)

        # Image Pages
        for title, slide_info, desc, path in images_to_compile:
            if not os.path.exists(path):
                print(f"Warning: {path} not found. Skipping.")
                continue

            fig = plt.figure(figsize=(11.69, 8.27), dpi=150, facecolor='#07090E')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#07090E')
            ax.axis('off')

            # Header text
            ax.text(0.05, 0.94, title.upper(), color='#38BDF8', fontsize=14, fontweight='bold', transform=ax.transAxes)
            ax.text(0.05, 0.90, f"WHERE TO USE IN PPT: {slide_info}", color='#10B981', fontsize=10.5, fontfamily='monospace', fontweight='bold', transform=ax.transAxes)
            ax.text(0.05, 0.87, f"EXPLANATION: {desc}", color='#94A3B8', fontsize=9, transform=ax.transAxes)

            # Embed Image
            img = Image.open(path)
            ax_img = fig.add_axes([0.05, 0.06, 0.90, 0.78])
            ax_img.imshow(img)
            ax_img.axis('off')

            pdf.savefig(fig, facecolor='#07090E')
            plt.close(fig)
            print(f"  + Added: {title}")

    print(f"\n[SUCCESS] PDF catalog compiled cleanly at:\n   {PDF_PATH}\n")


if __name__ == "__main__":
    build_images_pdf()
