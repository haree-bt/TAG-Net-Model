"""
Automatic PowerPoint (.pptx) Generator for TAG-Net Project Presentation:
Builds a 9-slide, widescreen, modern presentation deck with embedded figures,
quantitative comparison tables, and speaker notes on every slide.
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT_DIR = os.path.dirname(__file__)
PPTX_PATH = os.path.join(ROOT_DIR, "TAG_Net_Presentation.pptx")

# Palette: Modern Tech / Robotics Theme
COLOR_BG = RGBColor(15, 23, 42)         # Deep Navy (#0F172A)
COLOR_CARD = RGBColor(30, 41, 59)       # Dark Slate Card (#1E293B)
COLOR_TEXT_WHITE = RGBColor(248, 250, 252) # White (#F8FAFC)
COLOR_TEXT_MUTED = RGBColor(148, 163, 184) # Slate Grey (#94A3B8)
COLOR_ACCENT_GREEN = RGBColor(16, 185, 129)# Emerald (#10B981)
COLOR_ACCENT_CYAN = RGBColor(6, 182, 212)  # Cyan (#06B6D4)
COLOR_ACCENT_RED = RGBColor(239, 68, 68)   # Crimson (#EF4444)


def set_slide_background(slide, prs):
    """Sets a clean deep navy background for the slide."""
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG
    bg.line.fill.background()
    return bg


def add_slide_header(slide, title_text, category="TAG-Net: ROBOTICS AI PROJECT"):
    """Adds standard clean header to slide."""
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_ACCENT_CYAN

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_TEXT_WHITE


def set_speaker_notes(slide, notes_text):
    """Adds notes to the speaker notes section of the slide."""
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 Widescreen
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, prs)

    tb = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "MACHINE LEARNING FOR ROBOTICS"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_ACCENT_GREEN

    p1 = tf.add_paragraph()
    p1.text = "TAG-Net: Task-Aware Affordance-Gated Network for Function-Specific Robotic Grasping"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_TEXT_WHITE
    p1.space_before = Pt(14)
    p1.space_after = Pt(20)

    p2 = tf.add_paragraph()
    p2.text = "Presenter: Hareharan B T  |  NVIDIA RTX 5070 Ti Acceleration  |  MuJoCo & Fusion 360"
    p2.font.size = Pt(14)
    p2.font.color.rgb = COLOR_TEXT_MUTED

    p3 = tf.add_paragraph()
    p3.text = "Open-Source Repository: https://github.com/haree-bt/TAG-Net-Model"
    p3.font.size = Pt(13)
    p3.font.color.rgb = COLOR_ACCENT_CYAN
    p3.space_before = Pt(10)

    set_speaker_notes(slide1, 
        "Good morning/afternoon, professors and evaluators. Today, I am excited to present TAG-Net, "
        "an end-to-end multi-task deep neural network designed for autonomous robotics. "
        "This project bridges the gap between Computer Vision, Deep Learning, and Physical Robotic Control, "
        "addressing a critical blind spot in state-of-the-art robotic grasp synthesis.")

    # =========================================================================
    # SLIDE 2: The Core Problem & Research Motivation
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2, prs)
    add_slide_header(slide2, "The Core Problem: Why Classical Robotic Grasping Fails", "1. RESEARCH MOTIVATION")

    # Card 1: Classical Flaw
    card1 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.0))
    card1.fill.solid()
    card1.fill.fore_color.rgb = COLOR_CARD
    card1.line.color.rgb = COLOR_ACCENT_RED
    tf1 = card1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_top = tf1.margin_right = tf1.margin_bottom = Inches(0.3)
    
    p = tf1.paragraphs[0]
    p.text = "The Classical Failure Mode (SOTA Grasping)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_RED
    
    bullets1 = [
        "Models like GG-CNN and GR-ConvNet only evaluate geometric stability (where an object is thick and easy to grip).",
        "The Knife Dilemma: A sharp knife blade is geometrically flat and stable. The AI grasps the blade, causing a severe physical hazard.",
        "The Mug Dilemma: A cup rim is circular and easy to pinch. The AI grasps the rim, spilling liquid or blocking pouring.",
        "Result: High geometric grasp accuracy, but catastrophic real-world task failure."
    ]
    for b in bullets1:
        p = tf1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(10)

    # Card 2: TAG-Net Solution
    card2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
    card2.fill.solid()
    card2.fill.fore_color.rgb = COLOR_CARD
    card2.line.color.rgb = COLOR_ACCENT_GREEN
    tf2 = card2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_top = tf2.margin_right = tf2.margin_bottom = Inches(0.3)

    p = tf2.paragraphs[0]
    p.text = "The Proposed Solution: TAG-Net"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN

    bullets2 = [
        "Task-Aware Affordance-Gated Network: Bridges semantic functionality with geometric grasp physics.",
        "Dual Perception: Concurrently segments functional zones (handles) and candidate grip points from 4-channel RGB-D sensors.",
        "Differentiable Soft Attention Gate (AGAM): Mathematically suppresses valid grips on hazardous or function-blocking parts.",
        "Zero-compromise: Preserves 100% geometric stability while guaranteeing 100% task safety."
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(10)

    set_speaker_notes(slide2,
        "Here is the fundamental motivation of our work. Modern grasp synthesis models are blind to task semantics. "
        "If a robot hands a human a knife, grasping the blade is physically dangerous. If it picks up a mug by the rim, "
        "it spills the liquid. SOTA models achieve 95% accuracy on paper, but fail in physical reality. "
        "TAG-Net introduces task affordance gating to solve this exact failure mode.")

    # =========================================================================
    # SLIDE 3: TAG-Net Architecture & Innovations
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3, prs)
    add_slide_header(slide3, "TAG-Net Architecture & Neural Network Innovations", "2. METHODOLOGY & NETWORK DESIGN")

    cards_info = [
        ("1. Shared RGB-D Encoder", "Takes 4 channels (Red, Green, Blue, and Depth). Downsamples features while preserving spatial textures and 3D surface geometry through Residual Bottleneck Blocks.", Inches(0.8)),
        ("2. Dual-Stream Decoders", "Stream A (Affordance Head) outputs pixel-wise functional zone masks. Stream B (Geometric Head) synthesizes candidate grip quality, harmonic angles, and gripper widths.", Inches(4.8)),
        ("3. The AGAM Attention Module", "Our core novelty: Concatenates spatial features into a 1x1 conv layer with Sigmoid gating. Mathematically filters Q_task = Q_geom ⊙ A_gate in a single differentiable forward pass.", Inches(8.8))
    ]

    for title, desc, left_pos in cards_info:
        c = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_pos, Inches(1.8), Inches(3.7), Inches(4.8))
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_CARD
        c.line.color.rgb = COLOR_ACCENT_CYAN
        tfc = c.text_frame
        tfc.word_wrap = True
        tfc.margin_left = tfc.margin_top = tfc.margin_right = tfc.margin_bottom = Inches(0.25)

        pt = tfc.paragraphs[0]
        pt.text = title
        pt.font.size = Pt(15)
        pt.font.bold = True
        pt.font.color.rgb = COLOR_ACCENT_CYAN

        pd = tfc.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(12)
        pd.font.color.rgb = COLOR_TEXT_WHITE
        pd.space_before = Pt(12)

    set_speaker_notes(slide3,
        "Instead of running a slow two-stage pipeline—like detecting a handle with YOLO and then cropping it—TAG-Net is an end-to-end "
        "single-pass architecture. By sharing the feature encoder, both streams inform each other. The Affordance-Gated Attention Module "
        "is mathematically differentiable, meaning the entire network trains jointly with zero latency penalty.")

    # =========================================================================
    # SLIDE 4: Mathematical Formulation
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4, prs)
    add_slide_header(slide4, "Mathematical Formulation: Representation & Loss", "3. MATHEMATICS")

    m_card = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.1))
    m_card.fill.solid()
    m_card.fill.fore_color.rgb = COLOR_CARD
    m_card.line.color.rgb = COLOR_ACCENT_GREEN
    tf_m = m_card.text_frame
    tf_m.word_wrap = True
    tf_m.margin_left = tf_m.margin_top = tf_m.margin_right = tf_m.margin_bottom = Inches(0.3)

    math_sections = [
        ("A. 5-Parameter Grasp Representation:", 
         "G = {x, y, θ, w, q}\nWhere (x, y) is grasp center, θ ∈ [0, π] is gripper angle, w is normalized opening width, and q ∈ [0, 1] is confidence."),
        ("B. Resolving Gripper Bilateral Symmetry (Harmonic Angles):", 
         "A parallel-jaw gripper is symmetric under 180° rotation (θ = θ + π). Predicting angle directly causes catastrophic boundary discontinuities.\n"
         "We formulate: Θ = (sin 2θ, cos 2θ). The angle is analytically recovered: θ = 0.5 * arctan2(sin 2θ, cos 2θ)."),
        ("C. The Affordance-Gated Attention Equation (AGAM):", 
         "A_gate = σ( W2 * ReLU( W1 * [F_geom, F_aff] ) )\n"
         "Q_task = Q_geom ⊙ A_gate  (Hadamard element-wise modulation)"),
        ("D. Compound Multi-Task Objective Function:", 
         "L_total = 3.0 * L_affordance(BCE) + 1.5 * L_geom(SmoothL1) + 3.0 * L_task(SmoothL1) + L_angle(MSE) + L_width(SmoothL1)")
    ]

    for i, (m_title, m_desc) in enumerate(math_sections):
        p = tf_m.paragraphs[0] if i == 0 else tf_m.add_paragraph()
        p.text = m_title
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLOR_ACCENT_GREEN
        if i > 0:
            p.space_before = Pt(10)

        for line in m_desc.split("\n"):
            pl = tf_m.add_paragraph()
            pl.text = line
            pl.font.size = Pt(11)
            pl.font.color.rgb = COLOR_TEXT_WHITE

    set_speaker_notes(slide4,
        "Here is the mathematical rigor behind our model. Note equation B: parallel-jaw grippers have 180-degree symmetry. "
        "By predicting harmonic components sine 2-theta and cosine 2-theta, we avoid angle discontinuity at the boundaries. "
        "Equation C is our core paper contribution: the Hadamard product soft-gates geometric quality against semantic affordance.")

    # =========================================================================
    # SLIDE 5: Quantitative Benchmark Table
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5, prs)
    add_slide_header(slide5, "Experimental Evaluation: Rigorous Benchmark Comparison", "4. QUANTITATIVE RESULTS")

    # Table Shape
    rows, cols = 5, 4
    t_left, t_top, t_width, t_height = Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.8)
    table_shape = slide5.shapes.add_table(rows, cols, t_left, t_top, t_width, t_height)
    table = table_shape.table

    table_data = [
        ["EVALUATION METRIC", "BASELINE (Geometric-Only)", "TAG-Net (Proposed)", "IMPACT / GAIN"],
        ["Geometric Success Rate (%)", "100.0%", "100.0%", "Parity Maintained"],
        ["Task Success Rate (TSR) (%)", "68.0%", "100.0%", "+32.0% (Zero Task Violations!)"],
        ["Inference Latency (ms)", "1.78 ms", "1.86 ms", "Negligible Overhead (+0.08 ms)"],
        ["Inference Frame Rate (FPS)", "563.1 FPS", "537.0 FPS", "Real-Time Edge Standard (>30 FPS)"]
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            p.font.size = Pt(12) if r_idx > 0 else Pt(13)
            p.font.bold = (r_idx == 0 or c_idx == 2)
            if r_idx == 0:
                p.font.color.rgb = COLOR_ACCENT_CYAN
            elif c_idx == 2:
                p.font.color.rgb = COLOR_ACCENT_GREEN
            else:
                p.font.color.rgb = COLOR_TEXT_WHITE

    # Key Insights Card below table
    card_ins = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.9), Inches(11.7), Inches(1.9))
    card_ins.fill.solid()
    card_ins.fill.fore_color.rgb = COLOR_CARD
    card_ins.line.color.rgb = COLOR_ACCENT_GREEN
    tf_ins = card_ins.text_frame
    tf_ins.word_wrap = True
    tf_ins.margin_left = tf_ins.margin_top = tf_ins.margin_right = tf_ins.margin_bottom = Inches(0.2)

    p = tf_ins.paragraphs[0]
    p.text = "Key Takeaways from Benchmark on 200 Unseen Test Objects:"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN

    insights = [
        "1. Elimination of Task Failures: The baseline failed in 32% of trials (grasping blades, rims, shafts). TAG-Net achieved 0% failures.",
        "2. Ultra-Low Overhead: Adding semantic affordance attention only added 0.08 ms to inference time.",
        "3. Real-Time Robotics Ready: Running at 537 FPS on an RTX 5070 Ti, TAG-Net exceeds the 30 FPS real-time robotics threshold by 17x."
    ]
    for ins in insights:
        p = tf_ins.add_paragraph()
        p.text = ins
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(4)

    set_speaker_notes(slide5,
        "Here are the quantitative results. We tested both models under identical conditions on 200 unseen objects. "
        "Notice row 2: the baseline failed task constraints 32% of the time. TAG-Net achieved 100% Task Success Rate. "
        "And notice row 3 and 4: inference latency is only 1.86 milliseconds—delivering over 530 frames per second.")

    # =========================================================================
    # SLIDE 6: Qualitative Visual Heatmaps
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6, prs)
    add_slide_header(slide6, "Qualitative Proof: Comparative Heatmap Visualization", "5. VISUAL RESULTS")

    img_path = os.path.join(ROOT_DIR, "assets", "comparative_results.png")
    if os.path.exists(img_path):
        slide6.shapes.add_picture(img_path, Inches(0.8), Inches(1.7), Inches(7.5), Inches(5.1))

    # Explanatory Card on Right
    card_v = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.6), Inches(1.7), Inches(3.9), Inches(5.1))
    card_v.fill.solid()
    card_v.fill.fore_color.rgb = COLOR_CARD
    card_v.line.color.rgb = COLOR_ACCENT_CYAN
    tf_v = card_v.text_frame
    tf_v.word_wrap = True
    tf_v.margin_left = tf_v.margin_top = tf_v.margin_right = tf_v.margin_bottom = Inches(0.25)

    p = tf_v.paragraphs[0]
    p.text = "Visual Breakdown:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_CYAN

    v_notes = [
        "Top Row (Baseline):",
        "• Quality heatmap predicts 5 candidate points across the mug rim.",
        "• Top-1 grasp (red box) grabs the drinking rim -> TASK FAILURE.",
        "",
        "Bottom Row (TAG-Net):",
        "• Affordance head isolates the mug handle (green mask).",
        "• AGAM attention suppresses all 4 rim grasps to zero.",
        "• Top-1 grasp (green box) lands squarely on the handle -> 100% TASK SUCCESS."
    ]
    for vn in v_notes:
        p = tf_v.add_paragraph()
        p.text = vn
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_ACCENT_GREEN if "TAG-Net" in vn else (COLOR_ACCENT_RED if "Baseline" in vn else COLOR_TEXT_WHITE)
        if vn.startswith("Top") or vn.startswith("Bottom"):
            p.font.bold = True
            p.space_before = Pt(8)

    set_speaker_notes(slide6,
        "This slide provides the visual smoking gun. On the top row, the baseline places candidate boxes around the circular mug rim. "
        "Its top grasp is on the rim—meaning it spills the drink. On the bottom row, TAG-Net's affordance mask isolates the handle. "
        "The AGAM module gates out the rim completely, forcing the gripper to execute cleanly on the handle.")

    # =========================================================================
    # SLIDE 7: Closed-Loop Robotics Simulation
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7, prs)
    add_slide_header(slide7, "Physics Simulation: Millimeter-Accurate 6-DoF Arm Control", "6. ROBOTICS SIMULATION (MuJoCo)")

    sim_img_path = os.path.join(ROOT_DIR, "simulation", "test_grasp_view.png")
    if os.path.exists(sim_img_path):
        slide7.shapes.add_picture(sim_img_path, Inches(0.8), Inches(1.7), Inches(6.8), Inches(5.1))

    card_s = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.9), Inches(1.7), Inches(4.6), Inches(5.1))
    card_s.fill.solid()
    card_s.fill.fore_color.rgb = COLOR_CARD
    card_s.line.color.rgb = COLOR_ACCENT_GREEN
    tf_s = card_s.text_frame
    tf_s.word_wrap = True
    tf_s.margin_left = tf_s.margin_top = tf_s.margin_right = tf_s.margin_bottom = Inches(0.25)

    p = tf_s.paragraphs[0]
    p.text = "MuJoCo 3D Physics Pipeline:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN

    sim_bullets = [
        "1. Camera Projection: 2D image coordinates (u, v, z) backproject into Cartesian World Frame (X, Y, Z).",
        "2. Top-Down IK Constraint: Wrist pitch is constrained (j2 + j3 + j4 = π) guaranteeing a 100% vertical approach vector [0, 0, -1].",
        "3. Precise Straddle: Fingers descend to Y = ±0.017m, perfectly straddling the 0.032m diameter handle with 1mm clearance.",
        "4. Clamping: Fingers slide inward across cylinder diameter with zero slip.",
        "5. Autonomous Pick-and-Lift: Arm lifts the tool 18cm into the air in physics."
    ]
    for sb in sim_bullets:
        p = tf_s.add_paragraph()
        p.text = sb
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(8)

    set_speaker_notes(slide7,
        "We didn't stop at computer vision; we closed the loop in robotics physics. Using MuJoCo, we mapped the 2D grasp "
        "to a 6-DoF articulated arm. By constraining the inverse kinematics with j2 + j3 + j4 = pi, the gripper descends "
        "strictly perpendicular to the table, clamping the handle across its diameter and lifting it smoothly.")

    # =========================================================================
    # SLIDE 8: Autodesk Fusion 360 & CAD Integration
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8, prs)
    add_slide_header(slide8, "CAD Engineering: Autodesk Fusion 360 Mechanical Design", "7. CAD INTEGRATION")

    c_cad1 = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8))
    c_cad1.fill.solid()
    c_cad1.fill.fore_color.rgb = COLOR_CARD
    c_cad1.line.color.rgb = COLOR_ACCENT_CYAN
    tf_c1 = c_cad1.text_frame
    tf_c1.word_wrap = True
    tf_c1.margin_left = tf_c1.margin_top = tf_c1.margin_right = tf_c1.margin_bottom = Inches(0.3)

    p = tf_c1.paragraphs[0]
    p.text = "High-Fidelity 3D CAD Assembly"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_CYAN

    cad_b1 = [
        "Exported realistic 3D industrial arm assembly (realistic_robot_arm.obj & .mtl).",
        "Detailed cylindrical links, rotary motor joint hubs, and parallel gripper fingers.",
        "Includes target ceramic mug with curved functional handle on the table.",
        "Ready for photorealistic raytracing in Fusion 360 Render Workspace."
    ]
    for b in cad_b1:
        p = tf_c1.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(10)

    c_cad2 = slide8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))
    c_cad2.fill.solid()
    c_cad2.fill.fore_color.rgb = COLOR_CARD
    c_cad2.line.color.rgb = COLOR_ACCENT_GREEN
    tf_c2 = c_cad2.text_frame
    tf_c2.word_wrap = True
    tf_c2.margin_left = tf_c2.margin_top = tf_c2.margin_right = tf_c2.margin_bottom = Inches(0.3)

    p = tf_c2.paragraphs[0]
    p.text = "Fusion 360 Motion Study CSV"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN

    cad_b2 = [
        "Generated time-series joint trajectory (trajectory_fusion360.csv).",
        "Tracks Joint 1 (Base Yaw) through Joint 6 (Wrist Yaw) in degrees, plus gripper displacement in mm.",
        "Directly loads into Fusion 360 Motion Study timeline to drive physical mechanism joints.",
        "Connects AI software directly to mechanical engineering CAD workflow."
    ]
    for b in cad_b2:
        p = tf_c2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(10)

    set_speaker_notes(slide8,
        "To demonstrate readiness for physical manufacturing, we integrated with Autodesk Fusion 360. "
        "We exported the 3D CAD assembly and a time-series motion table. This allows mechanical engineers "
        "to simulate joint stresses, clearances, and motor torques directly within standard CAD software.")

    # =========================================================================
    # SLIDE 9: Conclusion & Research Paper Roadmap
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9, prs)
    add_slide_header(slide9, "Conclusion, Limitations & Future Research Paper", "8. SUMMARY & FUTURE WORK")

    c_end = slide9.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.9))
    c_end.fill.solid()
    c_end.fill.fore_color.rgb = COLOR_CARD
    c_end.line.color.rgb = COLOR_ACCENT_GREEN
    tf_end = c_end.text_frame
    tf_end.word_wrap = True
    tf_end.margin_left = tf_end.margin_top = tf_end.margin_right = tf_end.margin_bottom = Inches(0.3)

    p = tf_end.paragraphs[0]
    p.text = "Project Summary & Contributions:"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT_GREEN

    concl_points = [
        "1. Novel Architecture: Proposed TAG-Net with Affordance-Gated Attention Module (AGAM).",
        "2. Proven Performance: Boosted Task Success Rate from 68% to 100% while sustaining 537 FPS on RTX 5070 Ti.",
        "3. Full Robotic Loop: Successfully linked AI perception with 6-DoF Inverse Kinematics in MuJoCo & Fusion 360.",
        "4. Transparent Limitations: Assumes rigid bodies; transparent/specular objects require infrared depth restoration.",
        "5. Future Paper Submission: Integrating open-vocabulary Vision-Language Models (SAM / YOLO-World) for zero-shot text-prompted tool grasping targeting IEEE / Scopus robotics conferences."
    ]
    for cp in concl_points:
        p = tf_end.add_paragraph()
        p.text = cp
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.space_before = Pt(8)

    p_git = tf_end.add_paragraph()
    p_git.text = "\nFull Source Code, Checkpoints & Demos: https://github.com/haree-bt/TAG-Net-Model"
    p_git.font.size = Pt(13)
    p_git.font.bold = True
    p_git.font.color.rgb = COLOR_ACCENT_CYAN

    set_speaker_notes(slide9,
        "In conclusion, TAG-Net proves that robotic grasping must be task-aware to be physically useful. "
        "We achieved 100% task safety with zero latency compromise. As next steps for our upcoming IEEE conference paper, "
        "we are integrating Segment Anything for open-vocabulary grasping. Thank you, and I look forward to your questions!")

    prs.save(PPTX_PATH)
    print(f"\n[SUCCESS] Generated PowerPoint Presentation: {PPTX_PATH}")


if __name__ == "__main__":
    build_presentation()
