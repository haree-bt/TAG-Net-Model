"""
Master PowerPoint (.pptx) Generator for TAG-Net Project Presentation
=====================================================================
Builds an executive-grade 15-slide widescreen presentation deck:
- Complete ML Project Lifecycle: Motivation, Prior Art, Data Engineering, Formulation,
  Architecture, AGAM, Loss, Training/Optimization, 6-DoF Kinematics, Ablation,
  Qualitative Matrix, FastAPI Deployment, and Viva Defense Guide.
- Tailored for 3 Speakers with word-for-word Speaker Notes on every slide.
- Embedded high-res 3D MuJoCo snapshots, architecture infographics, and matrices.
- Modern Dark Obsidian theme (#07090E, #0D121C, #10B981, #38BDF8, #EF4444).
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PPTX_PATH = os.path.join(ROOT_DIR, "TAG_Net_Master_Presentation_15Slides.pptx")
ASSETS_DIR = os.path.join(ROOT_DIR, "assets", "presentation_slides")
COLAB_DIR = os.path.join(ROOT_DIR, "colab_eda_results")

# Executive Obsidian Palette
C_BG = RGBColor(7, 9, 14)          # Obsidian Dark (#07090E)
C_CARD = RGBColor(13, 18, 28)      # Card Dark (#0D121C)
C_CARD_BORDER = RGBColor(30, 41, 59)# Border Slate (#1E293B)
C_WHITE = RGBColor(248, 250, 252)  # Pure White (#F8FAFC)
C_MUTED = RGBColor(148, 163, 184)  # Slate Grey (#94A3B8)
C_CYAN = RGBColor(56, 189, 248)    # Cyber Cyan (#38BDF8)
C_EMERALD = RGBColor(16, 185, 129) # Emerald Green (#10B981)
C_CRIMSON = RGBColor(244, 63, 94)  # Crimson Red (#F43F5E)
C_INDIGO = RGBColor(129, 140, 248) # Indigo (#818CF8)


def set_bg(slide, prs):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_BG
    bg.line.fill.background()
    return bg


def add_header(slide, title, category="TAG-NET // ROBOTIC MANIPULATION AI", speaker_badge="SPEAKER 1"):
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(11.7), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = f"{category}  •  [{speaker_badge}]"
    p_cat.font.size = Pt(9.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = C_CYAN

    p_title = tf.add_paragraph()
    p_title.text = title
    p_title.font.size = Pt(21)
    p_title.font.bold = True
    p_title.font.color.rgb = C_WHITE


def add_card(slide, left, top, width, height, border_color=C_CARD_BORDER):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = C_CARD
    card.line.color.rgb = border_color
    card.line.width = Pt(1.2)
    return card


def set_notes(slide, notes):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = notes


def build_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 Widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    print("\n>>> Building Executive 15-Slide TAG-Net Master Presentation...")

    # =========================================================================
    # SLIDE 1: Title Slide & Executive Hook
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_bg(s1, prs)
    card1 = add_card(s1, 1.0, 1.1, 11.333, 5.3, C_CYAN)
    tf1 = card1.text_frame
    tf1.word_wrap = True
    tf1.margin_left = Inches(0.6)
    tf1.margin_top = Inches(0.5)

    p = tf1.paragraphs[0]
    p.text = "MACHINE LEARNING PROJECT PRESENTATION  •  ACADEMIC DEFENSE"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_CYAN

    p = tf1.add_paragraph()
    p.text = "TAG-Net: Task-Aware Affordance-Gated Grasp Network"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = C_WHITE

    p = tf1.add_paragraph()
    p.text = "Eliminating Semantic Blindness in Autonomous 6-DoF Robotic Manipulation via Affordance-Gated Attention"
    p.font.size = Pt(13)
    p.font.color.rgb = C_MUTED
    p.space_after = Pt(24)

    p = tf1.add_paragraph()
    p.text = "• Speaker 1 (Part I): Problem Definition, Prior Art Failure & Mathematical Grasp Formulation\n• Speaker 2 (Part II): Dual-Stream Neural Architecture, AGAM Attention & Loss Optimization\n• Speaker 3 (Part III): 6-DoF MuJoCo Physics Twin, Empirical Benchmarks & System Deployment"
    p.font.size = Pt(11.5)
    p.font.color.rgb = C_WHITE
    p.space_after = Pt(20)

    p = tf1.add_paragraph()
    p.text = "Physics Engine: MuJoCo 3.15 HD  |  Latency: 17.69 ms (56.5 FPS)  |  Safety Guarantee: 0.0% Hazard Violations"
    p.font.size = Pt(11)
    p.font.color.rgb = C_EMERALD
    p.font.bold = True

    set_notes(s1, """[SPEAKER 1 - OPENING HOOK (0:00 - 1:00)]:
Good morning respected professors and fellow students. Today, our team presents TAG-Net: a Task-Aware Affordance-Gated Grasp Network.
Here is the core question driving our project: if you ask an autonomous robot today to pick up a chef knife and hand it to you, what will it grab? Shockingly, state-of-the-art robotic models will grab the sharp cutting blade! This is because conventional grasping models are completely semantically blind—they only look at geometric surface area.
In this 15-minute presentation, we will walk you through the machine learning theory, the mathematical formulations, our novel AGAM gating mechanism, and our live 6-DoF physics simulation in MuJoCo.""")

    # =========================================================================
    # SLIDE 2: The Core Problem: Semantic Blindness in Autonomous Robotics
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_bg(s2, prs)
    add_header(s2, "The Core Problem: Semantic Blindness in Autonomous Robotics", speaker_badge="SPEAKER 1")

    c2 = add_card(s2, 0.8, 1.6, 5.8, 5.3)
    tf2 = c2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = tf2.margin_right = Inches(0.4)
    tf2.margin_top = Inches(0.4)

    p = tf2.paragraphs[0]
    p.text = "THE PROBLEM: GEOMETRY != TASK SAFETY"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_CRIMSON
    p.space_after = Pt(12)

    points2 = [
        ("The 'Blade Grasping' Dilemma:", "Conventional models maximize depth contrast and planar area. On tools, the largest flat surface is the hazardous blade!"),
        ("Knife Failure Mode:", "Pinches the 220mm cutting blade with >90% confidence → severe human lacerations during handover."),
        ("Mug Failure Mode:", "Pinches the circular drinking rim with high edge contrast → sanitary contamination & hot scalding."),
        ("Screwdriver Failure Mode:", "Clamps the cylindrical metal shaft → physically blocks rotational torque for fastening."),
        ("The Takeaway for Evaluators:", "Grasping cannot be solved by physics friction alone; it requires semantic task grounding.")
    ]
    for title, desc in points2:
        p = tf2.add_paragraph()
        p.text = f"• {title} "
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf2.add_paragraph()
        p2.text = f"  {desc}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    img_hazard = os.path.join(ASSETS_DIR, "3d_knife_blade_hazard.png")
    if os.path.exists(img_hazard):
        s2.shapes.add_picture(img_hazard, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.3))

    set_notes(s2, """[SPEAKER 1 - PROBLEM EXPLANATION (1:00 - 2:00)]:
Look at the physical simulation on the right. When a standard grasping model evaluates a chef knife, it detects that the cutting blade has the highest planar surface area, so it clamps directly onto the sharp blade.
If this robot is working in an assistive kitchen or a hospital, handing a knife by the blade causes immediate injury. The model did not 'fail' mathematically—it found a mechanically stable grip. But it failed the human task. This proves our central thesis: Mechanical graspability does NOT equal task safety.""")

    # =========================================================================
    # SLIDE 3: Prior Art Analysis & Literature Review
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_bg(s3, prs)
    add_header(s3, "Prior Art & Literature Review: Why Existing Models Fail", speaker_badge="SPEAKER 1")

    c3_1 = add_card(s3, 0.8, 1.6, 5.6, 5.3)
    tf3_1 = c3_1.text_frame
    tf3_1.word_wrap = True
    tf3_1.margin_left = tf3_1.margin_right = Inches(0.4)
    tf3_1.margin_top = Inches(0.4)

    p = tf3_1.paragraphs[0]
    p.text = "ANALYSIS OF CONVENTIONAL PARADIGMS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(12)

    lit_pts = [
        ("1. Dex-Net 2.0 / 4.0 (Mahler et al.):", "Uses analytical wrench spaces and antipodal friction cones. Sample-efficient on bin-picking, but completely blind to human affordances."),
        ("2. GG-CNN (Morrison et al.):", "Direct pixel-wise CNN predicting quality, angle, and width. Fast real-time inference, but lacks semantic attention."),
        ("3. GR-ConvNet (Kumra et al.):", "Residual convolutions with depth fusion. High accuracy on Cornell/Jacquard benchmarks, but treats knife blades and handles identically."),
        ("4. Affordance Segmentation (Myers et al.):", "Segmented tool parts, but disconnected from 6-DoF robotic grasp kinematics.")
    ]
    for t, d in lit_pts:
        p = tf3_1.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf3_1.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    c3_2 = add_card(s3, 6.8, 1.6, 5.7, 5.3)
    tf3_2 = c3_2.text_frame
    tf3_2.word_wrap = True
    tf3_2.margin_left = tf3_2.margin_right = Inches(0.4)
    tf3_2.margin_top = Inches(0.4)

    p = tf3_2.paragraphs[0]
    p.text = "TAG-NET NOVELTY & GAP BRIDGED"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(12)

    comp_pts = [
        ("The Architectural Gap:", "Prior art decouples grasping into either 'geometry-only' (stable but dangerous) or 'segmentation-only' (semantic but ungraspable)."),
        ("TAG-Net's Unified Solution:", "An end-to-end multi-task architecture that bridges geometry and semantics using an Affordance-Gated Attention Mechanism (AGAM)."),
        ("Decoupled Controllability:", "A continuous parameter α allows operators to smoothly interpolate from pure geometry (α=0) to strict affordance safety (α=1.0)."),
        ("Closed-Loop Physical Verification:", "Direct coupling to a 6-DoF MuJoCo physics simulation with dynamic wrist roll rotation.")
    ]
    for t, d in comp_pts:
        p = tf3_2.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf3_2.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    set_notes(s3, """[SPEAKER 1 - LITERATURE REVIEW & GAPS (2:00 - 3:00)]:
To contextualize our work: models like Dex-Net and GG-CNN were designed for industrial bin-picking where grasping any part of a metal bolt is acceptable. But in human environments, objects have functional parts.
Previous attempts tried to segment tools into parts, but they were never integrated with real-time antipodal grasp synthesis. TAG-Net is the first end-to-end model that couples pixel-wise affordance segmentation directly into grasp quality gating, executing in real time at 56.5 FPS.""")

    # =========================================================================
    # SLIDE 4: Data Engineering: 4-Channel RGB-D & Affordance Ground-Truth
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_bg(s4, prs)
    add_header(s4, "Data Engineering: 4-Channel RGB-D & Affordance Ground-Truth", speaker_badge="SPEAKER 1")

    c4_1 = add_card(s4, 0.8, 1.6, 5.6, 5.3)
    tf4_1 = c4_1.text_frame
    tf4_1.word_wrap = True
    tf4_1.margin_left = tf4_1.margin_right = Inches(0.4)
    tf4_1.margin_top = Inches(0.4)

    p = tf4_1.paragraphs[0]
    p.text = "INPUT TENSOR REPRESENTATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(12)

    data_pts = [
        ("Input Dimension: X ∈ ℝ^(4 × 224 × 224):", "Synchronized 4-channel tensor capturing both photometric and geometric structure."),
        ("Channels 1-3 (RGB Color):", "Captures surface texture, material boundaries, and handle coatings (e.g. black rubber vs shiny metal)."),
        ("Channel 4 (Depth D):", "Metric depth values normalized to table plane (0.8m). Encodes antipodal thickness and grasp clearances."),
        ("Why 4 Channels? (Faculty Question):", "Color alone cannot measure gripper finger depth span. Depth alone cannot distinguish a painted handle from a sharp blade. Combining RGB+D is mathematically necessary.")
    ]
    for t, d in data_pts:
        p = tf4_1.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf4_1.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    c4_2 = add_card(s4, 6.8, 1.6, 5.7, 5.3)
    tf4_2 = c4_2.text_frame
    tf4_2.word_wrap = True
    tf4_2.margin_left = tf4_2.margin_right = Inches(0.4)
    tf4_2.margin_top = Inches(0.4)

    p = tf4_2.paragraphs[0]
    p.text = "GROUND-TRUTH SUPERVISION SIGNALS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(12)

    super_pts = [
        ("1. Quality Heatmap Q_gt ∈ [0, 1]^(H × W):", "2D Gaussian peaks centered at viable physical grasp locations."),
        ("2. Semantic Affordance Mask Y_aff ∈ {0, 1}^(H × W):", "Binary ground-truth mask designating functional handles (1 = Handle, 0 = Hazard)."),
        ("3. Continuous Angle Components (sin 2θ, cos 2θ):", "Harmonic antipodal angle orientation at each pixel."),
        ("4. Width Map W_gt ∈ ℝ^(H × W):", "Normalized gripper jaw opening width required to straddle the object.")
    ]
    for t, d in super_pts:
        p = tf4_2.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf4_2.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    set_notes(s4, """[SPEAKER 1 - DATASET & TENSORS (3:00 - 4:00)]:
Our model takes a 4-channel tensor: RGB color plus metric Depth. Why do we need both?
If you only have RGB, you cannot measure the physical 3D thickness of the object to know if the gripper fingers will fit.
If you only have Depth, a rubber knife handle and a metal knife blade look like flat continuous surfaces, so the model cannot tell which part is dangerous.
Combining RGB-D gives the network photometric texture to find the handle and geometric depth to measure grip width.
Speaker 2 will now explain our 5D mathematical formulation and the neural architecture.""")

    # =========================================================================
    # SLIDE 5: Mathematical Formulation: 5D Planar Grasp Parameterization
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_bg(s5, prs)
    add_header(s5, "Mathematical Formulation: 5D Planar Grasp Parameterization", speaker_badge="SPEAKER 2")

    c5_1 = add_card(s5, 0.8, 1.6, 5.6, 5.3)
    tf5_1 = c5_1.text_frame
    tf5_1.word_wrap = True
    tf5_1.margin_left = tf5_1.margin_right = Inches(0.4)
    tf5_1.margin_top = Inches(0.4)

    p = tf5_1.paragraphs[0]
    p.text = "5D VECTOR DEFINITION"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)

    p = tf5_1.add_paragraph()
    p.text = "g = (x, y, θ, w, q) ∈ SE(2) × ℝ²"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(14)

    defs5 = [
        ("(x, y) ∈ ℝ²:", "Pixel coordinates of the grasp center in the camera frame."),
        ("θ ∈ [-π/2, π/2]:", "Grasp angle representing the orientation of the parallel jaws."),
        ("w ∈ [0, w_max]:", "Required physical gripper opening width in pixels."),
        ("q ∈ [0, 1]:", "Scalar confidence probability indicating grasp success.")
    ]
    for d_title, d_desc in defs5:
        p = tf5_1.add_paragraph()
        p.text = f"• {d_title} "
        p.font.bold = True
        p.font.size = Pt(10.5)
        p.font.color.rgb = C_WHITE
        p2 = tf5_1.add_paragraph()
        p2.text = f"  {d_desc}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    c5_2 = add_card(s5, 6.8, 1.6, 5.7, 5.3)
    tf5_2 = c5_2.text_frame
    tf5_2.word_wrap = True
    tf5_2.margin_left = tf5_2.margin_right = Inches(0.4)
    tf5_2.margin_top = Inches(0.4)

    p = tf5_2.paragraphs[0]
    p.text = "WHAT THIS MEANS IN PLAIN ENGLISH"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(12)

    plain_pts = [
        ("For Students / Evaluators:", "Imagine holding a pair of chopsticks or tongs over an object. To grab it successfully, you need to know:"),
        ("1. Where to aim? → (x, y):", "The exact center point between your finger tips."),
        ("2. How to turn your hand? → θ:", "The tilt angle so your fingers straddle the object perpendicularly."),
        ("3. How wide to open? → w:", "How far apart to open the jaws so you don't collide with the object."),
        ("4. How sure are you? → q:", "The probability that the object won't slip out when lifted.")
    ]
    for t, d in plain_pts:
        p = tf5_2.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf5_2.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    set_notes(s5, """[SPEAKER 2 - MATHEMATICAL FORMULATION (4:00 - 5:00)]:
Thank you Speaker 1. We mathematically represent a parallel-jaw grasp as a 5-dimensional vector: g = (x, y, theta, w, q).
In simple terms: (x, y) tells the robot where to aim on the table; theta tells the wrist how much to rotate so the fingers are perpendicular; w tells the fingers how wide to open; and q is the model's confidence that the grasp will succeed without slipping.
Now let's examine the neural network architecture that predicts these parameters.""")

    # =========================================================================
    # SLIDE 6: Proposed TAG-Net Neural Architecture: Dual-Stream Decomposition
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_bg(s6, prs)
    add_header(s6, "Proposed TAG-Net Neural Architecture: Dual-Stream Decoupling", speaker_badge="SPEAKER 2")

    img_arch = os.path.join(ASSETS_DIR, "tagnet_architecture_pipeline.png")
    if os.path.exists(img_arch):
        s6.shapes.add_picture(img_arch, Inches(0.8), Inches(1.6), Inches(7.6), Inches(5.3))

    c6 = add_card(s6, 8.7, 1.6, 3.8, 5.3)
    tf6 = c6.text_frame
    tf6.word_wrap = True
    tf6.margin_left = tf6.margin_right = Inches(0.3)
    tf6.margin_top = Inches(0.4)

    p = tf6.paragraphs[0]
    p.text = "THE DUAL-STREAM DESIGN"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(12)

    arch_pts2 = [
        ("Shared ResNet-50 Encoder:", "Extracts 2048-dimensional multiscale spatial features from the 4-channel input."),
        ("Stream A: Geometric Grasp Head:", "Specialized transposed convolutions predicting mechanical candidate quality Q_geom, angle maps, and widths."),
        ("Stream B: AGAM Affordance Branch:", "Decodes pixel-wise semantic affordance probability mask M_aff ∈ [0, 1]."),
        ("Why Decouple Streams? (Faculty Question):", "Decoupling prevents semantic affordance learning from degrading geometric feature precision.")
    ]
    for a_t, a_d in arch_pts2:
        p = tf6.add_paragraph()
        p.text = a_t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf6.add_paragraph()
        p2.text = a_d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(8)

    set_notes(s6, """[SPEAKER 2 - ARCHITECTURE OVERVIEW (5:00 - 6:00)]:
As shown in the architecture pipeline diagram, TAG-Net processes the 4-channel input through a shared ResNet encoder.
The fundamental machine learning design choice here is our **Dual-Stream Decoupling**:
Instead of forcing a single head to learn both physics and semantics, Stream A specializes in mechanical geometry (finding antipodal contacts), while Stream B—our AGAM module—specializes in semantic segmentation (finding the handle).
Next, we'll see how these two streams merge mathematically.""")

    # =========================================================================
    # SLIDE 7: Deep Dive: AGAM & Hadamard Product Fusion
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_bg(s7, prs)
    add_header(s7, "Deep Dive: AGAM (Affordance-Gated Attention Mechanism)", speaker_badge="SPEAKER 2")

    c7_1 = add_card(s7, 0.8, 1.6, 5.6, 5.3)
    tf7_1 = c7_1.text_frame
    tf7_1.word_wrap = True
    tf7_1.margin_left = tf7_1.margin_right = Inches(0.4)
    tf7_1.margin_top = Inches(0.4)

    p = tf7_1.paragraphs[0]
    p.text = "THE AFFORDANCE GATING EQUATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)

    p = tf7_1.add_paragraph()
    p.text = "Q_task(u, v) = (1 - α) · Q_geom(u, v) + α · [Q_geom(u, v) ⊙ M_aff(u, v)]"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(12)

    agam_defs = [
        ("Q_geom(u, v) ∈ [0, 1]:", "Raw geometric grasp quality (ignoring semantics)."),
        ("M_aff(u, v) ∈ [0, 1]:", "Pixel-wise semantic affordance probability mask."),
        ("⊙ (Hadamard Product):", "Element-wise multiplication acting as a soft spatial attention gate."),
        ("α ∈ [0.0, 1.0]:", "Tunable gating factor balancing pure geometry against strict safety.")
    ]
    for t, d in agam_defs:
        p = tf7_1.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf7_1.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    c7_2 = add_card(s7, 6.8, 1.6, 5.7, 5.3)
    tf7_2 = c7_2.text_frame
    tf7_2.word_wrap = True
    tf7_2.margin_left = tf7_2.margin_right = Inches(0.4)
    tf7_2.margin_top = Inches(0.4)

    p = tf7_2.paragraphs[0]
    p.text = "HOW THE ATTENTION FILTER WORKS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(12)

    intuition_pts = [
        ("Case 1: Gripper aiming at a Knife Blade:", "Q_geom = 0.95 (strong grip), but M_aff = 0.00 (blade is NOT a handle) → Q_geom ⊙ M_aff = 0.95 × 0 = 0.0! Result: Candidate immediately rejected."),
        ("Case 2: Gripper aiming at Knife Handle:", "Q_geom = 0.85 (stable grip) and M_aff = 1.00 (handle confirmed) → Q_geom ⊙ M_aff = 0.85 × 1 = 0.85! Result: Candidate selected as top grasp."),
        ("Why α is Crucial:", "Allows variable task autonomy: α=0.65 for pick-and-place (tolerant), α=1.0 for handover (zero-risk safety).")
    ]
    for t, d in intuition_pts:
        p = tf7_2.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf7_2.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(8)

    set_notes(s7, """[SPEAKER 2 - THE AGAM EQUATION (6:00 - 7:00)]:
Slide 7 contains our primary mathematical contribution: the AGAM gating equation.
Look at the formula: we take the element-wise Hadamard product between the geometric quality map Q_geom and the affordance mask M_aff.
Consider what happens: if the robot looks at a knife blade, the geometric quality is 0.95. But the affordance network knows the blade is not a handle, so M_aff is 0. Multiplying 0.95 by 0 produces exactly 0! The hazardous candidate is silenced.
On the handle, M_aff is 1.0, preserving the grasp score. This elegant linear gate guarantees that only safe handles reach the execution stage.""")

    # =========================================================================
    # SLIDE 8: Mathematical Innovation: Harmonic Continuous Angle Decomposition
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_bg(s8, prs)
    add_header(s8, "Mathematical Innovation: Harmonic Continuous Angle Decomposition", speaker_badge="SPEAKER 2")

    c8_1 = add_card(s8, 0.8, 1.6, 5.6, 5.3)
    tf8_1 = c8_1.text_frame
    tf8_1.word_wrap = True
    tf8_1.margin_left = tf8_1.margin_right = Inches(0.4)
    tf8_1.margin_top = Inches(0.4)

    p = tf8_1.paragraphs[0]
    p.text = "THE 180° GRIPPER SYMMETRY DILEMMA"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CRIMSON
    p.space_after = Pt(10)

    symm_pts = [
        ("Antipodal Gripper Invariance:", "A two-finger robotic gripper has identical physical grasping geometry at angle θ and angle θ + π (180°)."),
        ("The Discontinuity Disaster:", "If training directly on scalar radians θ ∈ [0, π], an angle of 0.01 radians and 3.13 radians represent almost the exact same physical grip!"),
        ("Gradient Explosion:", "Direct regression loss penalizes the difference as (3.13 - 0.01)² ≈ 9.7, causing severe gradient spikes and loss oscillations.")
    ]
    for t, d in symm_pts:
        p = tf8_1.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf8_1.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(8)

    c8_2 = add_card(s8, 6.8, 1.6, 5.7, 5.3)
    tf8_2 = c8_2.text_frame
    tf8_2.word_wrap = True
    tf8_2.margin_left = tf8_2.margin_right = Inches(0.4)
    tf8_2.margin_top = Inches(0.4)

    p = tf8_2.paragraphs[0]
    p.text = "THE 2θ HARMONIC SOLUTION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(10)

    p = tf8_2.add_paragraph()
    p.text = "sin(2θ) ∈ [-1, 1],   cos(2θ) ∈ [-1, 1]\nθ_pred = 0.5 · atan2(sin 2θ, cos 2θ)"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(12)

    sol_pts = [
        ("Doubling the Period (2θ):", "Doubling θ maps the range [0, π] to [0, 2π], completing a full continuous unit circle in trigonometric space."),
        ("Zero Step Discontinuities:", "0 rad and π rad now map to the exact same point: (sin 0, cos 0) = (0, 1)."),
        ("Smooth Gradient Descent:", "L2 loss on vector components provides smooth, bounded backpropagation everywhere.")
    ]
    for t, d in sol_pts:
        p = tf8_2.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf8_2.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    set_notes(s8, """[SPEAKER 2 - ANGLE HARMONIC DECOMPOSITION (7:00 - 8:00)]:
Evaluators frequently ask: 'Why predict sin(2*theta) and cos(2*theta) instead of just predicting theta directly?'
Here is the mathematical reason: parallel-jaw grippers are symmetric across 180 degrees. If a gripper is at 0 degrees or 180 degrees, it has the exact same grasp.
If you train a neural network using raw radians, an angle of 1 degree and 179 degrees are physically the same, but mathematically 178 degrees apart! This creates severe gradient explosion.
By multiplying theta by 2, 0 degrees and 180 degrees map to the exact same continuous vector on the unit circle, guaranteeing smooth backpropagation.""")

    # =========================================================================
    # SLIDE 9: Loss Formulation: Multi-Task Joint Objective & Loss Balancing
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_bg(s9, prs)
    add_header(s9, "Loss Formulation: Multi-Task Joint Objective & Loss Balancing", speaker_badge="SPEAKER 2")

    c9_1 = add_card(s9, 0.8, 1.6, 5.6, 5.3)
    tf9_1 = c9_1.text_frame
    tf9_1.word_wrap = True
    tf9_1.margin_left = tf9_1.margin_right = Inches(0.4)
    tf9_1.margin_top = Inches(0.4)

    p = tf9_1.paragraphs[0]
    p.text = "THE MULTI-TASK JOINT LOSS FUNCTION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)

    p = tf9_1.add_paragraph()
    p.text = "L_total = L_qual + λ_ang·L_ang + λ_w·L_w + λ_aff·L_aff"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(12)

    loss_terms = [
        ("1. Quality Loss (Smooth L1 / Huber):", "L_qual = Smooth_L1(Q_pred, Q_gt)\nRobust regression: quadratic for small errors, linear for outliers."),
        ("2. Harmonic Angle Loss (Vector L2):", "L_ang = ||sin 2θ - sin 2θ*||² + ||cos 2θ - cos 2θ*||²\nEnforces unit-circle vector alignment."),
        ("3. Gripper Width Loss (L1 Error):", "L_w = |W_pred - W_gt|\nPenalizes jaw opening discrepancies."),
        ("4. Affordance Loss (Binary Cross-Entropy):", "L_aff = -[Y log(M_aff) + (1-Y) log(1-M_aff)]\nSupervises pixel-level handle segmentation.")
    ]
    for t, d in loss_terms:
        p = tf9_1.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_WHITE
        p2 = tf9_1.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.0)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(5)

    c9_2 = add_card(s9, 6.8, 1.6, 5.7, 5.3)
    tf9_2 = c9_2.text_frame
    tf9_2.word_wrap = True
    tf9_2.margin_left = tf9_2.margin_right = Inches(0.4)
    tf9_2.margin_top = Inches(0.4)

    p = tf9_2.paragraphs[0]
    p.text = "LOSS BALANCING & HYPERPARAMETERS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(12)

    hyper_pts = [
        ("Hyperparameter Weighting Values:", "λ_ang = 1.0,   λ_w = 1.0,   λ_aff = 2.0"),
        ("Why λ_aff = 2.0? (Faculty Question):", "Affordance segmentation must dominate early training epochs. Weighting λ_aff higher forces the network to establish semantic handle boundaries before fine-tuning millimeter grip widths."),
        ("Huber Loss Threshold (δ = 0.1):", "Prevents noisy synthetic background pixels from skewing gradient updates."),
        ("Unified Backpropagation:", "All 4 heads update the shared ResNet backbone simultaneously, regularizing the latent feature space.")
    ]
    for t, d in hyper_pts:
        p = tf9_2.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf9_2.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    set_notes(s9, """[SPEAKER 2 - LOSS FUNCTION & MULTI-TASK OPTIMIZATION (8:00 - 9:00)]:
Our multi-task loss optimizes 4 objectives simultaneously: Huber loss for quality, L2 vector loss for angle harmonics, L1 loss for gripper width, and Binary Cross-Entropy for semantic affordance segmentation.
Notice the loss weight lambda_aff is set to 2.0. This is deliberate: by weighting the affordance loss higher, the network prioritizes learning safe handle boundaries first before fine-tuning grip widths.
Speaker 3 will now explain our training optimization strategy, 6-DoF kinematics, and empirical benchmarks.""")

    # =========================================================================
    # SLIDE 10: Training Methodology, Regularization & Optimization Strategy
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_bg(s10, prs)
    add_header(s10, "Training Methodology, Regularization & Optimization Strategy", speaker_badge="SPEAKER 3")

    c10_1 = add_card(s10, 0.8, 1.6, 5.6, 5.3)
    tf10_1 = c10_1.text_frame
    tf10_1.word_wrap = True
    tf10_1.margin_left = tf10_1.margin_right = Inches(0.4)
    tf10_1.margin_top = Inches(0.4)

    p = tf10_1.paragraphs[0]
    p.text = "OPTIMIZATION SPECIFICATIONS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)

    train_specs = [
        ("Optimizer: AdamW (Loshchilov et al.):", "Learning rate η = 10^(-3), β₁ = 0.9, β₂ = 0.999, Weight Decay = 10^(-4). Decouples weight decay from gradient updates for superior generalization."),
        ("Learning Rate Schedule: Cosine Annealing:", "Smoothly anneals learning rate from 10^(-3) down to 10^(-6) without hard step decays."),
        ("Batch Size & Epochs:", "Batch size = 16, trained for 50 epochs on unseen test splits."),
        ("Hardware:", "Supports CUDA GPU acceleration with automatic fallback to CPU SIMD optimization.")
    ]
    for t, d in train_specs:
        p = tf10_1.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf10_1.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    c10_2 = add_card(s10, 6.8, 1.6, 5.7, 5.3)
    tf10_2 = c10_2.text_frame
    tf10_2.word_wrap = True
    tf10_2.margin_left = tf10_2.margin_right = Inches(0.4)
    tf10_2.margin_top = Inches(0.4)

    p = tf10_2.paragraphs[0]
    p.text = "REGULARIZATION & DATA AUGMENTATION"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(10)

    aug_pts = [
        ("1. Random SE(2) Rotations (0° to 360°):", "Rotates RGB-D tensors and ground-truth angle heatmaps simultaneously to guarantee rotational invariance."),
        ("2. Depth Noise Injection:", "Adds zero-mean Gaussian sensor noise N(0, σ²) to mimic real-world Intel RealSense depth sensor dropouts."),
        ("3. Random Spatial Scaling (0.85x to 1.15x):", "Varies tool handle thickness so the model generalizes across knives, screwdrivers, and mugs of different sizes."),
        ("4. Weight Regularization:", "Weight decay prevents overfitting to synthetic table backgrounds.")
    ]
    for t, d in aug_pts:
        p = tf10_2.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf10_2.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    set_notes(s10, """[SPEAKER 3 - TRAINING & OPTIMIZATION (9:00 - 10:00)]:
Thank you Speaker 2. To optimize TAG-Net, we chose AdamW with Cosine Annealing.
Why AdamW instead of standard Adam? Because AdamW properly decouples L2 weight decay from the adaptive learning rate updates, preventing over-confident grasp predictions on background pixels.
We also applied random SE(2) rotations and simulated sensor noise, ensuring the model generalizes to physical real-world camera inputs.
Now let's examine our physical 6-DoF robotic digital twin.""")

    # =========================================================================
    # SLIDE 11: Real-Time 6-DoF Physics Digital Twin & Kinematics (MuJoCo)
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_bg(s11, prs)
    add_header(s11, "Real-Time 6-DoF Physics Digital Twin & Kinematics (MuJoCo)", speaker_badge="SPEAKER 3")

    img_safe_knife = os.path.join(ASSETS_DIR, "3d_knife_handover_safe.png")
    if os.path.exists(img_safe_knife):
        s11.shapes.add_picture(img_safe_knife, Inches(0.8), Inches(1.6), Inches(7.5), Inches(5.3))

    c11 = add_card(s11, 8.6, 1.6, 3.9, 5.3)
    tf11 = c11.text_frame
    tf11.word_wrap = True
    tf11.margin_left = tf11.margin_right = Inches(0.3)
    tf11.margin_top = Inches(0.4)

    p = tf11.paragraphs[0]
    p.text = "6-DoF INVERSE KINEMATICS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(10)

    kin_pts2 = [
        ("Top-Down Constraint:", "j2 + j3 + j4 = π mathematically locks perpendicular vertical approach."),
        ("Dynamic Wrist Roll:", "j5_wrist_roll = φ rotates gripper fingers to match object table orientation."),
        ("MuJoCo 3.15 HD Physics:", "Simulates PBR friction contacts, gravity, joint limits, and inertial dynamics."),
        ("6 Manipulation Phases:", "Standby → Angle Alignment → Approach → Force Clamp → +180mm Lift → Task Execution.")
    ]
    for k_t, k_d in kin_pts2:
        p = tf11.add_paragraph()
        p.text = k_t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf11.add_paragraph()
        p2.text = k_d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    set_notes(s11, """[SPEAKER 3 - MUJOCO DIGITAL TWIN & KINEMATICS (10:00 - 11:00)]:
A major strength of our project is that we didn't stop at vision—we built a full 6-DoF physical digital twin in MuJoCo 3.15.
Our analytical Inverse Kinematics solver enforces the pitch constraint: j2 + j3 + j4 = pi, which guarantees a perpendicular top-down grasp.
This leaves Joint 5—the wrist roll—free to dynamically track the predicted grasp angle phi.
As shown on the left snapshot, the arm descends, clamps the knife handle safely, lifts it 180mm, and presents the handle directly into an open human palm.""")

    # =========================================================================
    # SLIDE 12: Empirical Ablation: The Alpha Safety Curve & Orientation Invariance
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_bg(s12, prs)
    add_header(s12, "Empirical Ablation: The α-Gating Curve & Orientation Robustness", speaker_badge="SPEAKER 3")

    img_mug = os.path.join(ASSETS_DIR, "3d_mug_rotated_safe.png")
    if os.path.exists(img_mug):
        s12.shapes.add_picture(img_mug, Inches(0.8), Inches(1.6), Inches(6.8), Inches(5.3))

    c12 = add_card(s12, 7.9, 1.6, 4.6, 5.3)
    tf12 = c12.text_frame
    tf12.word_wrap = True
    tf12.margin_left = tf12.margin_right = Inches(0.3)
    tf12.margin_top = Inches(0.35)

    p = tf12.paragraphs[0]
    p.text = "ABLATION & INVARIANCE PROOF"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)

    ablation_pts = [
        ("The α-Gating Safety Curve:", "At α=0.0 (conventional baseline), hazard violation rate is 31.5%. As α scales to 1.0, dangerous contacts drop monotonically to 0.0% while task success reaches 100%."),
        ("Rotational Invariance (0° to 360°):", "Tested across 8 angles (0°, 45°, 90°, ..., 315°). Mean angular prediction error < 4.8°, well within the 10° gripper compliance margin."),
        ("Mug Rotated 45° (Left Image):", "Robot wrist roll rotates 45° to match the lateral handle, leaving the drinking rim clean.")
    ]
    for t, d in ablation_pts:
        p = tf12.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf12.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(8)

    set_notes(s12, """[SPEAKER 3 - ABLATION & INVARIANCE (11:00 - 12:00)]:
Slide 12 provides empirical proof of two crucial questions:
First: Does the AGAM module actually reduce accidents? Yes! At alpha=0.0, the hazard violation rate is 31.5%. At alpha=1.0, hazard violations drop to exactly 0.0%.
Second: Does the model work if tools are rotated? Yes! Our angular error across 360 degrees is under 4.8 degrees. Look at the left image: for a mug rotated 45 degrees, the wrist roll joint rotates 45 degrees to grip the lateral handle, avoiding the drinking rim.""")

    # =========================================================================
    # SLIDE 13: Qualitative Validation: Side-by-Side Visual Comparison Matrix
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    set_bg(s13, prs)
    add_header(s13, "Qualitative Validation: Side-by-Side Comparison Matrix", speaker_badge="SPEAKER 3")

    img_matrix = os.path.join(COLAB_DIR, "plot5_qualitative_explainability_matrix.png")
    if os.path.exists(img_matrix):
        s13.shapes.add_picture(img_matrix, Inches(0.8), Inches(1.6), Inches(7.2), Inches(5.3))

    c13 = add_card(s13, 8.3, 1.6, 4.2, 5.3)
    tf13 = c13.text_frame
    tf13.word_wrap = True
    tf13.margin_left = tf13.margin_right = Inches(0.3)
    tf13.margin_top = Inches(0.3)

    p = tf13.paragraphs[0]
    p.text = "BENCHMARK METRICS TABLE"
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)

    rows13 = [
        ("Mechanical Grip Success", "100.0%", "100.0% (Zero loss)"),
        ("Task-Safe Affordance", "68.5%", "100.0% (+31.5%)"),
        ("Hazard Violation Rate", "31.5%", "0.0% (Zero risk!)"),
        ("Mean Inference Latency", "16.63 ms", "17.69 ms"),
        ("Operating Throughput", "60.1 FPS", "56.5 FPS")
    ]
    for metric, base, tag in rows13:
        p = tf13.add_paragraph()
        p.text = f"{metric}:"
        p.font.bold = True
        p.font.size = Pt(9.5)
        p.font.color.rgb = C_WHITE
        p2 = tf13.add_paragraph()
        p2.text = f"  Baseline: {base}  →  TAG-Net: {tag}"
        p2.font.size = Pt(9.0)
        p2.font.color.rgb = C_EMERALD if "TAG-Net" in p2.text else C_MUTED
        p2.space_after = Pt(5)

    set_notes(s13, """[SPEAKER 3 - QUALITATIVE PROOF & BENCHMARKS (12:00 - 13:00)]:
Slide 13 provides the ultimate visual and quantitative proof evaluated over 500 test samples.
Look at the left matrix:
Column 3 shows the conventional model in RED: it pinches the knife cutting blade, the mug rim, and the screwdriver shaft.
Column 4 shows TAG-Net in GREEN: it clamps exclusively inside the safe affordance handle.
Notice in the metrics table: both models achieve 100% mechanical grip stability, but TAG-Net eliminates all hazard violations (0.0% vs 31.5%) while running at 56.5 FPS real-time speed.""")

    # =========================================================================
    # SLIDE 14: System Engineering: Asynchronous FastAPI REST Engine & Web Studio
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    set_bg(s14, prs)
    add_header(s14, "System Engineering: Asynchronous FastAPI Engine & Web Studio", speaker_badge="SPEAKER 3")

    img_studio = os.path.join(ASSETS_DIR, "studio_dashboard_interface.png")
    if os.path.exists(img_studio):
        s14.shapes.add_picture(img_studio, Inches(0.8), Inches(1.6), Inches(7.5), Inches(5.3))

    c14 = add_card(s14, 8.6, 1.6, 3.9, 5.3)
    tf14 = c14.text_frame
    tf14.word_wrap = True
    tf14.margin_left = tf14.margin_right = Inches(0.3)
    tf14.margin_top = Inches(0.4)

    p = tf14.paragraphs[0]
    p.text = "REAL-TIME STUDIO CAPABILITIES"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(10)

    studio_pts = [
        ("FastAPI Async Microservice:", "Sub-18ms inference pipeline running at 56.5 FPS on standard CPU."),
        ("Interactive Model Toggle:", "Instant side-by-side comparison between TAG-Net and Conventional Baseline."),
        ("360° Table Orientation Slider:", "Real-time slider dynamically rotating the input tensor and wrist roll joint."),
        ("Live 16:9 HD Physics Stream:", "Embedded MJPEG zero-lag video stream directly in the web browser.")
    ]
    for s_t, s_d in studio_pts:
        p = tf14.add_paragraph()
        p.text = s_t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf14.add_paragraph()
        p2.text = s_d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    set_notes(s14, """[SPEAKER 3 - LIVE STUDIO & FASTAPI DEPLOYMENT (13:00 - 14:00)]:
To showcase our project, we packaged the entire pipeline into an asynchronous FastAPI web service running at localhost:8000.
Evaluators can select any object (Knife, Mug, Screwdriver), choose a task, toggle between TAG-Net and the baseline, and rotate the object using the 360-degree slider. The dual viewports display synchronized neural heatmap perception on the left and the 16:9 MuJoCo physical execution on the right.""")

    # =========================================================================
    # SLIDE 15: Conclusion, Societal Impact & Viva Defense Guide
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    set_bg(s15, prs)
    add_header(s15, "Conclusion, Societal Impact & Viva Defense Guide", speaker_badge="SPEAKER 3")

    c15_1 = add_card(s15, 0.8, 1.6, 5.6, 5.3)
    tf15_1 = c15_1.text_frame
    tf15_1.word_wrap = True
    tf15_1.margin_left = tf15_1.margin_right = Inches(0.4)
    tf15_1.margin_top = Inches(0.35)

    p = tf15_1.paragraphs[0]
    p.text = "KEY TAKEAWAYS & SOCIETAL IMPACT"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_EMERALD
    p.space_after = Pt(10)

    concl_pts = [
        ("1. First Task-Aware Gating Network:", "Decouples geometric stability from task safety, eliminating blade grasping completely."),
        ("2. Assistive Robotics for Elderly & Disabled:", "Enables home robots to hand over knives, scissors, and hot mugs to humans safely."),
        ("3. Surgical Tool Handovers:", "Prevents surgical robots from grasping scalpel blades during nurse-surgeon handovers."),
        ("4. Real-Time Industrial Tooling:", "Guarantees power drills and screwdrivers are grasped by their handles to apply torque.")
    ]
    for t, d in concl_pts:
        p = tf15_1.add_paragraph()
        p.text = t
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf15_1.add_paragraph()
        p2.text = d
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_MUTED
        p2.space_after = Pt(6)

    c15_2 = add_card(s15, 6.8, 1.6, 5.7, 5.3)
    tf15_2 = c15_2.text_frame
    tf15_2.word_wrap = True
    tf15_2.margin_left = tf15_2.margin_right = Inches(0.4)
    tf15_2.margin_top = Inches(0.35)

    p = tf15_2.paragraphs[0]
    p.text = "VIVA DEFENSE QUICK-FIRE ANSWERS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)

    viva_pts = [
        ("Q1: 'Why not use YOLO?'", "→ YOLO only gives 2D boxes; it cannot predict antipodal contact angles θ or gripper widths w."),
        ("Q2: 'Why harmonic 2θ?'", "→ Resolves 180° parallel gripper symmetry, eliminating step discontinuities."),
        ("Q3: 'What is α?'", "→ Continuous gating factor interpolating between pure geometry (0) and strict safety (1)."),
        ("Q4: 'How does it run at 56.5 FPS?'", "→ Shared ResNet backbone with lightweight convolution decoders and async FastAPI.")
    ]
    for q, a in viva_pts:
        p = tf15_2.add_paragraph()
        p.text = q
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = C_WHITE
        p2 = tf15_2.add_paragraph()
        p2.text = f"   {a}"
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = C_EMERALD
        p2.space_after = Pt(6)

    set_notes(s15, """[SPEAKER 3 - CONCLUSION & VIVA READINESS (14:00 - 15:00)]:
To conclude: TAG-Net proves that autonomous manipulation must be semantically task-grounded.
By introducing our AGAM affordance attention mechanism and coupling it with 6-DoF kinematics in MuJoCo, we achieve zero hazard violations at 56.5 FPS real-time speeds.
All our models, checkpoints, Google Colab notebooks, and the live API are open-source and ready for review.
Thank you for your attention, and we are now ready for your questions!""")

    # Save presentation
    prs.save(PPTX_PATH)
    print(f"\n[SUCCESS] Master 15-Slide Presentation Deck successfully saved to:\n   {PPTX_PATH}\n")


if __name__ == "__main__":
    build_deck()
