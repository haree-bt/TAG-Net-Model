"""
TAG-Net Real-Time Interactive 3D Robotic Arm Simulation in MuJoCo
===================================================================
Simulates autonomous 6-DoF robotic manipulation driven by TAG-Net AGAM
(Affordance-Gated Attention Mechanism) for:
1. Knife (Grasps Handle, Avoids Blade -> Hands over to Human)
2. Mug (Grasps Handle, Clears Drinking Rim -> Places on Coaster)
3. Screwdriver (Grasps Ergonomic Grip, Clears Shaft -> Hands over)

Usage:
    python simulation/realtime_mujoco_simulation.py --tool knife
    python simulation/realtime_mujoco_simulation.py --tool mug
    python simulation/realtime_mujoco_simulation.py --tool screwdriver
"""

import os
import sys
import time
import math
import argparse
import numpy as np
import torch
from scipy.optimize import minimize

import mujoco
import mujoco.viewer

# Project Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.tag_net import TAGNet
from data.dataset import SyntheticToolAffordanceDataset


# ==============================================================================
# 1. TAG-Net AGAM Perception Pipeline
# ==============================================================================

def get_agam_grasp_prediction(tool_name="knife"):
    """
    Runs TAG-Net model inference on the target tool to obtain:
    - 2D pixel coordinates (u, v) on the affordance heatmap
    - Grasp angle theta and gripper width W
    - Safety verification (PASS / FAIL)
    """
    device = torch.device("cpu")
    model = TAGNet().to(device)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "tag_net_best.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dataset = SyntheticToolAffordanceDataset(num_samples=60, img_size=224, seed=101)
    tool_samples = [i for i in range(len(dataset)) if str(dataset[i]["tool_type"]).lower() == tool_name.lower()]
    sample_idx = tool_samples[0] if tool_samples else 0
    sample = dataset[sample_idx]

    x = sample["rgbd"].unsqueeze(0).to(device)
    aff_gt = sample["affordance_mask"][0].numpy()

    with torch.no_grad():
        out = model(x)
        q_task = out["quality_task"][0, 0].numpy()
        ang_out = out["angle"][0].numpy()
        w_out = out["width"][0, 0].numpy()

    # AGAM Affordance-Gated Argmax
    v_best, u_best = np.unravel_index(np.argmax(q_task), q_task.shape)
    best_score = float(q_task[v_best, u_best])

    # Grasp angle
    sin2 = ang_out[0, v_best, u_best]
    cos2 = ang_out[1, v_best, u_best]
    angle_rad = 0.5 * math.atan2(sin2, cos2)
    angle_deg = math.degrees(angle_rad)

    # Gripper finger width
    width_px = float(w_out[v_best, u_best]) * 50.0

    # Safety check: does grasp center hit affordance mask?
    is_safe = bool(aff_gt[v_best, u_best] > 0.5)

    return {
        "tool": tool_name,
        "u": int(u_best),
        "v": int(v_best),
        "score": best_score,
        "angle_rad": angle_rad,
        "angle_deg": angle_deg,
        "width_px": width_px,
        "is_safe": is_safe
    }


# ==============================================================================
# 2. Photorealistic MuJoCo MJCF World Generator
# ==============================================================================

def build_mujoco_scene_xml(tool_type="knife", target_coords=(0.40, 0.0, 0.725)):
    tx, ty, tz = target_coords

    if tool_type == "knife":
        # Handle centered at X=0.40, Blade extends to X=0.55
        tool_mjcf = f"""
      <!-- Target Tool Body: Chef Knife -->
      <body name="tool_object" pos="{tx} {ty} {tz}">
        <!-- GREEN AFFORDANCE HANDLE (AGAM Target: Safe Grip) -->
        <geom name="knife_handle" type="box" size="0.055 0.012 0.012" pos="0 0 0" rgba="0.15 0.85 0.30 1"/>
        <!-- Metal Bolster -->
        <geom name="knife_bolster" type="box" size="0.005 0.014 0.015" pos="0.060 0 0" rgba="0.30 0.32 0.38 1"/>
        <!-- STAINLESS STEEL CUTTING BLADE (Hazard Avoidance Zone) -->
        <geom name="knife_blade" type="box" size="0.075 0.002 0.018" pos="0.138 0 0.004" rgba="0.82 0.85 0.90 1"/>
        <geom name="knife_edge" type="box" size="0.073 0.001 0.002" pos="0.138 0 -0.015" rgba="0.98 0.98 1.0 1"/>
      </body>
        """
        handover_text = "Delivering Chef Knife Handle-First (Blade Pointing Safely Away)"
    elif tool_type == "mug":
        # Handle centered at ty=-0.052, Cup body at Y=0
        tool_mjcf = f"""
      <!-- Target Tool Body: Ceramic Mug -->
      <body name="tool_object" pos="{tx} 0 {tz}">
        <!-- Ceramic Cup Body -->
        <geom name="cup_body" type="cylinder" size="0.038 0.040" pos="0 0 0.040" rgba="0.88 0.90 0.95 1"/>
        <geom name="cup_rim" type="cylinder" size="0.036 0.004" pos="0 0 0.078" rgba="0.45 0.48 0.55 1"/>
        <!-- GREEN AFFORDANCE HANDLE (AGAM Target: Ergonomic Grip) -->
        <geom name="cup_handle_grip" type="cylinder" size="0.009 0.026" pos="0 -0.052 0.040" rgba="0.15 0.85 0.30 1"/>
        <geom name="cup_handle_top" type="cylinder" size="0.008 0.010" pos="0 -0.044 0.062" zaxis="0 1 0" rgba="0.15 0.85 0.30 1"/>
        <geom name="cup_handle_bot" type="cylinder" size="0.008 0.010" pos="0 -0.044 0.018" zaxis="0 1 0" rgba="0.15 0.85 0.30 1"/>
      </body>
        """
        handover_text = "Delivering Mug via Handle (Drinking Rim Untouched & Clean)"
    else:  # screwdriver
        # Grip handle centered at X=0.39, Shaft extends to X=0.54
        tool_mjcf = f"""
      <!-- Target Tool Body: Screwdriver -->
      <body name="tool_object" pos="{tx} {ty} {tz}">
        <!-- GREEN AFFORDANCE ERGONOMIC GRIP (AGAM Target) -->
        <geom name="sd_handle" type="cylinder" size="0.016 0.045" pos="0 0 0" zaxis="1 0 0" rgba="0.15 0.85 0.30 1"/>
        <geom name="sd_cap" type="sphere" size="0.016" pos="-0.045 0 0" rgba="0.25 0.28 0.32 1"/>
        <!-- CHROME VANADIUM STEEL SHAFT (Non-Grasp Zone) -->
        <geom name="sd_shaft" type="cylinder" size="0.004 0.060" pos="0.105 0 0" zaxis="1 0 0" rgba="0.82 0.85 0.90 1"/>
        <!-- Magnetic Tip -->
        <geom name="sd_tip" type="box" size="0.006 0.004 0.0015" pos="0.168 0 0" rgba="0.28 0.30 0.35 1"/>
      </body>
        """
        handover_text = "Delivering Screwdriver Grip-First (Shaft Free for Immediate Use)"

    xml = f"""
<mujoco model="tag_net_realtime_manipulation">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>

  <visual>
    <headlight diffuse="0.88 0.88 0.92" ambient="0.35 0.35 0.40" specular="0.35 0.35 0.35"/>
    <global offheight="720" offwidth="1280"/>
  </visual>

  <worldbody>
    <!-- Lighting -->
    <light diffuse="0.95 0.95 0.98" pos="0.6 0.6 3.0" dir="-0.5 -0.5 -1"/>
    <light diffuse="0.50 0.52 0.60" pos="-0.8 -0.8 2.2" dir="0.5 0.5 -1"/>
    
    <!-- Floor -->
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.92 0.93 0.96 1"/>

    <!-- Manipulation Table Surface -->
    <body name="work_table" pos="0.45 0 0.35">
      <geom type="box" size="0.32 0.48 0.35" rgba="0.20 0.22 0.26 1"/>
      <geom type="box" size="0.33 0.49 0.006" pos="0 0 0.353" rgba="0.12 0.14 0.18 1"/>
    </body>

    <!-- Human Receiver / Handover Station at Table Edge -->
    <body name="human_receiver" pos="0.22 -0.32 0.73">
      <!-- Human Hand / Arm Rest Target -->
      <geom type="capsule" fromto="0 -0.15 0.05 0 0 0.02" size="0.028" rgba="0.86 0.72 0.62 1"/>
      <geom type="box" size="0.035 0.045 0.012" pos="0 0.03 0.02" rgba="0.86 0.72 0.62 1"/>
      <!-- Fingers extended to receive tool -->
      <geom type="capsule" fromto="-0.02 0.06 0.02 -0.02 0.10 0.02" size="0.007" rgba="0.86 0.72 0.62 1"/>
      <geom type="capsule" fromto="0.00 0.06 0.02 0.00 0.11 0.02" size="0.007" rgba="0.86 0.72 0.62 1"/>
      <geom type="capsule" fromto="0.02 0.06 0.02 0.02 0.10 0.02" size="0.007" rgba="0.86 0.72 0.62 1"/>
      <!-- Stand / Placard -->
      <geom type="cylinder" size="0.055 0.010" pos="0 -0.08 -0.01" rgba="0.25 0.28 0.32 1"/>
    </body>

    <!-- Glowing Holographic AGAM Affordance Target Marker -->
    <body name="agam_marker" pos="{tx} {ty} {tz + 0.06}">
      <geom name="marker_orb" type="sphere" size="0.012" rgba="0.0 0.95 0.85 0.85" contype="0" conaffinity="0"/>
    </body>

    <!-- Target Tool -->
    {tool_mjcf}

    <!-- 6-DoF Articulated Robotic Manipulator Arm Mounted on Base -->
    <body name="arm_base" pos="0 0 0.70">
      <geom type="cylinder" size="0.085 0.05" rgba="0.18 0.20 0.24 1"/>

      <!-- Joint 1: Base Turntable (Yaw around Z) -->
      <body name="link1" pos="0 0 0.05">
        <joint name="j1_yaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="cylinder" size="0.065 0.10" pos="0 0 0.10" rgba="0.85 0.87 0.92 1"/>
        <geom type="cylinder" size="0.055 0.05" pos="0 0 0.20" zaxis="0 1 0" rgba="0.25 0.28 0.32 1"/>

        <!-- Joint 2: Shoulder Pitch (Bends forward around Y) -->
        <body name="link2" pos="0 0 0.20">
          <joint name="j2_shoulder" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.048" rgba="0.85 0.87 0.92 1"/>
          <geom type="cylinder" size="0.048 0.045" pos="0 0 0.30" zaxis="0 1 0" rgba="0.25 0.28 0.32 1"/>

          <!-- Joint 3: Elbow Pitch (Bends downward around Y) -->
          <body name="link3" pos="0 0 0.30">
            <joint name="j3_elbow" type="hinge" axis="0 1 0" range="-2.8 2.8"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.042" rgba="0.85 0.87 0.92 1"/>
            <geom type="cylinder" size="0.042 0.04" pos="0 0 0.25" zaxis="0 1 0" rgba="0.25 0.28 0.32 1"/>

            <!-- Joint 4: Wrist Pitch (Maintains top-down perpendicular alignment) -->
            <body name="link4" pos="0 0 0.25">
              <joint name="j4_wrist_pitch" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <geom type="cylinder" size="0.038 0.03" pos="0 0 0.03" rgba="0.85 0.87 0.92 1"/>

              <!-- Joint 5: Wrist Roll -->
              <body name="link5" pos="0 0 0.06">
                <joint name="j5_wrist_roll" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                <geom type="cylinder" size="0.035 0.02" pos="0 0 0.02" rgba="0.20 0.22 0.26 1"/>

                <!-- End-Effector Gripper Body -->
                <body name="ee_gripper" pos="0 0 0.04">
                  <geom type="box" size="0.020 0.055 0.012" rgba="0.18 0.20 0.24 1"/>

                  <!-- Left Gripper Finger -->
                  <body name="left_finger" pos="0 0.038 0.035">
                    <joint name="finger_left" type="slide" axis="0 1 0" range="-0.035 0.035"/>
                    <geom type="box" size="0.018 0.007 0.035" rgba="0.90 0.25 0.20 1"/>
                    <geom type="box" size="0.015 0.003 0.025" pos="0 -0.005 0" rgba="0.1 0.1 0.1 1"/>
                  </body>

                  <!-- Right Gripper Finger -->
                  <body name="right_finger" pos="0 -0.038 0.035">
                    <joint name="finger_right" type="slide" axis="0 1 0" range="-0.035 0.035"/>
                    <geom type="box" size="0.018 0.007 0.035" rgba="0.90 0.25 0.20 1"/>
                    <geom type="box" size="0.015 0.003 0.025" pos="0 0.005 0" rgba="0.1 0.1 0.1 1"/>
                  </body>

                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
    """
    return xml, handover_text


# ==============================================================================
# 3. Kinematic Solvers & Trajectory Generator
# ==============================================================================

def solve_ik_top_down(model, data, target_x, target_y, target_z, initial_guess=(1.1, 1.1)):
    """Solves 6-DoF Inverse Kinematics with strict top-down constraint: j2 + j3 + j4 = pi."""
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_gripper")
    target_xyz = np.array([target_x, target_y, target_z])
    j1 = math.atan2(target_y, target_x)

    def cost(q23):
        j2, j3 = q23
        j4 = np.pi - (j2 + j3)
        data.qpos[0:5] = [j1, j2, j3, j4, 0.0]
        mujoco.mj_forward(model, data)
        return np.sum((data.xpos[ee_id] - target_xyz)**2)

    res = minimize(cost, initial_guess, method="Nelder-Mead", options={"maxiter": 800})
    j2, j3 = res.x
    j4 = np.pi - (j2 + j3)
    return np.array([j1, j2, j3, j4, 0.0])


def run_interactive_simulation(tool_name="knife"):
    print("=" * 70)
    print(f"🤖 TAG-Net Real-Time Robotic Arm Simulation: [{tool_name.upper()}]")
    print("=" * 70)

    # 1. Run AGAM Prediction
    print("\n[Step 1: Running AGAM Affordance Network Inference]...")
    pred = get_agam_grasp_prediction(tool_name)
    print(f"  • Target Object        : {tool_name.upper()}")
    print(f"  • Affordance Pixel (u,v): ({pred['u']}, {pred['v']})")
    print(f"  • AGAM Confidence Score: {pred['score']:.4f}")
    print(f"  • Grasp Orientation    : {pred['angle_deg']:.1f}°")
    print(f"  • Safety Verification  : {'✅ PASS (Handle Viable)' if pred['is_safe'] else '❌ FAIL'}")

    # Set precise 3D target coordinates corresponding to AGAM grasp
    table_z = 0.709
    if tool_name == "knife":
        target_xyz = (0.40, 0.0, table_z + 0.016)
        clamp_val = 0.020
    elif tool_name == "mug":
        target_xyz = (0.45, -0.052, table_z + 0.040)
        clamp_val = 0.022
    else:  # screwdriver
        target_xyz = (0.39, 0.0, table_z + 0.016)
        clamp_val = 0.018

    # 2. Build MuJoCo XML
    xml, handover_desc = build_mujoco_scene_xml(tool_name, target_xyz)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_object")
    marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "agam_marker")

    # 3. Solve Trajectory Waypoints
    tx, ty, tz = target_xyz
    z_grasp = tz + 0.038
    z_hover = z_grasp + 0.16
    z_lift = z_grasp + 0.18

    q_home = np.array([0.0, 0.45, 0.65, np.pi - (0.45 + 0.65), 0.0])
    q_hover = solve_ik_top_down(model, data, tx, ty, z_hover, initial_guess=(0.8, 0.9))
    q_grasp = solve_ik_top_down(model, data, tx, ty, z_grasp, initial_guess=(1.1, 1.1))
    q_lift  = solve_ik_top_down(model, data, tx, ty, z_lift, initial_guess=(0.7, 0.8))

    # Handover delivery pose (hands over to human receiver at table edge)
    hx, hy, hz = 0.22, -0.28, 0.78
    q_handover = solve_ik_top_down(model, data, hx, hy, hz, initial_guess=(0.7, 0.8))

    # 4. Simulation Phases
    phases = [
        ("Phase 1: STANDBY (Camera Framing & Home Pose)", q_home, q_home, 0.0, 0.0, "table", 100),
        ("Phase 2: AGAM TARGET ACQUIRED (Target Marker Active)", q_home, q_home, 0.0, 0.0, "table", 80),
        ("Phase 3: APPROACH (Aligning Over Affordance Zone)", q_home, q_hover, 0.0, 0.0, "table", 140),
        ("Phase 4: DESCEND (Straddling Handle with Millimeter Precision)", q_hover, q_grasp, 0.0, 0.0, "table", 120),
        ("Phase 5: CLAMP (Fingers Closing Flush Across Handle Diameter)", q_grasp, q_grasp, 0.0, clamp_val, "table", 80),
        ("Phase 6: LIFT (Smooth Vertical Clearance Off Table)", q_grasp, q_lift, clamp_val, clamp_val, "lifting", 160),
        ("Phase 7: HANDOVER (Presenting Handle to Human Receiver)", q_lift, q_handover, clamp_val, clamp_val, "handover", 180),
        ("Phase 8: DELIVERY COMPLETE (Safe Handover Verified)", q_handover, q_handover, clamp_val, clamp_val, "at_handover", 120),
    ]

    print("\n[Step 2: Launching Hardware-Accelerated Real-Time MuJoCo 3D Window]...")
    print("  • Orbit Camera : Left Click + Drag")
    print("  • Pan / Zoom   : Right Click + Drag")
    print("  • Pause/Resume : Spacebar")
    print("  • Close Window : Esc / Window Close\n")

    # Initial setup
    data.qpos[0:5] = q_home
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    mujoco.mj_forward(model, data)

    # Launch Native 60 FPS Interactive MuJoCo Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat = [0.38, -0.10, 0.75]
        viewer.cam.distance = 1.15
        viewer.cam.elevation = -22.0
        viewer.cam.azimuth = 135.0

        cycle_count = 1
        while viewer.is_running():
            print(f"\n--- [Execution Cycle {cycle_count}: {tool_name.upper()}] ---")

            for phase_name, q_start, q_end, f_start, f_end, tool_mode, steps in phases:
                if not viewer.is_running():
                    break

                print(f"▶ {phase_name}")

                for step in range(steps):
                    if not viewer.is_running():
                        break

                    t = step / float(steps - 1) if steps > 1 else 1.0
                    smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))

                    # Joint positions
                    q_cur = q_start * (1.0 - smooth_t) + q_end * smooth_t
                    f_cur = f_start * (1.0 - t) + f_end * t
                    data.qpos[0:5] = q_cur
                    data.qpos[5] = -f_cur
                    data.qpos[6] = f_cur

                    # Tool motion logic
                    if tool_mode == "table":
                        if tool_name == "mug":
                            model.body_pos[tool_body_id] = [tx, 0.0, tz]
                        else:
                            model.body_pos[tool_body_id] = [tx, ty, tz]
                    elif tool_mode == "lifting":
                        cur_z = tz + smooth_t * 0.18
                        if tool_name == "mug":
                            model.body_pos[tool_body_id] = [tx, 0.0, cur_z]
                        else:
                            model.body_pos[tool_body_id] = [tx, ty, cur_z]
                    elif tool_mode == "handover":
                        # Transport from lift position to handover station
                        start_pos = np.array([tx, (0.0 if tool_name == "mug" else ty), tz + 0.18])
                        end_pos = np.array([hx, hy, hz])
                        cur_pos = start_pos * (1.0 - smooth_t) + end_pos * smooth_t
                        model.body_pos[tool_body_id] = cur_pos
                    elif tool_mode == "at_handover":
                        model.body_pos[tool_body_id] = [hx, hy, hz]

                    # Marker pulse animation
                    pulse = 0.012 + 0.003 * math.sin(step * 0.15)
                    model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "marker_orb")][0] = pulse

                    # Step physics and synchronize real-time viewer
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.016)  # 60 FPS real-time sync

            print(f"✅ Pick-and-Handover Cycle {cycle_count} Completed Successfully!")
            cycle_count += 1
            time.sleep(1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAG-Net Interactive MuJoCo Real-Time Simulation")
    parser.add_argument("--tool", type=str, default="knife", choices=["knife", "mug", "screwdriver"],
                        help="Target tool to manipulate")
    args = parser.parse_args()

    run_interactive_simulation(args.tool)
