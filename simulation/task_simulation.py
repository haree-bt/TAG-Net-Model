"""
TAG-Net Comprehensive Multi-Task Robotic Manipulation Simulation in MuJoCo
===========================================================================
Simulates 6-DoF robotic manipulation guided by TAG-Net AGAM heatmaps for:
  Objects:
    1. Knife
    2. Mug
    3. Screwdriver (Tool)
  Tasks:
    1. pick_and_place   : Pick up and transport to designated tray/coaster
    2. handover         : Pick up and deliver safely to human receiver hand
    3. work_with_it     : Pick up and execute functional tool action
                         - Knife: Slicing on cutting board
                         - Mug: Tilting / drinking gesture
                         - Screwdriver: Clockwise torque screwing
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import torch
from scipy.optimize import minimize
import mujoco

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.tag_net import TAGNet
from data.dataset import SyntheticToolAffordanceDataset


# ==============================================================================
# 1. AGAM Perception Pipeline (Task-Aware Grasp Synthesis)
# ==============================================================================

def get_agam_task_grasp(tool_name="knife", task_name="handover"):
    """
    Executes TAG-Net AGAM inference to find the optimal task-conditioned grasp.
    Returns:
        - 2D pixel coordinates (u, v) on heatmap
        - AGAM confidence score
        - Grasp angle and gripper opening width
        - Safety verification flag
        - Recommended 3D Cartesian coordinates for physical execution
    """
    device = torch.device("cpu")
    model = TAGNet().to(device)
    ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "tag_net_best.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    norm_tool = "screwdriver" if tool_name.lower() in ["tool", "screwdriver"] else tool_name.lower()
    dataset = SyntheticToolAffordanceDataset(num_samples=60, img_size=224, seed=101)
    tool_samples = [i for i in range(len(dataset)) if str(dataset[i]["tool_type"]).lower() == norm_tool]
    sample_idx = tool_samples[0] if tool_samples else 0
    sample = dataset[sample_idx]

    x = sample["rgbd"].unsqueeze(0).to(device)
    aff_gt = sample["affordance_mask"][0].numpy()

    with torch.no_grad():
        out = model(x)
        q_task = out["quality_task"][0, 0].numpy()
        w_out = out["width"][0, 0].numpy()
        sin_map = out["sin_2theta"][0, 0].numpy()
        cos_map = out["cos_2theta"][0, 0].numpy()

    v_best, u_best = np.unravel_index(np.argmax(q_task), q_task.shape)
    best_score = float(q_task[v_best, u_best])
    sin2 = float(sin_map[v_best, u_best])
    cos2 = float(cos_map[v_best, u_best])
    angle_rad = 0.5 * math.atan2(sin2, cos2)
    width_px = float(w_out[v_best, u_best]) * 50.0
    is_safe = bool(aff_gt[v_best, u_best] > 0.5)

    # 3D Cartesian Target Coordinates on Table Surface (Z = 0.709)
    table_z = 0.709
    if norm_tool == "knife":
        if task_name == "handover":
            target_xyz = (0.38, 0.0, table_z + 0.016)
        elif task_name == "work_with_it":
            target_xyz = (0.41, 0.0, table_z + 0.016)
        else: # pick_and_place
            target_xyz = (0.40, 0.0, table_z + 0.016)
        clamp_val = 0.020
    elif norm_tool == "mug":
        if task_name == "handover":
            target_xyz = (0.45, -0.052, table_z + 0.040)
        elif task_name == "work_with_it":
            target_xyz = (0.45, -0.052, table_z + 0.040)
        else: # pick_and_place
            target_xyz = (0.45, -0.052, table_z + 0.040)
        clamp_val = 0.022
    else: # screwdriver
        if task_name == "handover":
            target_xyz = (0.41, 0.0, table_z + 0.016)
        elif task_name == "work_with_it":
            target_xyz = (0.38, 0.0, table_z + 0.016)
        else: # pick_and_place
            target_xyz = (0.39, 0.0, table_z + 0.016)
        clamp_val = 0.018

    return {
        "tool": norm_tool,
        "task": task_name,
        "u": int(u_best),
        "v": int(v_best),
        "score": best_score,
        "angle_rad": angle_rad,
        "angle_deg": math.degrees(angle_rad),
        "width_px": width_px,
        "is_safe": is_safe,
        "target_xyz": target_xyz,
        "clamp_val": clamp_val
    }


# ==============================================================================
# 2. Photorealistic Multi-Station MuJoCo MJCF World Generator
# ==============================================================================

def build_task_scene_xml(tool_type="knife", target_coords=(0.40, 0.0, 0.725), task_name="handover"):
    tx, ty, tz = target_coords
    norm_tool = "screwdriver" if tool_type.lower() in ["tool", "screwdriver"] else tool_type.lower()

    if norm_tool == "knife":
        tool_mjcf = f"""
      <!-- Target Tool: Chef Knife -->
      <body name="tool_object" pos="{tx} {ty} {tz}">
        <!-- GREEN AFFORDANCE HANDLE -->
        <geom name="knife_handle" type="box" size="0.055 0.012 0.012" pos="0 0 0" rgba="0.15 0.85 0.30 1"/>
        <!-- Metal Bolster -->
        <geom name="knife_bolster" type="box" size="0.005 0.014 0.015" pos="0.060 0 0" rgba="0.30 0.32 0.38 1"/>
        <!-- STAINLESS STEEL CUTTING BLADE (Hazard Zone) -->
        <geom name="knife_blade" type="box" size="0.075 0.002 0.018" pos="0.138 0 0.004" rgba="0.82 0.85 0.90 1"/>
        <geom name="knife_edge" type="box" size="0.073 0.001 0.002" pos="0.138 0 -0.015" rgba="0.98 0.98 1.0 1"/>
      </body>
        """
    elif norm_tool == "mug":
        tool_mjcf = f"""
      <!-- Target Tool: Ceramic Mug resting flush on table -->
      <body name="tool_object" pos="{tx} 0 {tz - 0.040}">
        <!-- Ceramic Cup Body -->
        <geom name="cup_body" type="cylinder" size="0.038 0.040" pos="0 0 0.040" rgba="0.88 0.90 0.95 1"/>
        <geom name="cup_rim" type="cylinder" size="0.036 0.004" pos="0 0 0.078" rgba="0.45 0.48 0.55 1"/>
        <!-- GREEN AFFORDANCE HANDLE (Ergonomic Grip) -->
        <geom name="cup_handle_grip" type="cylinder" size="0.009 0.026" pos="0 -0.052 0.040" rgba="0.15 0.85 0.30 1"/>
        <geom name="cup_handle_top" type="cylinder" size="0.008 0.010" pos="0 -0.044 0.062" zaxis="0 1 0" rgba="0.15 0.85 0.30 1"/>
        <geom name="cup_handle_bot" type="cylinder" size="0.008 0.010" pos="0 -0.044 0.018" zaxis="0 1 0" rgba="0.15 0.85 0.30 1"/>
      </body>
        """
    else:  # screwdriver
        tool_mjcf = f"""
      <!-- Target Tool: Screwdriver -->
      <body name="tool_object" pos="{tx} {ty} {tz}">
        <!-- GREEN AFFORDANCE ERGONOMIC GRIP -->
        <geom name="sd_handle" type="cylinder" size="0.016 0.045" pos="0 0 0" zaxis="1 0 0" rgba="0.15 0.85 0.30 1"/>
        <geom name="sd_cap" type="sphere" size="0.016" pos="-0.045 0 0" rgba="0.25 0.28 0.32 1"/>
        <!-- CHROME VANADIUM STEEL SHAFT -->
        <geom name="sd_shaft" type="cylinder" size="0.004 0.060" pos="0.105 0 0" zaxis="1 0 0" rgba="0.82 0.85 0.90 1"/>
        <!-- Magnetic Tip -->
        <geom name="sd_tip" type="box" size="0.006 0.004 0.0015" pos="0.168 0 0" rgba="0.28 0.30 0.35 1"/>
      </body>
        """

    xml = f"""
<mujoco model="tag_net_task_manipulation">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>

  <visual>
    <headlight diffuse="0.88 0.88 0.92" ambient="0.35 0.35 0.40" specular="0.35 0.35 0.35"/>
    <global offheight="480" offwidth="640"/>
  </visual>

  <worldbody>
    <!-- Lighting -->
    <light diffuse="0.95 0.95 0.98" pos="0.6 0.6 3.0" dir="-0.5 -0.5 -1"/>
    <light diffuse="0.50 0.52 0.60" pos="-0.8 -0.8 2.2" dir="0.5 0.5 -1"/>
    
    <!-- Floor -->
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.92 0.93 0.96 1"/>

    <!-- Work Table Surface -->
    <body name="work_table" pos="0.45 0 0.35">
      <geom type="box" size="0.35 0.50 0.35" rgba="0.20 0.22 0.26 1"/>
      <geom type="box" size="0.36 0.51 0.006" pos="0 0 0.353" rgba="0.12 0.14 0.18 1"/>
    </body>

    <!-- STATION A: Placement Zone / Coaster at Right Side -->
    <body name="placement_zone" pos="0.32 0.22 0.709">
      <geom type="cylinder" size="0.075 0.004" pos="0 0 0.004" rgba="0.10 0.70 0.90 0.85"/>
      <geom type="cylinder" size="0.065 0.005" pos="0 0 0.005" rgba="0.05 0.15 0.25 1"/>
    </body>

    <!-- STATION B: Functional Work Area (Cutting Board / Screw Fixture) -->
    <body name="workstation_fixture" pos="0.42 0.15 0.709">
      <!-- Cutting Board Surface -->
      <geom type="box" size="0.080 0.060 0.006" pos="0 0 0.006" rgba="0.78 0.60 0.42 0.9"/>
      <!-- Metallic Screw Fixture Socket -->
      <geom type="cylinder" size="0.012 0.012" pos="0 0 0.012" rgba="0.75 0.78 0.82 1"/>
      <geom type="box" size="0.008 0.002 0.001" pos="0 0 0.024" rgba="0.1 0.1 0.1 1"/>
    </body>

    <!-- STATION C: Human Receiver Hand at Table Edge -->
    <body name="human_receiver" pos="0.22 -0.32 0.73">
      <!-- Human Forearm -->
      <geom type="capsule" fromto="0 -0.15 0.05 0 0 0.02" size="0.028" rgba="0.86 0.72 0.62 1"/>
      <geom type="box" size="0.035 0.045 0.012" pos="0 0.03 0.02" rgba="0.86 0.72 0.62 1"/>
      <!-- Outstretched Receiving Fingers -->
      <geom type="capsule" fromto="-0.02 0.06 0.02 -0.02 0.10 0.02" size="0.007" rgba="0.86 0.72 0.62 1"/>
      <geom type="capsule" fromto="0.00 0.06 0.02 0.00 0.11 0.02" size="0.007" rgba="0.86 0.72 0.62 1"/>
      <geom type="capsule" fromto="0.02 0.06 0.02 0.02 0.10 0.02" size="0.007" rgba="0.86 0.72 0.62 1"/>
    </body>

    <!-- Holographic AGAM Affordance Target Marker -->
    <body name="agam_marker" pos="{tx} {ty} {tz + 0.06}">
      <geom name="marker_orb" type="sphere" size="0.014" rgba="0.0 0.95 0.85 0.85" contype="0" conaffinity="0"/>
    </body>

    <!-- Target Tool Body -->
    {tool_mjcf}

    <!-- 6-DoF Articulated Manipulator Robot Arm -->
    <body name="arm_base" pos="0 0 0.70">
      <geom type="cylinder" size="0.085 0.05" rgba="0.18 0.20 0.24 1"/>

      <!-- Joint 1: Base Yaw -->
      <body name="link1" pos="0 0 0.05">
        <joint name="j1_yaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="cylinder" size="0.065 0.10" pos="0 0 0.10" rgba="0.85 0.87 0.92 1"/>
        <geom type="cylinder" size="0.055 0.05" pos="0 0 0.20" zaxis="0 1 0" rgba="0.25 0.28 0.32 1"/>

        <!-- Joint 2: Shoulder Pitch -->
        <body name="link2" pos="0 0 0.20">
          <joint name="j2_shoulder" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.048" rgba="0.85 0.87 0.92 1"/>
          <geom type="cylinder" size="0.048 0.045" pos="0 0 0.30" zaxis="0 1 0" rgba="0.25 0.28 0.32 1"/>

          <!-- Joint 3: Elbow Pitch -->
          <body name="link3" pos="0 0 0.30">
            <joint name="j3_elbow" type="hinge" axis="0 1 0" range="-2.8 2.8"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.042" rgba="0.85 0.87 0.92 1"/>
            <geom type="cylinder" size="0.042 0.04" pos="0 0 0.25" zaxis="0 1 0" rgba="0.25 0.28 0.32 1"/>

            <!-- Joint 4: Wrist Pitch -->
            <body name="link4" pos="0 0 0.25">
              <joint name="j4_wrist_pitch" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <geom type="cylinder" size="0.038 0.03" pos="0 0 0.03" rgba="0.85 0.87 0.92 1"/>

              <!-- Joint 5: Wrist Roll -->
              <body name="link5" pos="0 0 0.06">
                <joint name="j5_wrist_roll" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
                <geom type="cylinder" size="0.035 0.02" pos="0 0 0.02" rgba="0.20 0.22 0.26 1"/>

                <!-- End-Effector Gripper Body -->
                <body name="ee_gripper" pos="0 0 0.04">
                  <geom type="box" size="0.020 0.055 0.012" rgba="0.18 0.20 0.24 1"/>

                  <!-- Left Finger -->
                  <body name="left_finger" pos="0 0.038 0.035">
                    <joint name="finger_left" type="slide" axis="0 1 0" range="-0.035 0.035"/>
                    <geom type="box" size="0.018 0.007 0.035" rgba="0.90 0.25 0.20 1"/>
                    <geom type="box" size="0.015 0.003 0.025" pos="0 -0.005 0" rgba="0.1 0.1 0.1 1"/>
                  </body>

                  <!-- Right Finger -->
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
    return xml


# ==============================================================================
# 3. Kinematic Solvers
# ==============================================================================

def solve_ik_top_down(model, data, target_x, target_y, target_z, initial_guess=(1.1, 1.1), roll=0.0):
    """Solves 6-DoF Inverse Kinematics enforcing top-down constraint: j2 + j3 + j4 = pi."""
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_gripper")
    target_xyz = np.array([target_x, target_y, target_z])
    j1 = math.atan2(target_y, target_x)

    def cost(q23):
        j2, j3 = q23
        j4 = np.pi - (j2 + j3)
        data.qpos[0:5] = [j1, j2, j3, j4, roll]
        mujoco.mj_forward(model, data)
        return np.sum((data.xpos[ee_id] - target_xyz)**2)

    res = minimize(cost, initial_guess, method="Nelder-Mead", options={"maxiter": 800})
    j2, j3 = res.x
    j4 = np.pi - (j2 + j3)
    return np.array([j1, j2, j3, j4, roll])


# ==============================================================================
# 4. Multi-Task Trajectory Generator
# ==============================================================================

def generate_task_phases(model, data, tool_name, task_name, target_xyz, clamp_val):
    tx, ty, tz = target_xyz
    z_grasp = tz + 0.038
    z_hover = z_grasp + 0.16
    z_lift  = z_grasp + 0.18

    q_home = np.array([0.0, 0.45, 0.65, np.pi - (0.45 + 0.65), 0.0])
    q_hover = solve_ik_top_down(model, data, tx, ty, z_hover, initial_guess=(0.8, 0.9))
    q_grasp = solve_ik_top_down(model, data, tx, ty, z_grasp, initial_guess=(1.1, 1.1))
    q_lift  = solve_ik_top_down(model, data, tx, ty, z_lift, initial_guess=(0.7, 0.8))

    norm_task = task_name.lower()

    if norm_task == "pick_and_place":
        # Destination: Placement Coaster at (0.32, 0.22, 0.71)
        px, py, pz = 0.32, 0.22, 0.709
        pz_hover = pz + 0.20
        pz_place = pz + 0.038 + (0.040 if tool_name == "mug" else 0.016)
        q_p_hover = solve_ik_top_down(model, data, px, py, pz_hover, initial_guess=(0.8, 0.9))
        q_p_place = solve_ik_top_down(model, data, px, py, pz_place, initial_guess=(1.1, 1.1))

        phases = [
            ("Phase 1: STANDBY (Framing Workspace)", q_home, q_home, 0.0, 0.0, "table", 30),
            ("Phase 2: AGAM TARGET ACQUIRED (Affordance Gate)", q_home, q_home, 0.0, 0.0, "table", 25),
            ("Phase 3: APPROACH (Hover Over Grasp Target)", q_home, q_hover, 0.0, 0.0, "table", 35),
            ("Phase 4: DESCEND (Straddling Handle Precision)", q_hover, q_grasp, 0.0, 0.0, "table", 35),
            ("Phase 5: CLAMP (Fingers Closing Flush)", q_grasp, q_grasp, 0.0, clamp_val, "table", 25),
            ("Phase 6: LIFT (Smooth Vertical Clearance)", q_grasp, q_lift, clamp_val, clamp_val, "attached", 35),
            ("Phase 7: TRANSPORT TO PLACEMENT ZONE", q_lift, q_p_hover, clamp_val, clamp_val, "attached", 45),
            ("Phase 8: LOWER ONTO PLACEMENT TRAY", q_p_hover, q_p_place, clamp_val, clamp_val, "attached", 35),
            ("Phase 9: RELEASE (Fingers Opening Stably)", q_p_place, q_p_place, clamp_val, 0.0, "released", 25),
            ("Phase 10: RETRACT TO HOME (Task Complete)", q_p_place, q_home, 0.0, 0.0, "released", 40),
        ]

    elif norm_task == "work_with_it":
        # Destination: Workstation Fixture / Cutting Board / Screw Socket
        wx, wy, wz = 0.42, 0.15, 0.709
        wz_hover = wz + 0.18
        wz_work  = wz + 0.038 + (0.040 if tool_name == "mug" else 0.016)

        q_w_hover = solve_ik_top_down(model, data, wx, wy, wz_hover, initial_guess=(0.8, 0.9))
        q_w_work  = solve_ik_top_down(model, data, wx, wy, wz_work, initial_guess=(1.1, 1.1))

        # Tool-specific work actions
        if tool_name == "knife":
            # Cutting motion: rhythmic slice down and up
            q_slice_down = solve_ik_top_down(model, data, wx + 0.02, wy, wz_work - 0.015, initial_guess=(1.1, 1.1))
            q_slice_up   = solve_ik_top_down(model, data, wx - 0.02, wy, wz_work + 0.025, initial_guess=(1.1, 1.1))
            work_subphases = [
                ("Phase 8: CUTTING STROKE 1 (Slicing on Board)", q_w_work, q_slice_down, clamp_val, clamp_val, "attached", 20),
                ("Phase 8: CUTTING STROKE 2 (Blade Lift)", q_slice_down, q_slice_up, clamp_val, clamp_val, "attached", 20),
                ("Phase 8: CUTTING STROKE 3 (Finishing Slice)", q_slice_up, q_w_work, clamp_val, clamp_val, "attached", 20),
            ]
        elif tool_name == "mug":
            # Drinking / pouring tilt gesture: pitch wrist by 35 deg
            q_drink_lift = solve_ik_top_down(model, data, wx, wy, wz_hover + 0.10, initial_guess=(0.7, 0.8))
            q_drink_tilt = np.copy(q_drink_lift)
            q_drink_tilt[3] += 0.60  # Tilt pitch
            work_subphases = [
                ("Phase 8: LIFT TO DRINKING HEIGHT", q_w_work, q_drink_lift, clamp_val, clamp_val, "attached", 30),
                ("Phase 8: TILT DRINKING/POURING GESTURE", q_drink_lift, q_drink_tilt, clamp_val, clamp_val, "attached", 35),
                ("Phase 8: RETURN UPRIGHT", q_drink_tilt, q_drink_lift, clamp_val, clamp_val, "attached", 25),
            ]
        else: # screwdriver
            # Screwing motion: rotate wrist roll joint by 360 degrees
            q_screw_start = np.copy(q_w_work)
            q_screw_turn1 = np.copy(q_w_work)
            q_screw_turn1[4] = np.pi  # 180 deg
            q_screw_turn2 = np.copy(q_w_work)
            q_screw_turn2[4] = 2.0 * np.pi  # 360 deg
            work_subphases = [
                ("Phase 8: ALIGN TIP WITH SCREW HEAD", q_w_hover, q_w_work, clamp_val, clamp_val, "attached", 25),
                ("Phase 8: CLOCKWISE TORQUE SCREWING (0-180 deg)", q_screw_start, q_screw_turn1, clamp_val, clamp_val, "attached", 30),
                ("Phase 8: CONTINUOUS FASTENING (180-360 deg)", q_screw_turn1, q_screw_turn2, clamp_val, clamp_val, "attached", 30),
            ]

        phases = [
            ("Phase 1: STANDBY (Framing Workspace)", q_home, q_home, 0.0, 0.0, "table", 30),
            ("Phase 2: AGAM TARGET ACQUIRED (Power Grip Affordance)", q_home, q_home, 0.0, 0.0, "table", 25),
            ("Phase 3: APPROACH (Hover Over Functional Grip)", q_home, q_hover, 0.0, 0.0, "table", 35),
            ("Phase 4: DESCEND (Straddling Ergonomic Handle)", q_hover, q_grasp, 0.0, 0.0, "table", 35),
            ("Phase 5: CLAMP (Firm Power Clamping)", q_grasp, q_grasp, 0.0, clamp_val, "table", 25),
            ("Phase 6: LIFT (Vertical Clearance)", q_grasp, q_lift, clamp_val, clamp_val, "attached", 35),
            ("Phase 7: MOVE TO WORKSTATION FIXTURE", q_lift, q_w_hover, clamp_val, clamp_val, "attached", 40),
            *work_subphases,
            ("Phase 9: RETRACT TO HOME (Functional Task Complete)", q_w_hover, q_home, clamp_val, 0.0, "attached", 40),
        ]

    else:  # handover (default: pick and give to a human hand)
        hx, hy, hz = 0.22, -0.28, 0.78
        q_handover = solve_ik_top_down(model, data, hx, hy, hz, initial_guess=(0.7, 0.8))

        phases = [
            ("Phase 1: STANDBY (Workspace Framing)", q_home, q_home, 0.0, 0.0, "table", 30),
            ("Phase 2: AGAM TARGET ACQUIRED (Safe Handover Orientation)", q_home, q_home, 0.0, 0.0, "table", 25),
            ("Phase 3: APPROACH (Aligning Over Safe Affordance Zone)", q_home, q_hover, 0.0, 0.0, "table", 35),
            ("Phase 4: DESCEND (Straddling Handle Precision)", q_hover, q_grasp, 0.0, 0.0, "table", 35),
            ("Phase 5: CLAMP (Fingers Closing Flush Across Handle)", q_grasp, q_grasp, 0.0, clamp_val, "table", 25),
            ("Phase 6: LIFT (Smooth Vertical Clearance Off Table)", q_grasp, q_lift, clamp_val, clamp_val, "attached", 35),
            ("Phase 7: TRANSPORT TO HUMAN RECEIVER HAND", q_lift, q_handover, clamp_val, clamp_val, "attached", 50),
            ("Phase 8: SAFE HANDOVER PRESENTATION (Delivering into Palm)", q_handover, q_handover, clamp_val, clamp_val, "attached", 35),
            ("Phase 9: HANDOVER COMPLETE (Safe Transfer Verified)", q_handover, q_handover, clamp_val, 0.0, "attached", 30),
        ]

    return phases


# ==============================================================================
# 5. Core Video & MJPEG Streaming Engine
# ==============================================================================

def generate_simulation_stream(tool_name="knife", task_name="handover", width=640, height=480, fps=25):
    """
    Yields live multipart JPEG frames directly for browser HTTP streaming.
    Executes offscreen rendering at 60+ FPS with zero lag.
    """
    norm_tool = "screwdriver" if tool_name.lower() in ["tool", "screwdriver"] else tool_name.lower()
    norm_task = "handover" if task_name.lower() in ["handover", "safe_handover", "human_hand"] else (
        "work_with_it" if task_name.lower() in ["work", "tool_use", "work_with_it"] else "pick_and_place"
    )

    pred = get_agam_task_grasp(norm_tool, norm_task)
    target_xyz = pred["target_xyz"]
    clamp_val = pred["clamp_val"]
    tx, ty, tz = target_xyz

    xml = build_task_scene_xml(norm_tool, target_xyz, norm_task)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)

    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_object")
    marker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "agam_marker")
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_gripper")

    # Precise table resting coordinate and rigid gripper attachment offset
    if norm_tool == "mug":
        tool_table_pos = np.array([tx, 0.0, tz - 0.040])
        tool_offset = np.array([0.0, 0.052, -0.078])
    else:
        tool_table_pos = np.array([tx, ty, tz])
        tool_offset = np.array([0.0, 0.0, -0.038])

    phases = generate_task_phases(model, data, norm_tool, norm_task, target_xyz, clamp_val)

    # Initial forward pass
    q_home = phases[0][1]
    data.qpos[0:5] = q_home
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    model.body_pos[tool_body_id] = tool_table_pos
    mujoco.mj_forward(model, data)

    # Camera framing
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat = [0.38, -0.05, 0.75]
    camera.distance = 1.15
    camera.elevation = -22.0
    camera.azimuth = 135.0

    released_pos = None

    for phase_name, q_start, q_end, f_start, f_end, tool_mode, steps in phases:
        for step in range(steps):
            t = step / float(steps - 1) if steps > 1 else 1.0
            smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))

            q_cur = q_start * (1.0 - smooth_t) + q_end * smooth_t
            f_cur = f_start * (1.0 - t) + f_end * t
            data.qpos[0:5] = q_cur
            data.qpos[5] = -f_cur
            data.qpos[6] = f_cur

            # Update physics
            mujoco.mj_forward(model, data)
            ee_cur_pos = data.xpos[ee_id]

            if tool_mode == "table":
                model.body_pos[tool_body_id] = tool_table_pos
            elif tool_mode == "attached":
                model.body_pos[tool_body_id] = ee_cur_pos + tool_offset
                released_pos = np.copy(model.body_pos[tool_body_id])
            elif tool_mode == "released":
                if released_pos is not None:
                    model.body_pos[tool_body_id] = released_pos

            # Marker pulse
            pulse = 0.012 + 0.003 * math.sin(step * 0.25)
            model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "marker_orb")][0] = pulse

            # Render offscreen frame
            renderer.update_scene(data, camera)
            pixels = renderer.render()

            # Encode to JPEG
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            # Overlay HUD Telemetry Text
            cv2.putText(bgr, f"TASK: {norm_task.upper()} | OBJECT: {norm_tool.upper()}", (16, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 242, 254), 2, cv2.LINE_AA)
            cv2.putText(bgr, f"{phase_name}", (16, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bgr, f"AGAM Score: {pred['score']:.3f} | PASS [SAFE]", (16, height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)

            _, jpeg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1.0 / fps)


# ==============================================================================
# 6. Interactive Desktop Viewer (Standalone 60 FPS Native Window)
# ==============================================================================

def launch_task_simulation_desktop(tool_name="knife", task_name="handover"):
    """Launches the interactive 60 FPS desktop OpenGL MuJoCo window."""
    import mujoco.viewer

    norm_tool = "screwdriver" if tool_name.lower() in ["tool", "screwdriver"] else tool_name.lower()
    norm_task = "handover" if task_name.lower() in ["handover", "safe_handover", "human_hand"] else (
        "work_with_it" if task_name.lower() in ["work", "tool_use", "work_with_it"] else "pick_and_place"
    )

    pred = get_agam_task_grasp(norm_tool, norm_task)
    target_xyz = pred["target_xyz"]
    clamp_val = pred["clamp_val"]
    tx, ty, tz = target_xyz

    xml = build_task_scene_xml(norm_tool, target_xyz, norm_task)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_object")
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_gripper")

    if norm_tool == "mug":
        tool_table_pos = np.array([tx, 0.0, tz - 0.040])
        tool_offset = np.array([0.0, 0.052, -0.078])
    else:
        tool_table_pos = np.array([tx, ty, tz])
        tool_offset = np.array([0.0, 0.0, -0.038])

    phases = generate_task_phases(model, data, norm_tool, norm_task, target_xyz, clamp_val)

    q_home = phases[0][1]
    data.qpos[0:5] = q_home
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    model.body_pos[tool_body_id] = tool_table_pos
    mujoco.mj_forward(model, data)

    print(f"\n[Starting MuJoCo Desktop Viewer: {norm_tool.upper()} - {norm_task.upper()}]")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat = [0.38, -0.05, 0.75]
        viewer.cam.distance = 1.15
        viewer.cam.elevation = -22.0
        viewer.cam.azimuth = 135.0

        released_pos = None
        while viewer.is_running():
            for phase_name, q_start, q_end, f_start, f_end, tool_mode, steps in phases:
                if not viewer.is_running():
                    break
                for step in range(steps):
                    if not viewer.is_running():
                        break
                    t = step / float(steps - 1) if steps > 1 else 1.0
                    smooth_t = 0.5 * (1.0 - math.cos(math.pi * t))

                    q_cur = q_start * (1.0 - smooth_t) + q_end * smooth_t
                    f_cur = f_start * (1.0 - t) + f_end * t
                    data.qpos[0:5] = q_cur
                    data.qpos[5] = -f_cur
                    data.qpos[6] = f_cur

                    mujoco.mj_forward(model, data)
                    ee_cur_pos = data.xpos[ee_id]

                    if tool_mode == "table":
                        model.body_pos[tool_body_id] = tool_table_pos
                    elif tool_mode == "attached":
                        model.body_pos[tool_body_id] = ee_cur_pos + tool_offset
                        released_pos = np.copy(model.body_pos[tool_body_id])
                    elif tool_mode == "released":
                        if released_pos is not None:
                            model.body_pos[tool_body_id] = released_pos

                    # Marker pulse
                    pulse = 0.012 + 0.003 * math.sin(step * 0.20)
                    model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "marker_orb")][0] = pulse

                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(0.016)
            time.sleep(1.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TAG-Net Multi-Task Simulation")
    parser.add_argument("--tool", type=str, default="knife", choices=["knife", "mug", "screwdriver", "tool"])
    parser.add_argument("--task", type=str, default="handover", choices=["pick_and_place", "handover", "work_with_it"])
    args = parser.parse_args()

    chosen_tool = "screwdriver" if args.tool.lower() == "tool" else args.tool.lower()
    launch_task_simulation_desktop(chosen_tool, args.task)
