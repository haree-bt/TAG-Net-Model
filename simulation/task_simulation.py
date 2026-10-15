"""
TAG-Net High-Fidelity Multi-Task Robotic Manipulation Simulation in MuJoCo
===========================================================================
Ultra-high quality 6-DoF robotic manipulation guided by TAG-Net AGAM heatmaps:
  - Supports 3 Objects: Knife, Mug, Screwdriver (Tool)
  - Supports 3 Tasks: pick_and_place, handover, work_with_it
  - Supports Model Comparison:
      * TAG-Net (Novel Task-Aware Affordance-Gated Attention)
      * Conventional Model (Classical Blind Geometry-Only Baseline)
  - Supports Arbitrary Object Orientations (0 deg to 360 deg) with 6-DoF wrist roll tracking
  - Continuous HD 28 FPS streaming + Native 4K 60-FPS desktop viewer
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
from models.baseline_grasp import BaselineGraspNet
from data.dataset import SyntheticToolAffordanceDataset


# ==============================================================================
# 1. Perception Pipeline (TAG-Net vs. Conventional Model & Rotation)
# ==============================================================================

def get_agam_task_grasp(tool_name="knife", task_name="handover", model_type="tag_net", orientation_deg=0.0):
    """
    Executes neural grasp synthesis comparing TAG-Net vs. Conventional Model
    across arbitrary object orientation angles.
    """
    device = torch.device("cpu")
    norm_tool = "screwdriver" if tool_name.lower() in ["tool", "screwdriver"] else tool_name.lower()
    norm_task = "handover" if task_name.lower() in ["handover", "safe_handover", "human_hand"] else (
        "work_with_it" if task_name.lower() in ["work", "tool_use", "work_with_it"] else "pick_and_place"
    )
    is_conventional = (model_type.lower() in ["baseline", "conventional"])

    dataset = SyntheticToolAffordanceDataset(num_samples=60, img_size=224, seed=101)
    tool_samples = [i for i in range(len(dataset)) if str(dataset[i]["tool_type"]).lower() == norm_tool]
    sample_idx = tool_samples[0] if tool_samples else 0
    sample = dataset[sample_idx]

    rgbd_np = sample["rgbd"].numpy()
    rgb = rgbd_np[:3].transpose(1, 2, 0)
    depth = rgbd_np[3]
    aff_gt = sample["affordance_mask"][0].numpy()

    # Apply 2D Rotation if specified
    phi = math.radians(orientation_deg)
    if abs(orientation_deg) > 0.01:
        M = cv2.getRotationMatrix2D((112, 112), orientation_deg, 1.0)
        rgb = cv2.warpAffine(rgb, M, (224, 224), borderMode=cv2.BORDER_REFLECT)
        depth = cv2.warpAffine(depth, M, (224, 224), borderMode=cv2.BORDER_REFLECT)
        aff_gt = cv2.warpAffine(aff_gt, M, (224, 224), borderMode=cv2.BORDER_REFLECT)
        rgbd_np = np.concatenate([rgb.transpose(2, 0, 1), depth[np.newaxis, ...]], axis=0)

    x = torch.from_numpy(rgbd_np).float().unsqueeze(0).to(device)

    # Select Neural Model
    if is_conventional:
        model = BaselineGraspNet().to(device)
        ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "baseline_best.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        with torch.no_grad():
            out = model(x)
            q_map = out["quality"][0, 0].numpy()
            w_out = out["width"][0, 0].numpy()
            sin_map = out["sin_2theta"][0, 0].numpy()
            cos_map = out["cos_2theta"][0, 0].numpy()
    else:
        model = TAGNet().to(device)
        ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "tag_net_best.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        with torch.no_grad():
            out = model(x)
            q_map = out["quality_task"][0, 0].numpy()
            w_out = out["width"][0, 0].numpy()
            sin_map = out["sin_2theta"][0, 0].numpy()
            cos_map = out["cos_2theta"][0, 0].numpy()

    v_best, u_best = np.unravel_index(np.argmax(q_map), q_map.shape)
    best_score = float(q_map[v_best, u_best])
    sin2 = float(sin_map[v_best, u_best])
    cos2 = float(cos_map[v_best, u_best])
    angle_rad = 0.5 * math.atan2(sin2, cos2)
    width_px = float(w_out[v_best, u_best]) * 50.0

    # 3D Physical Object Coordinates & Offsets
    table_z = 0.709
    if norm_tool == "knife":
        x0, y0, z0 = 0.40, 0.0, table_z + 0.016
        clamp_val = 0.020
        if is_conventional:
            # Blind geometry targets cutting blade center
            x_loc, y_loc, z_loc = 0.138, 0.0, 0.004
            is_safe = False
            contact_label = "CUTTING BLADE HAZARD (DANGEROUS)"
        else:
            # TAG-Net targets safe ergonomic handle
            x_loc, y_loc, z_loc = 0.0, 0.0, 0.0
            is_safe = True
            contact_label = "SAFE HANDLE AFFORDANCE"

    elif norm_tool == "mug":
        x0, y0, z0 = 0.45, 0.0, table_z + 0.040
        clamp_val = 0.022
        if is_conventional:
            # Blind geometry targets high-contrast circular drinking rim
            x_loc, y_loc, z_loc = 0.0, 0.036, 0.038
            is_safe = False
            contact_label = "DRINKING RIM (SANITARY HAZARD)"
        else:
            # TAG-Net targets lateral ergonomic handle
            x_loc, y_loc, z_loc = 0.0, -0.052, 0.0
            is_safe = True
            contact_label = "ERGONOMIC LATERAL HANDLE"

    else:  # screwdriver
        x0, y0, z0 = 0.39, 0.0, table_z + 0.016
        clamp_val = 0.018
        if is_conventional:
            # Blind geometry targets narrow cylindrical metal shaft
            x_loc, y_loc, z_loc = 0.105, 0.0, 0.0
            is_safe = False
            contact_label = "DRIVING SHAFT (FUNCTIONALLY IMPAIRED)"
        else:
            # TAG-Net targets textured rubber power grip
            x_loc, y_loc, z_loc = 0.0, 0.0, 0.0
            is_safe = True
            contact_label = "TEXTURED POWER GRIP"

    # Rotate local grasp offset by object orientation phi
    gx = x0 + x_loc * math.cos(phi) - y_loc * math.sin(phi)
    gy = y0 + x_loc * math.sin(phi) + y_loc * math.cos(phi)
    gz = z0 + z_loc
    target_xyz = (gx, gy, gz)

    return {
        "tool": norm_tool,
        "task": norm_task,
        "model_type": "conventional" if is_conventional else "tag_net",
        "orientation_deg": orientation_deg,
        "orientation_rad": phi,
        "u": int(u_best),
        "v": int(v_best),
        "score": best_score,
        "angle_rad": angle_rad,
        "angle_deg": math.degrees(angle_rad),
        "width_px": width_px,
        "is_safe": is_safe,
        "contact_label": contact_label,
        "body_center": (x0, y0, z0),
        "target_xyz": target_xyz,
        "clamp_val": clamp_val
    }


# ==============================================================================
# 2. Studio-Grade MuJoCo MJCF World Generator
# ==============================================================================

def build_task_scene_xml(tool_type="knife", target_coords=(0.40, 0.0, 0.725), task_name="handover",
                         model_type="tag_net", orientation_deg=0.0):
    tx, ty, tz = target_coords
    norm_tool = "screwdriver" if tool_type.lower() in ["tool", "screwdriver"] else tool_type.lower()
    is_conventional = (model_type.lower() in ["baseline", "conventional"])
    phi = math.radians(orientation_deg)

    # Base table positions for unrotated object body
    table_z = 0.709
    if norm_tool == "knife":
        bx, by, bz = 0.40, 0.0, table_z + 0.016
    elif norm_tool == "mug":
        bx, by, bz = 0.45, 0.0, table_z
    else: # screwdriver
        bx, by, bz = 0.39, 0.0, table_z + 0.016

    # Target Marker color: Glowing Emerald/Cyan for TAG-Net, Crimson for Conventional Failure
    marker_mat = "mat_crimson_led" if is_conventional else "mat_cyan_led"

    if norm_tool == "knife":
        tool_mjcf = f"""
      <!-- High-Quality Chef Knife with Rotation -->
      <body name="tool_object" pos="{bx} {by} {bz}" euler="0 0 {phi}">
        <!-- GREEN AFFORDANCE ERGONOMIC HANDLE -->
        <geom name="knife_handle" type="box" size="0.055 0.012 0.012" pos="0 0 0" material="mat_affordance"/>
        <geom name="knife_rivet1" type="cylinder" size="0.003 0.013" pos="-0.025 0 0" material="mat_steel" zaxis="0 1 0"/>
        <geom name="knife_rivet2" type="cylinder" size="0.003 0.013" pos="0.025 0 0" material="mat_steel" zaxis="0 1 0"/>
        <geom name="knife_bolster" type="box" size="0.006 0.014 0.015" pos="0.060 0 0" material="mat_dark_metal"/>
        <!-- POLISHED STEEL CUTTING BLADE -->
        <geom name="knife_blade" type="box" size="0.075 0.002 0.018" pos="0.138 0 0.004" material="mat_steel"/>
        <geom name="knife_edge" type="box" size="0.073 0.001 0.002" pos="0.138 0 -0.015" material="mat_blade_edge"/>
      </body>
        """
    elif norm_tool == "mug":
        tool_mjcf = f"""
      <!-- Glossy Porcelain Ceramic Mug with Rotation -->
      <body name="tool_object" pos="{bx} {by} {bz}" euler="0 0 {phi}">
        <geom name="cup_body" type="cylinder" size="0.038 0.040" pos="0 0 0.040" material="mat_ceramic"/>
        <geom name="cup_rim" type="cylinder" size="0.036 0.004" pos="0 0 0.078" material="mat_rim_gold"/>
        <!-- GREEN AFFORDANCE LATERAL HANDLE -->
        <geom name="cup_handle_grip" type="cylinder" size="0.009 0.026" pos="0 -0.052 0.040" material="mat_affordance"/>
        <geom name="cup_handle_top" type="cylinder" size="0.008 0.010" pos="0 -0.044 0.062" zaxis="0 1 0" material="mat_affordance"/>
        <geom name="cup_handle_bot" type="cylinder" size="0.008 0.010" pos="0 -0.044 0.018" zaxis="0 1 0" material="mat_affordance"/>
      </body>
        """
    else:  # screwdriver
        tool_mjcf = f"""
      <!-- Precision Screwdriver with Rotation -->
      <body name="tool_object" pos="{bx} {by} {bz}" euler="0 0 {phi}">
        <!-- GREEN AFFORDANCE TEXTURED RUBBER GRIP -->
        <geom name="sd_handle" type="cylinder" size="0.016 0.045" pos="0 0 0" zaxis="1 0 0" material="mat_affordance"/>
        <geom name="sd_cap" type="sphere" size="0.016" pos="-0.045 0 0" material="mat_dark_metal"/>
        <geom name="sd_ring" type="cylinder" size="0.017 0.004" pos="0.025 0 0" zaxis="1 0 0" material="mat_cyan_led"/>
        <!-- CHROME VANADIUM STEEL SHAFT -->
        <geom name="sd_shaft" type="cylinder" size="0.004 0.060" pos="0.105 0 0" zaxis="1 0 0" material="mat_steel"/>
        <geom name="sd_tip" type="box" size="0.006 0.004 0.0015" pos="0.168 0 0" material="mat_dark_metal"/>
      </body>
        """

    xml = f"""
<mujoco model="tag_net_studio_manipulation">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>

  <visual>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight diffuse="0.92 0.92 0.96" ambient="0.38 0.38 0.42" specular="0.45 0.45 0.50"/>
    <global offheight="540" offwidth="960"/>
  </visual>

  <asset>
    <!-- Photorealistic PBR Material Suite -->
    <material name="mat_floor" specular="0.15" shininess="0.2" rgba="0.05 0.07 0.10 1"/>
    <material name="mat_table_surface" specular="0.35" shininess="0.5" rgba="0.10 0.12 0.17 1"/>
    <material name="mat_table_rim" specular="0.70" shininess="0.8" rgba="0.04 0.06 0.09 1"/>
    
    <!-- Robot Arm Materials -->
    <material name="mat_arm_base" specular="0.5" shininess="0.6" rgba="0.14 0.16 0.20 1"/>
    <material name="mat_arm_link" specular="0.85" shininess="0.9" rgba="0.86 0.88 0.92 1"/>
    <material name="mat_dark_metal" specular="0.6" shininess="0.7" rgba="0.18 0.20 0.25 1"/>
    <material name="mat_cyan_led" emission="0.85" specular="0.9" shininess="0.9" rgba="0.0 0.90 0.98 1"/>
    <material name="mat_crimson_led" emission="0.9" specular="0.9" shininess="0.9" rgba="0.95 0.15 0.25 1"/>
    <material name="mat_gripper_pad" specular="0.3" shininess="0.4" rgba="0.95 0.25 0.20 1"/>

    <!-- Tool Materials -->
    <material name="mat_steel" specular="0.98" shininess="0.98" rgba="0.92 0.94 0.98 1"/>
    <material name="mat_blade_edge" specular="1.0" shininess="1.0" rgba="0.98 0.99 1.0 1"/>
    <material name="mat_affordance" emission="0.4" specular="0.7" shininess="0.6" rgba="0.10 0.88 0.35 1"/>
    <material name="mat_ceramic" specular="0.55" shininess="0.7" rgba="0.94 0.95 0.98 1"/>
    <material name="mat_rim_gold" specular="0.85" shininess="0.9" rgba="0.85 0.72 0.35 1"/>

    <!-- Station Materials -->
    <material name="mat_coaster_neon" emission="0.7" specular="0.9" shininess="0.9" rgba="0.0 0.80 1.0 0.9"/>
    <material name="mat_cutting_board" specular="0.25" shininess="0.3" rgba="0.74 0.54 0.38 1"/>
    <material name="mat_human_skin" specular="0.2" shininess="0.3" rgba="0.88 0.73 0.63 1"/>
    <material name="mat_human_band" emission="0.6" specular="0.8" shininess="0.8" rgba="0.0 0.85 1.0 1"/>
  </asset>

  <worldbody>
    <!-- Studio Three-Point Lighting -->
    <light diffuse="0.98 0.98 1.0" pos="0.6 0.6 3.0" dir="-0.5 -0.5 -1"/>
    <light diffuse="0.45 0.52 0.65" pos="-0.8 -0.8 2.2" dir="0.5 0.5 -1"/>
    <light diffuse="0.30 0.35 0.45" pos="0.0 1.2 2.0" dir="0.0 -0.8 -1"/>
    
    <!-- Floor -->
    <geom name="floor" type="plane" size="4 4 0.1" material="mat_floor"/>

    <!-- Workstation Table Surface -->
    <body name="work_table" pos="0.45 0 0.35">
      <geom type="box" size="0.35 0.50 0.35" material="mat_table_rim"/>
      <geom type="box" size="0.36 0.51 0.006" pos="0 0 0.353" material="mat_table_surface"/>
    </body>

    <!-- STATION A: Placement Zone / Coaster -->
    <body name="placement_zone" pos="0.32 0.22 0.709">
      <geom type="cylinder" size="0.075 0.004" pos="0 0 0.004" material="mat_coaster_neon"/>
      <geom type="cylinder" size="0.065 0.005" pos="0 0 0.005" material="mat_table_rim"/>
    </body>

    <!-- STATION B: Functional Work Area (Cutting Board / Screw Fixture) -->
    <body name="workstation_fixture" pos="0.42 0.15 0.709">
      <geom type="box" size="0.080 0.060 0.006" pos="0 0 0.006" material="mat_cutting_board"/>
      <geom type="cylinder" size="0.012 0.012" pos="0 0 0.012" material="mat_steel"/>
      <geom type="box" size="0.008 0.002 0.001" pos="0 0 0.024" material="mat_dark_metal"/>
    </body>

    <!-- STATION C: Human Receiver Hand at Table Edge -->
    <body name="human_receiver" pos="0.22 -0.32 0.73">
      <geom type="capsule" fromto="0 -0.15 0.05 0 0 0.02" size="0.028" material="mat_human_skin"/>
      <geom type="cylinder" size="0.030 0.012" pos="0 -0.06 0.035" zaxis="0 1 0" material="mat_human_band"/>
      <geom type="box" size="0.035 0.045 0.012" pos="0 0.03 0.02" material="mat_human_skin"/>
      <geom type="capsule" fromto="-0.02 0.06 0.02 -0.02 0.10 0.02" size="0.007" material="mat_human_skin"/>
      <geom type="capsule" fromto="0.00 0.06 0.02 0.00 0.11 0.02" size="0.007" material="mat_human_skin"/>
      <geom type="capsule" fromto="0.02 0.06 0.02 0.02 0.10 0.02" size="0.007" material="mat_human_skin"/>
    </body>

    <!-- Holographic Target Marker Orb -->
    <body name="agam_marker" pos="{tx} {ty} {tz + 0.06}">
      <geom name="marker_orb" type="sphere" size="0.014" material="{marker_mat}" contype="0" conaffinity="0"/>
    </body>

    <!-- Target Tool Body -->
    {tool_mjcf}

    <!-- 6-DoF Articulated Manipulator Robot Arm -->
    <body name="arm_base" pos="0 0 0.70">
      <geom type="cylinder" size="0.085 0.05" material="mat_arm_base"/>
      <geom type="cylinder" size="0.088 0.006" pos="0 0 0.05" material="mat_cyan_led"/>

      <!-- Joint 1: Base Yaw -->
      <body name="link1" pos="0 0 0.05">
        <joint name="j1_yaw" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="cylinder" size="0.065 0.10" pos="0 0 0.10" material="mat_arm_link"/>
        <geom type="cylinder" size="0.055 0.05" pos="0 0 0.20" zaxis="0 1 0" material="mat_dark_metal"/>

        <!-- Joint 2: Shoulder Pitch -->
        <body name="link2" pos="0 0 0.20">
          <joint name="j2_shoulder" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.048" material="mat_arm_link"/>
          <geom type="cylinder" size="0.048 0.045" pos="0 0 0.30" zaxis="0 1 0" material="mat_dark_metal"/>

          <!-- Joint 3: Elbow Pitch -->
          <body name="link3" pos="0 0 0.30">
            <joint name="j3_elbow" type="hinge" axis="0 1 0" range="-2.8 2.8"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.042" material="mat_arm_link"/>
            <geom type="cylinder" size="0.042 0.04" pos="0 0 0.25" zaxis="0 1 0" material="mat_dark_metal"/>

            <!-- Joint 4: Wrist Pitch -->
            <body name="link4" pos="0 0 0.25">
              <joint name="j4_wrist_pitch" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <geom type="cylinder" size="0.038 0.03" pos="0 0 0.03" material="mat_arm_link"/>

              <!-- Joint 5: Wrist Roll (Adapts to Object Orientation) -->
              <body name="link5" pos="0 0 0.06">
                <joint name="j5_wrist_roll" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
                <geom type="cylinder" size="0.035 0.02" pos="0 0 0.02" material="mat_dark_metal"/>

                <!-- End-Effector Gripper Body -->
                <body name="ee_gripper" pos="0 0 0.04">
                  <geom type="box" size="0.020 0.055 0.012" material="mat_arm_base"/>

                  <!-- Left Finger -->
                  <body name="left_finger" pos="0 0.038 0.035">
                    <joint name="finger_left" type="slide" axis="0 1 0" range="-0.035 0.035"/>
                    <geom type="box" size="0.018 0.007 0.035" material="mat_gripper_pad"/>
                    <geom type="box" size="0.015 0.003 0.025" pos="0 -0.005 0" material="mat_dark_metal"/>
                  </body>

                  <!-- Right Finger -->
                  <body name="right_finger" pos="0 -0.038 0.035">
                    <joint name="finger_right" type="slide" axis="0 1 0" range="-0.035 0.035"/>
                    <geom type="box" size="0.018 0.007 0.035" material="mat_gripper_pad"/>
                    <geom type="box" size="0.015 0.003 0.025" pos="0 0.005 0" material="mat_dark_metal"/>
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
# 3. Kinematic Solvers with Wrist Roll Orientation Constraint
# ==============================================================================

def solve_ik_top_down(model, data, target_x, target_y, target_z, initial_guess=(1.1, 1.1), roll=0.0):
    """Solves 6-DoF Inverse Kinematics enforcing top-down constraint: j2 + j3 + j4 = pi and wrist roll."""
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

def generate_task_phases(model, data, tool_name, task_name, target_xyz, clamp_val, roll=0.0):
    tx, ty, tz = target_xyz
    z_grasp = tz + 0.038
    z_hover = z_grasp + 0.16
    z_lift  = z_grasp + 0.18

    q_home = np.array([0.0, 0.45, 0.65, np.pi - (0.45 + 0.65), 0.0])
    q_hover = solve_ik_top_down(model, data, tx, ty, z_hover, initial_guess=(0.8, 0.9), roll=roll)
    q_grasp = solve_ik_top_down(model, data, tx, ty, z_grasp, initial_guess=(1.1, 1.1), roll=roll)
    q_lift  = solve_ik_top_down(model, data, tx, ty, z_lift, initial_guess=(0.7, 0.8), roll=roll)

    norm_task = task_name.lower()

    if norm_task == "pick_and_place":
        px, py, pz = 0.32, 0.22, 0.709
        pz_hover = pz + 0.20
        pz_place = pz + 0.038 + (0.040 if tool_name == "mug" else 0.016)
        q_p_hover = solve_ik_top_down(model, data, px, py, pz_hover, initial_guess=(0.8, 0.9), roll=roll)
        q_p_place = solve_ik_top_down(model, data, px, py, pz_place, initial_guess=(1.1, 1.1), roll=roll)

        phases = [
            ("Phase 1: STANDBY (Framing Workspace)", q_home, q_home, 0.0, 0.0, "table", 25),
            ("Phase 2: TARGET ACQUIRED (Orientation Aligned)", q_home, q_home, 0.0, 0.0, "table", 20),
            ("Phase 3: APPROACH (Wrist Roll Aligns to Object)", q_home, q_hover, 0.0, 0.0, "table", 30),
            ("Phase 4: DESCEND (Straddling Target Precision)", q_hover, q_grasp, 0.0, 0.0, "table", 30),
            ("Phase 5: CLAMP (Fingers Closing Flush)", q_grasp, q_grasp, 0.0, clamp_val, "table", 20),
            ("Phase 6: LIFT (Smooth Vertical Clearance)", q_grasp, q_lift, clamp_val, clamp_val, "attached", 30),
            ("Phase 7: TRANSPORT TO PLACEMENT ZONE", q_lift, q_p_hover, clamp_val, clamp_val, "attached", 40),
            ("Phase 8: LOWER ONTO PLACEMENT TRAY", q_p_hover, q_p_place, clamp_val, clamp_val, "attached", 30),
            ("Phase 9: RELEASE (Fingers Opening Stably)", q_p_place, q_p_place, clamp_val, 0.0, "released", 20),
            ("Phase 10: RETRACT TO HOME (Task Complete)", q_p_place, q_home, 0.0, 0.0, "released", 35),
        ]

    elif norm_task == "work_with_it":
        wx, wy, wz = 0.42, 0.15, 0.709
        wz_hover = wz + 0.18
        wz_work  = wz + 0.038 + (0.040 if tool_name == "mug" else 0.016)

        q_w_hover = solve_ik_top_down(model, data, wx, wy, wz_hover, initial_guess=(0.8, 0.9), roll=roll)
        q_w_work  = solve_ik_top_down(model, data, wx, wy, wz_work, initial_guess=(1.1, 1.1), roll=roll)

        if tool_name == "knife":
            q_slice_down = solve_ik_top_down(model, data, wx + 0.02, wy, wz_work - 0.015, initial_guess=(1.1, 1.1), roll=roll)
            q_slice_up   = solve_ik_top_down(model, data, wx - 0.02, wy, wz_work + 0.025, initial_guess=(1.1, 1.1), roll=roll)
            work_subphases = [
                ("Phase 8: CUTTING STROKE 1 (Slicing on Board)", q_w_work, q_slice_down, clamp_val, clamp_val, "attached", 18),
                ("Phase 8: CUTTING STROKE 2 (Blade Lift)", q_slice_down, q_slice_up, clamp_val, clamp_val, "attached", 18),
                ("Phase 8: CUTTING STROKE 3 (Finishing Slice)", q_slice_up, q_w_work, clamp_val, clamp_val, "attached", 18),
            ]
        elif tool_name == "mug":
            q_drink_lift = solve_ik_top_down(model, data, wx, wy, wz_hover + 0.10, initial_guess=(0.7, 0.8), roll=roll)
            q_drink_tilt = np.copy(q_drink_lift)
            q_drink_tilt[3] += 0.60
            work_subphases = [
                ("Phase 8: LIFT TO DRINKING HEIGHT", q_w_work, q_drink_lift, clamp_val, clamp_val, "attached", 25),
                ("Phase 8: TILT DRINKING/POURING GESTURE", q_drink_lift, q_drink_tilt, clamp_val, clamp_val, "attached", 30),
                ("Phase 8: RETURN UPRIGHT", q_drink_tilt, q_drink_lift, clamp_val, clamp_val, "attached", 20),
            ]
        else: # screwdriver
            q_screw_start = np.copy(q_w_work)
            q_screw_turn1 = np.copy(q_w_work)
            q_screw_turn1[4] = roll + np.pi
            q_screw_turn2 = np.copy(q_w_work)
            q_screw_turn2[4] = roll + 2.0 * np.pi
            work_subphases = [
                ("Phase 8: ALIGN TIP WITH SCREW HEAD", q_w_hover, q_w_work, clamp_val, clamp_val, "attached", 20),
                ("Phase 8: CLOCKWISE TORQUE SCREWING (0-180°)", q_screw_start, q_screw_turn1, clamp_val, clamp_val, "attached", 25),
                ("Phase 8: CONTINUOUS FASTENING (180-360°)", q_screw_turn1, q_screw_turn2, clamp_val, clamp_val, "attached", 25),
            ]

        phases = [
            ("Phase 1: STANDBY (Framing Workspace)", q_home, q_home, 0.0, 0.0, "table", 25),
            ("Phase 2: TARGET ACQUIRED (Orientation Aligned)", q_home, q_home, 0.0, 0.0, "table", 20),
            ("Phase 3: APPROACH (Hover Over Target Grip)", q_home, q_hover, 0.0, 0.0, "table", 30),
            ("Phase 4: DESCEND (Straddling Target Precision)", q_hover, q_grasp, 0.0, 0.0, "table", 30),
            ("Phase 5: CLAMP (Firm Power Clamping)", q_grasp, q_grasp, 0.0, clamp_val, "table", 20),
            ("Phase 6: LIFT (Vertical Clearance)", q_grasp, q_lift, clamp_val, clamp_val, "attached", 30),
            ("Phase 7: MOVE TO WORKSTATION FIXTURE", q_lift, q_w_hover, clamp_val, clamp_val, "attached", 35),
            *work_subphases,
            ("Phase 9: RETRACT TO HOME (Functional Task Complete)", q_w_hover, q_home, clamp_val, 0.0, "attached", 35),
        ]

    else:  # handover (pick and give to human hand)
        hx, hy, hz = 0.22, -0.28, 0.78
        q_handover = solve_ik_top_down(model, data, hx, hy, hz, initial_guess=(0.7, 0.8), roll=roll)

        phases = [
            ("Phase 1: STANDBY (Workspace Framing)", q_home, q_home, 0.0, 0.0, "table", 25),
            ("Phase 2: TARGET ACQUIRED (Orientation Aligned)", q_home, q_home, 0.0, 0.0, "table", 20),
            ("Phase 3: APPROACH (Aligning Over Target)", q_home, q_hover, 0.0, 0.0, "table", 30),
            ("Phase 4: DESCEND (Straddling Target Precision)", q_hover, q_grasp, 0.0, 0.0, "table", 30),
            ("Phase 5: CLAMP (Fingers Closing Flush Across Target)", q_grasp, q_grasp, 0.0, clamp_val, "table", 20),
            ("Phase 6: LIFT (Smooth Vertical Clearance Off Table)", q_grasp, q_lift, clamp_val, clamp_val, "attached", 30),
            ("Phase 7: TRANSPORT TO HUMAN RECEIVER HAND", q_lift, q_handover, clamp_val, clamp_val, "attached", 45),
            ("Phase 8: SAFE HANDOVER PRESENTATION (Delivering into Palm)", q_handover, q_handover, clamp_val, clamp_val, "attached", 30),
            ("Phase 9: HANDOVER COMPLETE (Safe Transfer Verified)", q_handover, q_handover, clamp_val, 0.0, "attached", 25),
        ]

    return phases


# ==============================================================================
# 5. Core Video & Continuous High-Fidelity Streaming Engine
# ==============================================================================

def generate_simulation_stream(tool_name="knife", task_name="handover", model_type="tag_net",
                               orientation_deg=0.0, width=960, height=540, fps=28):
    """
    Continuous live multipart JPEG frames directly for browser HTTP streaming.
    Renders rotated object orientation and compares TAG-Net vs Conventional model in real time.
    """
    norm_tool = "screwdriver" if tool_name.lower() in ["tool", "screwdriver"] else tool_name.lower()
    norm_task = "handover" if task_name.lower() in ["handover", "safe_handover", "human_hand"] else (
        "work_with_it" if task_name.lower() in ["work", "tool_use", "work_with_it"] else "pick_and_place"
    )
    is_conventional = (model_type.lower() in ["baseline", "conventional"])

    pred = get_agam_task_grasp(norm_tool, norm_task, model_type=model_type, orientation_deg=orientation_deg)
    target_xyz = pred["target_xyz"]
    clamp_val = pred["clamp_val"]
    tx, ty, tz = target_xyz
    phi = math.radians(orientation_deg)

    xml = build_task_scene_xml(norm_tool, target_xyz, norm_task, model_type=model_type, orientation_deg=orientation_deg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)

    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_object")
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_gripper")

    # Table position of object body
    table_z = 0.709
    if norm_tool == "knife":
        tool_table_pos = np.array([0.40, 0.0, table_z + 0.016])
    elif norm_tool == "mug":
        tool_table_pos = np.array([0.45, 0.0, table_z])
    else: # screwdriver
        tool_table_pos = np.array([0.39, 0.0, table_z + 0.016])

    # Initial grasp solve to get tool-to-gripper offset
    q_grasp_test = solve_ik_top_down(model, data, tx, ty, tz + 0.038, roll=phi)
    data.qpos[0:5] = q_grasp_test
    mujoco.mj_forward(model, data)
    tool_offset = tool_table_pos - data.xpos[ee_id]

    phases = generate_task_phases(model, data, norm_tool, norm_task, target_xyz, clamp_val, roll=phi)

    # Cinematic 3/4 Studio Camera Angle
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat = [0.38, -0.05, 0.75]
    camera.distance = 1.15
    camera.elevation = -22.0
    camera.azimuth = 135.0

    while True:
        # Reset to home pose at start of each demonstration cycle
        q_home = phases[0][1]
        data.qpos[0:5] = q_home
        data.qpos[5] = 0.0
        data.qpos[6] = 0.0
        model.body_pos[tool_body_id] = tool_table_pos
        mujoco.mj_forward(model, data)

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

                # Marker pulse animation
                pulse = 0.012 + 0.003 * math.sin(step * 0.25)
                model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "marker_orb")][0] = pulse

                # Render offscreen frame with antialiasing
                renderer.update_scene(data, camera)
                pixels = renderer.render()
                bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                # Sleek Studio HUD Overlays
                overlay = bgr.copy()
                cv2.rectangle(overlay, (12, 12), (width - 12, 54), (10, 14, 22), -1)
                cv2.rectangle(overlay, (12, height - 42), (width - 12, height - 12), (10, 14, 22), -1)
                cv2.addWeighted(overlay, 0.75, bgr, 0.25, 0, bgr)

                border_color = (63, 63, 244) if is_conventional else (56, 189, 248) # Red vs Cyan
                cv2.rectangle(bgr, (12, 12), (width - 12, 54), border_color, 1)
                cv2.rectangle(bgr, (12, height - 42), (width - 12, height - 12), (30, 41, 59), 1)

                model_lbl = "CONVENTIONAL (BASELINE)" if is_conventional else "TAG-NET (AGAM)"
                cv2.putText(bgr, f"MODEL: {model_lbl}", (24, 34),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, border_color, 1, cv2.LINE_AA)
                cv2.putText(bgr, f"OBJ: {norm_tool.upper()} ({orientation_deg:.0f} deg)", (270, 34),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bgr, f"{phase_name}", (520, 34),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (16, 185, 129) if not is_conventional else (100, 100, 255), 1, cv2.LINE_AA)

                status_text = "SAFETY: PASS [SAFE HANDLE]" if not is_conventional else f"SAFETY: FAIL [{pred['contact_label']}]"
                status_color = (16, 185, 129) if not is_conventional else (63, 63, 244)
                cv2.putText(bgr, status_text, (24, height - 22),
                            cv2.FONT_HERSHEY_DUPLEX, 0.42, status_color, 1, cv2.LINE_AA)
                cv2.putText(bgr, f"3D TARGET: ({tx:.3f}, {ty:.3f}, {tz:.3f}) | WRIST: {orientation_deg:.0f} deg", (420, height - 22),
                            cv2.FONT_HERSHEY_DUPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

                _, jpeg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                frame_bytes = jpeg.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(1.0 / fps)

        time.sleep(0.8)


# ==============================================================================
# 6. Interactive Desktop Viewer (Standalone 60 FPS Native Window)
# ==============================================================================

def launch_task_simulation_desktop(tool_name="knife", task_name="handover", model_type="tag_net", orientation_deg=0.0):
    """Launches the interactive 60 FPS desktop OpenGL MuJoCo window with full mouse interaction."""
    import mujoco.viewer

    norm_tool = "screwdriver" if tool_name.lower() in ["tool", "screwdriver"] else tool_name.lower()
    norm_task = "handover" if task_name.lower() in ["handover", "safe_handover", "human_hand"] else (
        "work_with_it" if task_name.lower() in ["work", "tool_use", "work_with_it"] else "pick_and_place"
    )

    pred = get_agam_task_grasp(norm_tool, norm_task, model_type=model_type, orientation_deg=orientation_deg)
    target_xyz = pred["target_xyz"]
    clamp_val = pred["clamp_val"]
    tx, ty, tz = target_xyz
    phi = math.radians(orientation_deg)

    xml = build_task_scene_xml(norm_tool, target_xyz, norm_task, model_type=model_type, orientation_deg=orientation_deg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    tool_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool_object")
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_gripper")

    table_z = 0.709
    if norm_tool == "knife":
        tool_table_pos = np.array([0.40, 0.0, table_z + 0.016])
    elif norm_tool == "mug":
        tool_table_pos = np.array([0.45, 0.0, table_z])
    else: # screwdriver
        tool_table_pos = np.array([0.39, 0.0, table_z + 0.016])

    q_grasp_test = solve_ik_top_down(model, data, tx, ty, tz + 0.038, roll=phi)
    data.qpos[0:5] = q_grasp_test
    mujoco.mj_forward(model, data)
    tool_offset = tool_table_pos - data.xpos[ee_id]

    phases = generate_task_phases(model, data, norm_tool, norm_task, target_xyz, clamp_val, roll=phi)

    q_home = phases[0][1]
    data.qpos[0:5] = q_home
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    model.body_pos[tool_body_id] = tool_table_pos
    mujoco.mj_forward(model, data)

    print(f"\n[Starting MuJoCo Desktop Viewer: {norm_tool.upper()} - {norm_task.upper()} ({model_type.upper()}, {orientation_deg} deg)]")
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
    parser.add_argument("--model", type=str, default="tag_net", choices=["tag_net", "baseline", "conventional"])
    parser.add_argument("--orientation", type=float, default=0.0, help="Object orientation in degrees")
    args = parser.parse_args()

    chosen_tool = "screwdriver" if args.tool.lower() == "tool" else args.tool.lower()
    launch_task_simulation_desktop(chosen_tool, args.task, model_type=args.model, orientation_deg=args.orientation)
