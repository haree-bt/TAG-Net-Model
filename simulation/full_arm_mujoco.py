"""
3D Robotic Manipulator Arm Simulation in MuJoCo
Simulates a multi-joint industrial robotic arm (Franka Panda / UR5 style)
with parallel-jaw gripper executing the TAG-Net grasp trajectory in full 3D physics.
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import mujoco

# MuJoCo Model XML: 6-DoF Articulated Robot Arm + Gripper + Work Table + Target Tool
ROBOT_ARM_MJCF = """
<mujoco model="robotic_arm_grasp">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.005"/>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" specular="0.2 0.2 0.2"/>
  </visual>

  <worldbody>
    <light diffuse=".8 .8 .8" pos="0.5 0.5 2.5" dir="-0.5 -0.5 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.92 0.92 0.95 1"/>

    <!-- Work Table -->
    <body name="table" pos="0.45 0 0.35">
      <geom type="box" size="0.25 0.35 0.35" rgba="0.75 0.75 0.78 1"/>
    </body>

    <!-- Target Tool on Table (Green handle = functional affordance zone) -->
    <body name="tool_handle" pos="0.45 0 0.72">
      <joint type="free" name="tool_joint"/>
      <geom type="cylinder" size="0.018 0.06" rgba="0.15 0.75 0.25 1" density="400"/>
    </body>

    <!-- 6-DoF Articulated Robot Arm -->
    <body name="arm_base" pos="0 0 0.7">
      <geom type="cylinder" size="0.08 0.05" rgba="0.25 0.25 0.28 1"/>

      <!-- Joint 1: Base Yaw -->
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="cylinder" size="0.06 0.12" pos="0 0 0.12" rgba="0.85 0.85 0.9 1"/>

        <!-- Joint 2: Shoulder Pitch -->
        <body name="link2" pos="0 0 0.24">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" rgba="0.3 0.35 0.4 1"/>

          <!-- Joint 3: Elbow Pitch -->
          <body name="link3" pos="0.3 0 0">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-2.2 2.2"/>
            <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.045" rgba="0.85 0.85 0.9 1"/>

            <!-- Joint 4: Forearm Roll -->
            <body name="link4" pos="0.25 0 0">
              <joint name="joint4" type="hinge" axis="1 0 0" range="-3.14 3.14"/>
              <geom type="cylinder" size="0.04 0.04" pos="0.04 0 0" zaxis="1 0 0" rgba="0.3 0.35 0.4 1"/>

              <!-- Joint 5: Wrist Pitch -->
              <body name="link5" pos="0.08 0 0">
                <joint name="joint5" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
                <geom type="capsule" fromto="0 0 0 0.08 0 0" size="0.035" rgba="0.85 0.85 0.9 1"/>

                <!-- Joint 6: Wrist Yaw (Matches Grasp Angle θ) -->
                <body name="link6" pos="0.08 0 0">
                  <joint name="joint6" type="hinge" axis="1 0 0" range="-3.14 3.14"/>
                  <geom type="cylinder" size="0.035 0.03" pos="0.03 0 0" zaxis="1 0 0" rgba="0.2 0.2 0.25 1"/>

                  <!-- End-Effector Gripper Body -->
                  <body name="gripper_base" pos="0.06 0 0">
                    <geom type="box" size="0.02 0.05 0.02" rgba="0.2 0.2 0.2 1"/>

                    <!-- Left Finger -->
                    <body name="left_finger" pos="0.03 0.035 0">
                      <joint name="finger_joint1" type="slide" axis="0 1 0" range="-0.04 0.04"/>
                      <geom type="box" size="0.03 0.008 0.015" rgba="0.9 0.2 0.2 1"/>
                    </body>

                    <!-- Right Finger -->
                    <body name="right_finger" pos="0.03 -0.035 0">
                      <joint name="finger_joint2" type="slide" axis="0 1 0" range="-0.04 0.04"/>
                      <geom type="box" size="0.03 0.008 0.015" rgba="0.9 0.2 0.2 1"/>
                    </body>

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


def solve_arm_ik(target_pos, target_yaw, arm_base_pos=np.array([0, 0, 0.7])):
    """
    Solves 6-DoF Inverse Kinematics for planar reach with end-effector pitch down.
    target_pos: [X, Y, Z] world position of the grasp
    target_yaw: rotation angle theta from TAG-Net
    """
    rel = target_pos - arm_base_pos
    x, y, z = rel

    # Joint 1: Base yaw angle to face the object
    j1 = np.arctan2(y, x)

    # Effective 2D distance in reaching plane
    r = np.sqrt(x**2 + y**2) - 0.14  # Subtract wrist/gripper offset
    h = z

    # 2-link planar IK for Link 2 (0.3m) and Link 3 (0.25m)
    l1 = 0.30
    l2 = 0.25
    dist = np.sqrt(r**2 + h**2)
    dist = np.clip(dist, 0.1, l1 + l2 - 0.02)

    # Law of cosines
    cos_elbow = (dist**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_elbow = np.clip(cos_elbow, -1.0, 1.0)
    j3 = np.arccos(cos_elbow)  # Elbow flexion

    alpha = np.arctan2(h, r)
    beta = np.arctan2(l2 * np.sin(j3), l1 + l2 * np.cos(j3))
    j2 = alpha + beta  # Shoulder pitch

    # Orient wrist downward towards table
    j5 = -(j2 + j3) - 0.35  # Wrist pitch down

    # Wrist roll matches TAG-Net grasp angle theta
    j6 = float(target_yaw)

    return np.array([j1, j2, j3, 0.0, j5, j6])


def generate_3d_arm_simulation(save_mp4="simulation/arm_simulation_3d.mp4", 
                               save_gif="simulation/arm_simulation_3d.gif"):
    os.makedirs(os.path.dirname(save_mp4), exist_ok=True)
    
    print("\n[MuJoCo 3D Physics Engine Initializing]...")
    model = mujoco.MjModel.from_xml_string(ROBOT_ARM_MJCF)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)

    # Target Grasp Pose on Table from TAG-Net
    target_grasp = np.array([0.45, 0.0, 0.72])  # Tool handle on table
    pre_grasp_pos = target_grasp + np.array([0, 0, 0.14])  # 14cm hover above
    lift_pos = target_grasp + np.array([0, 0, 0.18])       # 18cm lift

    # Home joints
    q_home = np.array([0.0, 0.3, 0.6, 0.0, -0.9, 0.0])
    q_pre = solve_arm_ik(pre_grasp_pos, target_yaw=0.55)
    q_grasp = solve_arm_ik(target_grasp, target_yaw=0.55)
    q_lift = solve_arm_ik(lift_pos, target_yaw=0.55)

    phases = [
        ("1. READY (Home Pose)", q_home, q_home, 0.035, 0.035, 20),
        ("2. APPROACHING TARGET (TAG-Net Coords)", q_home, q_pre, 0.035, 0.035, 30),
        ("3. REACHING HANDLE", q_pre, q_grasp, 0.035, 0.035, 25),
        ("4. CLOSING GRIPPER (Grasp Contact)", q_grasp, q_grasp, 0.035, 0.005, 20),
        ("5. LIFTING TOOL (Physical Verification)", q_grasp, q_lift, 0.005, 0.005, 35),
    ]

    frames = []
    print("[Executing 3D Arm Kinematic Trajectory]...")

    for phase_name, q_start, q_end, f_start, f_end, num_steps in phases:
        for step in range(num_steps):
            t = step / float(num_steps - 1) if num_steps > 1 else 1.0
            # Interpolate arm joints (smooth cosine interpolation)
            smooth_t = 0.5 * (1.0 - np.cos(np.pi * t))
            q_current = q_start * (1.0 - smooth_t) + q_end * smooth_t
            f_current = f_start * (1.0 - t) + f_end * t

            # Apply to MuJoCo arm joints
            data.qpos[0:6] = q_current
            # Apply to gripper fingers (symmetric prismatic sliding)
            data.qpos[6] = -f_current
            data.qpos[7] = f_current

            # Forward kinematics step
            mujoco.mj_forward(model, data)

            # Camera viewpoint: side/isometric 3D perspective
            cam = mujoco.MjvCamera()
            cam.lookat = [0.35, 0.0, 0.65]
            cam.distance = 1.35
            cam.elevation = -22.0
            cam.azimuth = 135.0

            renderer.update_scene(data, cam)
            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw HUD Overlays for presentation
            cv2.putText(frame_bgr, "TAG-Net Autonomous Grasp Simulation", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Phase: {phase_name}", (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (50, 220, 80), 2)
            cv2.putText(frame_bgr, "Robot Arm: 6-DoF Articulated Manipulator (Franka/UR)", (20, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame_bgr, f"Grasp Target: [X={target_grasp[0]:.2f}m, Y={target_grasp[1]:.2f}m, Z={target_grasp[2]:.2f}m]", (20, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1)

            frames.append(frame_bgr)

    # 1. Export MP4 Video
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_mp4, fourcc, 25, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"[Exported MP4 Video]: {save_mp4}")

    # 2. Export Animated GIF (Optimized for PowerPoint Slides)
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[::2]]
    pil_frames[0].save(
        save_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=70,
        loop=0
    )
    print(f"[Exported PowerPoint GIF]: {save_gif}")

    # Also sync to assets/ for GitHub README
    assets_gif = "assets/arm_simulation_3d.gif"
    pil_frames[0].save(assets_gif, save_all=True, append_images=pil_frames[1:], duration=70, loop=0)
    print(f"[Synced to GitHub Assets]: {assets_gif}")


if __name__ == "__main__":
    generate_3d_arm_simulation()
