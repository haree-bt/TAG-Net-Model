# How to Open and Animate TAG-Net in Autodesk Fusion 360

## Step 1: Open the 3D CAD Model
1. Open **Autodesk Fusion 360**.
2. Click **File -> Open... -> Open from my computer**.
3. Select `robot_arm_scene.obj` from:
   `tag_grasp_net/simulation/cad_exports/robot_arm_scene.obj`
4. The complete assembly with the Table, Target Tool Handle, Robot Links, and Gripper will load directly into your 3D workspace.

## Step 2: Render in Photorealistic Quality (For Your Presentation)
1. Switch from the **Design** workspace to the **Render** workspace (top-left dropdown).
2. Choose a lighting environment (e.g., *Photobooth* or *Warm Light*).
3. Click **In-Canvas Render** or **Render** to produce high-resolution, photorealistic slides.

## Step 3: Animate the Arm Motion Study
1. Open `trajectory_fusion360.csv`.
2. In Fusion 360, under the **Assemble** menu, click **Motion Study**.
3. Map the Joint rotation angles from the CSV to your timeline:
   - **Joint 1 (Base):** Revolute Joint around Z
   - **Joint 2 (Shoulder):** Revolute Joint around Y
   - **Joint 3 (Elbow):** Revolute Joint around Y
   - **Joint 6 (Wrist):** Matches TAG-Net grasp angle
   - **Gripper:** Slide Joint (0mm to 35mm)
4. Play the motion study to watch the arm reach, grasp the handle, and lift the tool!
