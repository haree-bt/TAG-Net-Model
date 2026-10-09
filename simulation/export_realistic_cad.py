"""
High-Fidelity 3D CAD Exporter for Autodesk Fusion 360:
Generates smooth cylindrical robotic arm links, rotary motor joint hubs,
an industrial parallel-jaw gripper, and a 3D tool with handle on the table.
"""

import os
import sys
import numpy as np

CAD_DIR = os.path.join(os.path.dirname(__file__), "cad_exports")
os.makedirs(CAD_DIR, exist_ok=True)
OBJ_PATH = os.path.join(CAD_DIR, "realistic_robot_arm.obj")
MTL_PATH = os.path.join(CAD_DIR, "realistic_robot_arm.mtl")


class OBJBuilder:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.materials = []
        self.v_count = 0

    def add_cylinder(self, p1, p2, radius, mtl_name, segments=32):
        """Generates a smooth 3D cylinder between p1 and p2."""
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        v_axis = p2 - p1
        height = np.linalg.norm(v_axis)
        if height < 1e-6:
            return

        z_unit = v_axis / height
        # Pick orthogonal vector
        if abs(z_unit[0]) < 0.9:
            ortho = np.cross(z_unit, np.array([1, 0, 0]))
        else:
            ortho = np.cross(z_unit, np.array([0, 1, 0]))
        x_unit = ortho / np.linalg.norm(ortho)
        y_unit = np.cross(z_unit, x_unit)

        theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        circle_x = radius * np.cos(theta)
        circle_y = radius * np.sin(theta)

        # Vertices: bottom ring then top ring
        start_v = self.v_count + 1
        for i in range(segments):
            pt = p1 + circle_x[i] * x_unit + circle_y[i] * y_unit
            self.vertices.append(pt)
        for i in range(segments):
            pt = p2 + circle_x[i] * x_unit + circle_y[i] * y_unit
            self.vertices.append(pt)

        # Bottom center & top center
        p1_idx = start_v + 2 * segments
        p2_idx = start_v + 2 * segments + 1
        self.vertices.append(p1)
        self.vertices.append(p2)
        self.v_count += 2 * segments + 2

        self.faces.append(f"usemtl {mtl_name}")

        # Side faces
        for i in range(segments):
            i_next = (i + 1) % segments
            v1 = start_v + i
            v2 = start_v + i_next
            v3 = start_v + segments + i_next
            v4 = start_v + segments + i
            self.faces.append(f"f {v1} {v2} {v3} {v4}")

        # Cap faces
        for i in range(segments):
            i_next = (i + 1) % segments
            # Bottom cap
            self.faces.append(f"f {p1_idx} {start_v + i_next} {start_v + i}")
            # Top cap
            self.faces.append(f"f {p2_idx} {start_v + segments + i} {start_v + segments + i_next}")

    def add_box(self, center, size, mtl_name):
        """Generates a 3D box centered at center with dimensions size."""
        cx, cy, cz = center
        sx, sy, sz = size
        dx, dy, dz = sx / 2.0, sy / 2.0, sz / 2.0
        
        start_v = self.v_count + 1
        verts = [
            (cx - dx, cy - dy, cz - dz),
            (cx + dx, cy - dy, cz - dz),
            (cx + dx, cy + dy, cz - dz),
            (cx - dx, cy + dy, cz - dz),
            (cx - dx, cy - dy, cz + dz),
            (cx + dx, cy - dy, cz + dz),
            (cx + dx, cy + dy, cz + dz),
            (cx - dx, cy + dy, cz + dz)
        ]
        self.vertices.extend(verts)
        self.v_count += 8

        box_faces = [
            (1, 2, 3, 4), # bottom
            (5, 8, 7, 6), # top
            (1, 5, 6, 2), # front
            (2, 6, 7, 3), # right
            (3, 7, 8, 4), # back
            (4, 8, 5, 1)  # left
        ]
        self.faces.append(f"usemtl {mtl_name}")
        for f in box_faces:
            self.faces.append(f"f {f[0] + start_v - 1} {f[1] + start_v - 1} {f[2] + start_v - 1} {f[3] + start_v - 1}")

    def add_torus_handle(self, center, major_r, minor_r, mtl_name, segments=24):
        """Generates a curved half-torus for a mug handle."""
        cx, cy, cz = center
        theta = np.linspace(-np.pi/2, np.pi/2, segments)
        phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)

        pts_rings = []
        for t in theta:
            ring = []
            center_ring = np.array([cx + major_r * np.cos(t), cy, cz + major_r * np.sin(t)])
            dir_out = np.array([np.cos(t), 0, np.sin(t)])
            dir_up = np.array([-np.sin(t), 0, np.cos(t)])
            dir_y = np.array([0, 1, 0])
            for p in phi:
                pt = center_ring + minor_r * np.cos(p) * dir_out + minor_r * np.sin(p) * dir_y
                ring.append(pt)
            pts_rings.append(ring)

        start_v = self.v_count + 1
        for ring in pts_rings:
            for pt in ring:
                self.vertices.append(pt)
                self.v_count += 1

        self.faces.append(f"usemtl {mtl_name}")
        for i in range(len(pts_rings) - 1):
            for j in range(12):
                j_next = (j + 1) % 12
                v1 = start_v + i * 12 + j
                v2 = start_v + i * 12 + j_next
                v3 = start_v + (i + 1) * 12 + j_next
                v4 = start_v + (i + 1) * 12 + j
                self.faces.append(f"f {v1} {v2} {v3} {v4}")

    def save(self, obj_path, mtl_filename="realistic_robot_arm.mtl"):
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write(f"# Realistic Industrial Robot Arm for Autodesk Fusion 360\n")
            f.write(f"mtllib {mtl_filename}\n\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            f.write("\n")
            for face in self.faces:
                f.write(f"{face}\n")


def export_realistic_cad():
    builder = OBJBuilder()

    # 1. Work Table (Granite / Matte Grey)
    builder.add_box((0.45, 0.0, 0.35), (0.55, 0.75, 0.70), "Matte_Table")
    # Table top bevel trim
    builder.add_box((0.45, 0.0, 0.705), (0.57, 0.77, 0.01), "Dark_Joints")

    # 2. Target Object on Table: Real Mug with Curved Functional Handle
    # Mug cylindrical body
    builder.add_cylinder((0.45, 0.0, 0.71), (0.45, 0.0, 0.82), radius=0.045, mtl_name="Ceramic_White")
    # Hollow inner cavity
    builder.add_cylinder((0.45, 0.0, 0.72), (0.45, 0.0, 0.821), radius=0.038, mtl_name="Dark_Joints")
    # Curved functional handle (affordance zone = TAG-Net target!)
    builder.add_torus_handle((0.45 - 0.045, 0.0, 0.765), major_r=0.032, minor_r=0.007, mtl_name="Handle_Green")

    # 3. Robot Arm Base Turntable
    builder.add_cylinder((0.0, 0.0, 0.70), (0.0, 0.0, 0.76), radius=0.09, mtl_name="Dark_Joints")
    builder.add_cylinder((0.0, 0.0, 0.76), (0.0, 0.0, 0.82), radius=0.075, mtl_name="Metallic_Arm")

    # 4. Joint 1 & Shoulder Link
    builder.add_cylinder((0.0, 0.0, 0.82), (0.0, 0.0, 0.98), radius=0.065, mtl_name="Metallic_Arm")
    # Shoulder rotary hub (cross cylinder)
    builder.add_cylinder((0.0, -0.06, 0.98), (0.0, 0.06, 0.98), radius=0.06, mtl_name="Dark_Joints")

    # 5. Upper Arm (Link 2) - Angled reaching towards table
    p_shoulder = np.array([0.0, 0.0, 0.98])
    p_elbow = np.array([0.22, 0.0, 1.15])
    builder.add_cylinder(p_shoulder, p_elbow, radius=0.05, mtl_name="Metallic_Arm")

    # 6. Elbow Joint Rotary Hub
    builder.add_cylinder(p_elbow + np.array([0, -0.05, 0]), p_elbow + np.array([0, 0.05, 0]), radius=0.052, mtl_name="Dark_Joints")

    # 7. Forearm Link (Link 3) - Angled down towards handle
    p_wrist = np.array([0.38, 0.0, 0.90])
    builder.add_cylinder(p_elbow, p_wrist, radius=0.042, mtl_name="Metallic_Arm")

    # 8. Wrist Rotary Joint Hub
    builder.add_cylinder(p_wrist + np.array([0, -0.035, 0]), p_wrist + np.array([0, 0.035, 0]), radius=0.042, mtl_name="Dark_Joints")

    # 9. Wrist Roll / Flange (Link 6)
    p_flange = np.array([0.43, 0.0, 0.84])
    builder.add_cylinder(p_wrist, p_flange, radius=0.035, mtl_name="Metallic_Arm")

    # 10. Gripper Base
    p_grip_center = np.array([0.43, 0.0, 0.81])
    builder.add_box(p_grip_center, (0.04, 0.10, 0.04), "Dark_Joints")

    # 11. Parallel-Jaw Gripper Fingers (Closing around the green mug handle!)
    f_left = np.array([0.43, 0.025, 0.765])
    f_right = np.array([0.43, -0.025, 0.765])
    builder.add_box(f_left, (0.025, 0.012, 0.07), "Anodized_Red")
    builder.add_box(f_right, (0.025, 0.012, 0.07), "Anodized_Red")

    # Rubber Gripper Contact Pads (Inner side)
    builder.add_box(f_left - np.array([0, 0.005, 0]), (0.022, 0.004, 0.06), "Dark_Joints")
    builder.add_box(f_right + np.array([0, 0.005, 0]), (0.022, 0.004, 0.06), "Dark_Joints")

    # Save OBJ
    builder.save(OBJ_PATH)

    # Save MTL (Materials & Shaders)
    with open(MTL_PATH, "w", encoding="utf-8") as f:
        f.write("""# Fusion 360 Material Shaders
newmtl Metallic_Arm
Kd 0.90 0.90 0.94
Ks 0.70 0.70 0.70
Ns 60.0

newmtl Dark_Joints
Kd 0.22 0.22 0.26
Ks 0.35 0.35 0.35
Ns 35.0

newmtl Anodized_Red
Kd 0.85 0.12 0.15
Ks 0.55 0.55 0.55
Ns 45.0

newmtl Handle_Green
Kd 0.15 0.82 0.28
Ks 0.40 0.40 0.40
Ns 30.0

newmtl Ceramic_White
Kd 0.95 0.95 0.95
Ks 0.60 0.60 0.60
Ns 70.0

newmtl Matte_Table
Kd 0.65 0.68 0.72
Ks 0.15 0.15 0.15
Ns 10.0
""")

    print(f"[Done] Exported Realistic CAD Model: {OBJ_PATH}")
    print(f"       -> Companion Shaders: {MTL_PATH}")


if __name__ == "__main__":
    export_realistic_cad()
