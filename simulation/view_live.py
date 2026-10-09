"""
Live 3D Robotic Arm Simulation Viewer:
Opens an interactive desktop GUI window showing the robotic arm
moving its joints, reaching, and lifting the object in real time.
"""

import os
import sys
import cv2
import numpy as np

# Path to the generated 3D simulation video
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "arm_simulation_3d.mp4")


def play_live_simulation():
    if not os.path.exists(VIDEO_PATH):
        print("[Notice] Video not found. Generating simulation first...")
        import full_arm_mujoco
        full_arm_mujoco.generate_3d_arm_simulation()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {VIDEO_PATH}")
        return

    window_name = "TAG-Net 3D Robotic Arm Simulation (Press Q to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("\n" + "=" * 65)
    print("  PLAYING 3D ROBOTIC ARM SIMULATION WINDOW")
    print("  -> Press 'Q' or 'ESC' in the window to close it.")
    print("  -> Press 'SPACE' to pause / resume.")
    print("=" * 65 + "\n")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            # Loop video continuously
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow(window_name, frame)

        key = cv2.waitKey(40) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):  # SPACEBAR to pause
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("Simulation viewer closed.")


if __name__ == "__main__":
    play_live_simulation()
