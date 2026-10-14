"""
TAG-Net Real-Time Interactive 3D Robotic Arm Simulation in MuJoCo
===================================================================
Simulates autonomous 6-DoF robotic manipulation driven by TAG-Net AGAM
(Affordance-Gated Attention Mechanism) across:
  Objects: knife, mug, screwdriver (tool)
  Tasks: pick_and_place, handover, work_with_it

Usage:
    python simulation/realtime_mujoco_simulation.py --tool knife --task handover
    python simulation/realtime_mujoco_simulation.py --tool mug --task pick_and_place
    python simulation/realtime_mujoco_simulation.py --tool tool --task work_with_it
"""

import sys
import os
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from simulation.task_simulation import (
    launch_task_simulation_desktop,
    get_agam_task_grasp,
    build_task_scene_xml,
    generate_task_phases,
    solve_ik_top_down
)

# Backwards compatibility aliases
get_agam_grasp_prediction = get_agam_task_grasp
build_mujoco_scene_xml = build_task_scene_xml
run_interactive_simulation = launch_task_simulation_desktop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAG-Net Interactive MuJoCo Real-Time Simulation")
    parser.add_argument("--tool", type=str, default="knife", choices=["knife", "mug", "screwdriver", "tool"],
                        help="Target tool to manipulate")
    parser.add_argument("--task", type=str, default="handover", choices=["pick_and_place", "handover", "work_with_it"],
                        help="Task execution intention")
    args = parser.parse_args()

    chosen_tool = "screwdriver" if args.tool.lower() == "tool" else args.tool.lower()
    launch_task_simulation_desktop(chosen_tool, args.task)
