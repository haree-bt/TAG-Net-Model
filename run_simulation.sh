#!/usr/bin/env bash
# ==============================================================================
# TAG-Net: Task-Aware Robotic Grasping & Dexterous Manipulation Simulation
# Environment: ROS 2 Jazzy Jalisco + Gazebo Harmonic 8.15 on Ubuntu 24.04 (WSL 2)
# Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU
# ==============================================================================

set -e

echo "=========================================================================="
echo "  TAG-Net: Task-Aware Affordance-Gated Robotic Manipulation"
echo "  Middleware: ROS 2 Jazzy | Simulator: Gazebo Harmonic"
echo "=========================================================================="

# 1. Source ROS 2 Core
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    echo "[✓] ROS 2 Jazzy environment sourced."
else
    echo "[✗] ERROR: ROS 2 Jazzy not found in /opt/ros/jazzy."
    exit 1
fi

# 2. Source Workspace
WS_DIR="$HOME/ros2_ws_hare"
if [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
    echo "[✓] Workspace sourced from $WS_DIR."
else
    echo "[!] Building workspace packages first..."
    cd "$WS_DIR"
    colcon build --symlink-install --packages-select tag_net_ros
    source "$WS_DIR/install/setup.bash"
fi

# 3. Launch the full ROS 2 + Gazebo Harmonic pipeline
echo "=========================================================================="
echo ">>> Booting Gazebo Harmonic World, ROS Bridge, and TAG-Net Perception Node..."
echo "=========================================================================="

ros2 launch tag_net_ros simulation.launch.py
