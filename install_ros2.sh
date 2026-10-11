#!/bin/bash
set -e

echo "=========================================================="
echo ">>> [1/5] Setting up UTF-8 Locales & Prerequisite Tools..."
echo "=========================================================="
apt-get update -y
apt-get install -y locales curl gnupg software-properties-common

locale-gen en_US en_US.UTF-8
update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "=========================================================="
echo ">>> [2/5] Adding ROS 2 Official GPG Key & Repository..."
echo "=========================================================="
add-apt-repository -y universe
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

echo "=========================================================="
echo ">>> [3/5] Updating Apt Package Index for Noble..."
echo "=========================================================="
apt-get update -y

echo "=========================================================="
echo ">>> [4/5] Installing ROS 2 Jazzy Desktop & Build Tools..."
echo "=========================================================="
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ros-jazzy-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-jazzy-ros-gz

echo "=========================================================="
echo ">>> [5/5] Configuring Environment in ~/.bashrc..."
echo "=========================================================="
if [ -d "/home/hareh" ]; then
    if ! grep -q "source /opt/ros/jazzy/setup.bash" /home/hareh/.bashrc 2>/dev/null; then
        echo "source /opt/ros/jazzy/setup.bash" >> /home/hareh/.bashrc
    fi
    chown hareh:hareh /home/hareh/.bashrc
fi

echo "=========================================================="
echo ">>> SUCCESS: ROS 2 Jazzy & Gazebo Harmonic Installed!"
echo "=========================================================="
