#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
set +u
source /opt/ros/humble/setup.zsh
set -u

colcon build --base-paths robotic_arm_v4_urdf --event-handlers console_direct+
set +u
source install/setup.zsh
set -u

# Set use_moveit_rviz:=false for headless runs.
ros2 launch robotic_arm_v4_urdf moveit_control.launch.py use_moveit_rviz:=true
