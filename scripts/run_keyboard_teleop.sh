#!/usr/bin/env bash
# Run keyboard teleop in this terminal.
# The main bringup (run_ros2_humble_moveit_control.sh) must already be running.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

echo "Keyboard teleop — run in a dedicated terminal while bringup is active."
ros2 run arm_teleop keyboard_teleop_node
