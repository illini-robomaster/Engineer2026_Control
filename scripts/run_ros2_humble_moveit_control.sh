#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
set +u
source /opt/ros/humble/setup.bash
set -u

# Build all packages: robot model, teleop, hardware, bringup
colcon build \
  --base-paths \
    robotic_arm_v4_urdf \
    arm_teleop \
    arm_hardware \
    arm_bringup \
  --event-handlers console_direct+

set +u
source install/setup.bash
set -u

# ── Launch options ────────────────────────────────────────────────────────────
# Simulation only (default):
#   ros2 launch arm_bringup arm_bringup.launch.py
#
# With real robot (set UART_PORT to your actual device):
#   ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true uart_port:=/dev/ttyS3
#
# With keyboard debug (launches in xterm):
#   ros2 launch arm_bringup arm_bringup.launch.py use_keyboard:=true
#
# Headless (no RViz):
#   ros2 launch arm_bringup arm_bringup.launch.py use_moveit_rviz:=false
# ─────────────────────────────────────────────────────────────────────────────

ros2 launch arm_bringup arm_bringup.launch.py "$@"
