#!/usr/bin/env bash
# Print live joint angles while MoveIt bringup is running.
#
# Shows two kinds of output:
#   [CURRENT]   Joint1=+xx.xx°  ...  Joint6=+xx.xx°   (updated in-place from /joint_states)
#   [PLAN GOAL] ...                                     (printed whenever a MoveIt plan is computed)
#
# The main bringup (run_ros2_humble_moveit_control.sh) must already be running.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="$HOME/.local/bin:$PATH"
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

echo "Joint monitor — drag the arm in MoveIt RViz, angles appear below."
echo "Press Ctrl-C to stop."
echo ""
ros2 run arm_teleop joint_monitor_node
