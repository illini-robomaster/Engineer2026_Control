#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PATH="$HOME/.local/bin:$PATH"
set +u
source /opt/ros/humble/setup.bash
if [[ -f install/setup.bash ]]; then
  source install/setup.bash
fi
set -u

ros2 launch arm_bringup arm_bringup.launch.py \
  teleop_mode:=ik_direct \
  use_teleop:=true \
  uart_port:=/dev/ttyTHS0 \
  use_real_robot:=true \
  control_orientation:=true \
  log-level:=debug \
  ori_err:=true \
  use_moveit_rviz:=false \
  debug_rx:=false \
  "$@"
