#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
WORKSPACE_IMAGE_PATH="${WORKSPACE_IMAGE_PATH:-/opt/Engineer2026_Control}"
WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-/workspace/Engineer2026_Control_host}"
WORKSPACE_ACTIVE_PATH="${WORKSPACE_ACTIVE_PATH:-/workspace/Engineer2026_Control}"

if [ -f /opt/ros/humble/setup.bash ]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  set -u
fi

selected_workspace="${WORKSPACE_IMAGE_PATH}"
if [ -f "${WORKSPACE_HOST_PATH}/scripts/run_ros2_humble_arm_container.sh" ]; then
  selected_workspace="${WORKSPACE_HOST_PATH}"
fi

if [ ! -L "${WORKSPACE_ACTIVE_PATH}" ] || [ "$(readlink "${WORKSPACE_ACTIVE_PATH}")" != "${selected_workspace}" ]; then
  rm -rf "${WORKSPACE_ACTIVE_PATH}"
  ln -s "${selected_workspace}" "${WORKSPACE_ACTIVE_PATH}"
fi

cd "${WORKSPACE_ACTIVE_PATH}"

exec "$@"
