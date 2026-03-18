#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_TAG="${ARM_ROS2_IMAGE_TAG:-engineer2026-control:ros2-humble}"

group_gid() {
  getent group "$1" | cut -d: -f3 || true
}

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed. Run: bash scripts/setup_ubuntu20_arm_host.sh"
  exit 1
fi

docker build \
  --build-arg USERNAME="$(id -un)" \
  --build-arg UID="$(id -u)" \
  --build-arg GID="$(id -g)" \
  --build-arg HOST_DIALOUT_GID="$(group_gid dialout)" \
  --build-arg HOST_VIDEO_GID="$(group_gid video)" \
  --build-arg HOST_RENDER_GID="$(group_gid render)" \
  --build-arg HOST_PLUGDEV_GID="$(group_gid plugdev)" \
  --build-arg HOST_I2C_GID="$(group_gid i2c)" \
  --build-arg HOST_GPIO_GID="$(group_gid gpio)" \
  -t "${IMAGE_TAG}" \
  -f "${REPO_ROOT}/docker/ros2_humble_arm/Dockerfile" \
  "${REPO_ROOT}"
