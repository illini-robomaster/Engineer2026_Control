#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_TAG="${ARM_ROS2_IMAGE_TAG:-engineer2026-control:ros2-humble}"
CONTAINER_NAME="${ARM_ROS2_CONTAINER_NAME:-engineer2026-control-humble}"
WITH_X11="auto"
BUILD_IMAGE=0
LIBGL_ALWAYS_SOFTWARE="${ARM_CONTAINER_LIBGL_ALWAYS_SOFTWARE:-1}"
UART_PORTS=(/dev/ttyTHS0)
CAMERA_DEVICES=()
RUNTIME_ARGS=()
ROS_LOCALHOST_ONLY_VALUE="${ROS_LOCALHOST_ONLY:-1}"
CMD=(bash)

usage() {
  cat <<EOF
Usage: bash scripts/run_ros2_humble_arm_container.sh [options]

Options:
  --build-image           Build the Docker image before running.
  --headless              Skip X11 forwarding.
  --x11                   Force X11 forwarding when DISPLAY is available.
  --uart-port <device>    Pass through a serial device, e.g. /dev/ttyUSB0.
  --camera <device>       Pass through a camera device, e.g. /dev/video0.
  --nvidia                Request the NVIDIA container runtime.
  --ros-network           Allow ROS 2 discovery on the LAN (sets ROS_LOCALHOST_ONLY=0).
  --ros-local-only        Force ROS 2 discovery to stay local (sets ROS_LOCALHOST_ONLY=1).
  --name <container>      Override the container name.
  --image-tag <tag>       Override the Docker image tag.
  --cmd <command...>      Run a specific command instead of an interactive shell.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-image)
      BUILD_IMAGE=1
      shift
      ;;
    --headless)
      WITH_X11=0
      shift
      ;;
    --x11)
      WITH_X11=1
      shift
      ;;
    --uart-port)
      UART_PORTS+=("$2")
      shift 2
      ;;
    --camera)
      CAMERA_DEVICES+=("$2")
      shift 2
      ;;
    --nvidia)
      RUNTIME_ARGS+=(--runtime nvidia)
      LIBGL_ALWAYS_SOFTWARE=0
      shift
      ;;
    --ros-network)
      ROS_LOCALHOST_ONLY_VALUE=0
      shift
      ;;
    --ros-local-only)
      ROS_LOCALHOST_ONLY_VALUE=1
      shift
      ;;
    --name)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --cmd)
      shift
      if [[ $# -eq 0 ]]; then
        echo "--cmd requires at least one argument."
        exit 1
      fi
      CMD=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed. Run: bash scripts/setup_ubuntu20_arm_host.sh"
  exit 1
fi

if (( BUILD_IMAGE )) || ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  bash "${SCRIPT_DIR}/build_ros2_humble_arm_container.sh"
fi

DOCKER_ARGS=(
  run
  --rm
  -it
  --name "${CONTAINER_NAME}"
  --network host
  --ipc host
  -e "HOME=/home/$(id -un)"
  -e "USER=$(id -un)"
  -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
  -e "ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY_VALUE}"
  -e "QT_X11_NO_MITSHM=1"
  -e "LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE}"
  --mount "type=bind,source=${REPO_ROOT},target=/workspace/Engineer2026_Control_host"
  -w /workspace/Engineer2026_Control
  -v /etc/localtime:/etc/localtime:ro
)

for group_name in dialout video render plugdev i2c gpio; do
  group_gid="$(getent group "${group_name}" | cut -d: -f3 || true)"
  if [[ -n "${group_gid}" ]]; then
    DOCKER_ARGS+=(--group-add "${group_gid}")
  fi
done

if [[ "${WITH_X11}" != 0 && -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  if command -v xhost >/dev/null 2>&1; then
    xhost +SI:localuser:"$(id -un)" >/dev/null 2>&1 || true
  fi
  DOCKER_ARGS+=(
    -e "DISPLAY=${DISPLAY}"
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro
  )
  if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    DOCKER_ARGS+=(
      -e "XAUTHORITY=${XAUTHORITY}"
      -v "${XAUTHORITY}:${XAUTHORITY}:ro"
    )
  fi
else
  echo "DISPLAY not detected or X11 disabled; starting headless container."
fi

for dev in "${UART_PORTS[@]}" "${CAMERA_DEVICES[@]}"; do
  if [[ ! -e "${dev}" ]]; then
    echo "Device not found: ${dev}"
    exit 1
  fi
  DOCKER_ARGS+=(--device "${dev}:${dev}")
done

docker "${DOCKER_ARGS[@]}" "${RUNTIME_ARGS[@]}" "${IMAGE_TAG}" "${CMD[@]}"
