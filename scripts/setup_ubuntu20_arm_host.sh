#!/usr/bin/env bash
set -euo pipefail

if [ -r /etc/os-release ]; then
  . /etc/os-release
  if [ "${VERSION_ID:-}" != "20.04" ]; then
    echo "Warning: this helper was written for Ubuntu 20.04, but detected ${PRETTY_NAME:-unknown}."
  fi
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required for host package installation."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  curl \
  git \
  gnupg2 \
  lsb-release \
  software-properties-common \
  docker.io \
  python3-dev \
  python3-opencv \
  python3-pip \
  python3-venv \
  python3-numpy \
  python3-scipy \
  python3-yaml \
  xauth \
  x11-xserver-utils \
  mesa-utils \
  usbutils \
  v4l-utils \
  libhidapi-dev \
  libudev-dev \
  libusb-1.0-0-dev

sudo systemctl enable --now docker

if getent group docker >/dev/null 2>&1; then
  sudo usermod -aG docker "$USER"
fi

cat <<EOF

Host prerequisites installed.

Next steps:
  1. Log out and back in so docker-group membership applies.
  2. Build the ROS 2 Humble image:
       bash scripts/build_ros2_humble_arm_container.sh
  3. Start a container shell:
       bash scripts/run_ros2_humble_arm_container.sh --headless
  4. Set up the host-side vision venv:
       bash scripts/setup_arm_vision_venv.sh

EOF
