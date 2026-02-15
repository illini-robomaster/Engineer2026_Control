#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  ros-humble-controller-manager \
  ros-humble-joint-state-publisher \
  ros-humble-joint-state-publisher-gui \
  ros-humble-moveit \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-xacro

# Some mirrors/images do not publish gazebo_ros package names for Humble.
# Install Gazebo-related ROS deps only when available.
GAZEBO_OPTIONAL_PKGS=(
  ros-humble-gazebo-ros
  ros-humble-gazebo-ros-pkgs
  ros-humble-gazebo-plugins
  ros-humble-gazebo-msgs
  ros-humble-gazebo-dev
)

available_gazebo_pkgs=()
skipped_gazebo_pkgs=()
for pkg in "${GAZEBO_OPTIONAL_PKGS[@]}"; do
  # Ensure package exists and is actually installable with current apt sources.
  if apt-cache show "$pkg" >/dev/null 2>&1 && apt-get -s install "$pkg" >/dev/null 2>&1; then
    available_gazebo_pkgs+=("$pkg")
  else
    skipped_gazebo_pkgs+=("$pkg")
  fi
done

if (( ${#available_gazebo_pkgs[@]} > 0 )); then
  sudo apt-get install -y "${available_gazebo_pkgs[@]}"
else
  echo "Warning: No Gazebo ROS package candidates were found in apt for Humble."
fi

if (( ${#skipped_gazebo_pkgs[@]} > 0 )); then
  echo "Skipped unavailable or unsatisfied Gazebo packages: ${skipped_gazebo_pkgs[*]}"
fi

# Optional if colcon is not installed system-wide.
if ! command -v colcon >/dev/null 2>&1; then
  python3 -m pip install --user colcon-common-extensions
  echo "Installed colcon to ~/.local/bin"
fi

echo "Dependency setup complete."
