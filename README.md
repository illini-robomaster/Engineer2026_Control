# Engineer2026 Control

Control stack for the **Illini RoboMaster** Engineer robot (2026 cycle).

This repository contains ROS 2 Humble tooling for:
- robot description (`URDF`)
- `ros2_control` controller setup
- MoveIt-based multi-DOF motion planning and execution

## Project Structure

- `robotic_arm_v4_urdf/`: robot description package, launch files, and configs
- `scripts/`: setup and run helpers for ROS 2 Humble
- `build/`, `install/`, `log/`: colcon outputs (ignored by git)

## Quick Start

1. Install dependencies:

```bash
zsh scripts/setup_ros2_humble_deps.sh
```

2. Launch control stack only:

```bash
zsh scripts/run_ros2_humble_control.sh
```

3. Launch MoveIt + control (multi-DOF planning/execution):

```bash
zsh scripts/run_ros2_humble_moveit_control.sh
```

## Notes

- This project targets **ROS 2 Humble**.
- For headless runs, disable RViz in launch arguments.
- For package-level migration details, see:
  - `robotic_arm_v4_urdf/README_ros2_humble.md`
