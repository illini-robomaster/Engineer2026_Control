# Ubuntu 20.04 Setup for the Arm

This repository targets Ubuntu 22.04 with ROS 2 Humble, but this host is Ubuntu 20.04. The most reliable setup on Focal is:

1. Keep Ubuntu 20.04 on the host.
2. Run the ROS 2 Humble arm stack inside a Jammy Docker container.
3. Run `arm_vision/` natively on the host in a Python venv.

That avoids an unsupported native Humble binary install on Focal while still letting the host talk to cameras, serial devices, and the containerized ROS stack.

## 1. Install host prerequisites

Run:

```bash
bash scripts/setup_ubuntu20_arm_host.sh
```

This installs:

- Docker
- Python venv tooling
- system OpenCV / NumPy / SciPy / PyYAML for `arm_vision`
- common camera / X11 / USB helper packages

If the script adds you to the `docker` group, log out and back in before using Docker.

## 2. Build the ROS 2 Humble image

Run:

```bash
bash scripts/build_ros2_humble_arm_container.sh
```

The image contains Ubuntu 22.04 + ROS 2 Humble + MoveIt2 + `ros2_control` + the Python dependencies used by the ROS packages in this repo.

## 3. Start a ROS 2 Humble shell for the arm stack

Headless:

```bash
bash scripts/run_ros2_humble_arm_container.sh --headless
```

With an attached serial device:

```bash
bash scripts/run_ros2_humble_arm_container.sh --uart-port /dev/ttyUSB0
```

Inside the container, build the workspace:

```bash
colcon build --base-paths robotic_arm_v4_urdf arm_teleop arm_hardware arm_bringup
source install/setup.bash
```

Or use the existing helper:

```bash
bash scripts/run_ros2_humble_moveit_control.sh
```

Examples from inside the container:

Simulation only:

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_teleop:=false
```

Real robot:

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true uart_port:=/dev/ttyUSB0
```

## 4. Set up the host-side vision client

Run:

```bash
bash scripts/setup_arm_vision_venv.sh
```

This creates `arm_vision/.venv` and reuses the host's system OpenCV when available. That is especially useful on Jetson-based Ubuntu 20.04 systems where vendor OpenCV is already present.

Activate it:

```bash
source arm_vision/.venv/bin/activate
```

Start the vision client:

```bash
python arm_vision/main.py run --host 127.0.0.1 --show
```

If ROS is running on another machine, replace `127.0.0.1` with that machine's IP.

Optional SpaceMouse support:

```bash
bash scripts/setup_arm_vision_venv.sh --with-spacemouse
```

## Jetson notes

- This host reports a Jetson Orin Nano environment, so the default path is to keep camera access on the host and avoid depending on GPU-enabled Docker just to run `arm_vision`.
- The container script defaults to software GL. If you later want to experiment with NVIDIA container runtime, use:

```bash
bash scripts/run_ros2_humble_arm_container.sh --nvidia
```

## Troubleshooting

- No GUI: run the container with `--headless`.
- No Docker access after install: re-log into your shell session so the new `docker` group membership applies.
- No serial device: verify the device path and pass it with `--uart-port`.
- No camera in `arm_vision`: check `/dev/video*` on the host first; the vision client is meant to run on the host, not in the container.
