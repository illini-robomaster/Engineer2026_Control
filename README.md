# Engineer2026 Control

Control stack for the **Illini RoboMaster** Engineer robot (2026 cycle).

This repository contains ROS 2 Humble packages for a **6-DOF robotic arm**: URDF description, ros2\_control setup, MoveIt2 motion planning, and teleoperation via AprilTag cube or keyboard.  It targets an OrangePi SBC communicating with an STM32 motor controller over UART.

The arm is teleoperated using an AprilTag cube held by the operator.  A **standalone Python client** (`arm_vision/`) handles all camera work — calibration, AprilTag detection, and pose estimation — and streams the target end-effector pose to the ROS stack over a plain TCP socket.  No ROS is required on the client machine.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  arm_vision/  (standalone Python — no ROS)                       │
│                                                                  │
│  webcam → AprilTag detection → cube pose → workspace mapping     │
│                                         ↓                        │
│                            6D EE target pose (JSON over TCP)     │
└─────────────────────────────┬────────────────────────────────────┘
                              │ TCP socket (default port 9999)
┌─────────────────────────────▼────────────────────────────────────┐
│  ROS 2 Humble  (runs on OrangePi)                                │
│                                                                  │
│  socket_teleop_node   receives target pose, P-controller         │
│  keyboard_teleop_node ──► /servo_node/delta_twist_cmds           │
│                           /servo_node/delta_joint_cmds           │
│  servo_node (MoveIt2) ──► /arm_controller/joint_trajectory       │
│  move_group                                                      │
│  ros2_control         ──► mock_components  OR  UART → STM32      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

| Path | Description |
|---|---|
| `robotic_arm_v4_urdf/` | Robot URDF, ros2\_control config, MoveIt2 config, core launch files |
| `arm_teleop/` | `socket_teleop_node` (TCP receiver + P-controller) and `keyboard_teleop_node` |
| `arm_hardware/` | UART bridge node that forwards joint commands to the STM32 |
| `arm_bringup/` | Top-level launch that ties all ROS packages together |
| `arm_vision/` | Standalone Python client: camera cal, cube cal, detection, socket sender |
| `scripts/` | Shell helpers for setup, build, and launch |

---

## Prerequisites

**OS:** Ubuntu 22.04 (Jammy) — ROS 2 Humble must already be installed.

Install all ROS and Python dependencies:

```bash
bash scripts/setup_ros2_humble_deps.sh
```

Install standalone client dependencies (no ROS needed):

```bash
cd arm_vision
pip install -r requirements.txt
```

---

## Build (ROS)

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths robotic_arm_v4_urdf arm_teleop arm_hardware arm_bringup
source install/setup.bash
```

Or use the helper script (builds then launches):

```bash
bash scripts/run_ros2_humble_moveit_control.sh
```

---

## Running

### 1 — Start the ROS stack

```bash
source install/setup.bash
ros2 launch arm_bringup arm_bringup.launch.py
```

With real robot:

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true uart_port:=/dev/ttyS4
```

Headless (no RViz):

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_moveit_rviz:=false
```

### 2 — Start the arm\_vision client (separate machine or terminal)

```bash
cd arm_vision
python main.py run --host <ros-host-ip> --show
```

The client opens the webcam, detects the AprilTag cube, maps its pose to a
target EE position, and streams it to the ROS `socket_teleop_node` at 30 Hz.

### 3 — Homing (real robot only)

Homing moves each joint to 0 rad in a safe sequential order so the arm starts from a known configuration.  The default order is **Joint2 → Joint1 → Joint3 → Joint4 → Joint5 → Joint6** (shoulder tucks in first, then base rotation returns, then distal joints fold).

Run automatically at startup by adding `run_homing:=true` to the bringup:

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true uart_port:=/dev/ttyS4 run_homing:=true
```

Or run it as a standalone node while the stack is already running:

```bash
source install/setup.bash
ros2 run arm_hardware homing_node
```

Key parameters (override with `--ros-args -p name:=value`):

| Parameter | Default | Description |
|---|---|---|
| `homing_order` | `[Joint2,Joint1,Joint3,Joint4,Joint5,Joint6]` | Joints to zero, in order |
| `joint_speed_rad_s` | `0.3` | Max speed used to compute each step duration |
| `min_duration_s` | `2.0` | Minimum time allocated per joint step |
| `settle_time_s` | `0.5` | Extra settle wait after each step |

### 4 — Joint Monitor (observe live joint angles)

The joint monitor prints joint angles to the terminal while the arm moves or a MoveIt plan is computed.  It is a lightweight diagnostic tool — no effect on motion.

**Recommended: dedicated terminal (cleanest output)**

```bash
bash scripts/run_joint_monitor.sh
```

Or inline via bringup:

```bash
ros2 launch arm_bringup arm_bringup.launch.py print_joints:=true
```

Example output:

```
[CURRENT]   Joint1=  +0.00°  Joint2= -45.23°  Joint3= +12.10°  Joint4=  +0.00°  Joint5=  +5.50°  Joint6=  +0.00°
[PLAN GOAL] Joint1=  +0.00°  Joint2=  +0.00°  Joint3=  +0.00°  Joint4=  +0.00°  Joint5=  +0.00°  Joint6=  +0.00°
```

- **`[CURRENT]`** — updates in-place at each `/joint_states` tick (live arm position).
- **`[PLAN GOAL]`** — printed whenever MoveIt computes a plan (drag the interactive marker in RViz and click **Plan**).

### 5 — Keyboard teleoperation (optional, separate terminal)

```bash
source install/setup.bash
ros2 run arm_teleop keyboard_teleop_node
```

Or:

```bash
bash scripts/run_keyboard_teleop.sh
```

---

## Calibration Workflow

All calibration is done from the `arm_vision/` directory, with no ROS running.

### Step 1 — Camera calibration

Print a checkerboard (default 9×6 inner corners, 25 mm squares).  Show it
from multiple angles while pressing **SPACE** to capture frames.  Press **Q**
when done (≥10 captures recommended).

```bash
cd arm_vision
python main.py calibrate-camera
```

Output: `arm_vision/data/camera_calibration.yaml`

### Step 2 — Cube calibration (5 faces)

For each of the 5 AprilTag faces on the cube:

1. Point that face toward the camera.
2. Press **SPACE** to capture.
3. Enter the face label when prompted (`+X`, `-X`, `+Y`, `-Y`, `+Z`).

```bash
python main.py calibrate-cube --side-length 0.06
```

Output: `arm_vision/data/cube_config.yaml`

### Step 3 — Workspace mapping

Edit `arm_vision/config/workspace.yaml` to match your physical setup:

| Field | Description |
|---|---|
| `cam_origin` | Neutral cube position in camera frame (m) — where the arm should sit at rest |
| `ee_origin` | EE position in `base_link` at neutral — read from `/tf` at home pose |
| `scale` | Amplification: how much robot moves per unit of cube movement |
| `cam_to_robot` | 3×3 rotation matrix mapping camera axes → robot base\_link axes |

---

## Launch Arguments (`arm_bringup.launch.py`)

| Argument | Default | Description |
|---|---|---|
| `use_real_robot` | `false` | `true` to start the UART bridge to the STM32 |
| `uart_port` | `/dev/ttyS3` | Serial device for the STM32 connection |
| `baud_rate` | `115200` | UART baud rate |
| `socket_host` | `0.0.0.0` | TCP bind address for the pose socket |
| `socket_port` | `9999` | TCP bind port for the pose socket |
| `use_moveit_rviz` | `true` | Show MoveIt2 RViz panel |
| `run_homing` | `false` | Run sequential homing on startup (real robot only) |
| `print_joints` | `false` | Print live joint angles + MoveIt plan goals to terminal |

---

## arm\_vision CLI Reference

```
python main.py calibrate-camera  [--device N] [--cols 9] [--rows 6]
                                  [--square-size 0.025] [--output FILE]

python main.py calibrate-cube    [--device N] [--tag-family tag36h11]
                                  [--side-length 0.06] [--camera-cal FILE]
                                  [--output FILE]

python main.py run               [--device N] [--host IP] [--port 9999]
                                  [--camera-cal FILE] [--cube-config FILE]
                                  [--workspace-config FILE] [--show]
```

---

## Hardware

- **SBC:** OrangePi (RK3588, ARM64)
- **MCU:** STM32 connected via UART (`/dev/ttyS3`)
- **Camera:** USB webcam (any OpenCV-compatible device)
- **Arm:** 6-DOF (Joint1 – Joint6)
