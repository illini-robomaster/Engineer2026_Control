# Engineer2026 Control

Control stack for the **Illini RoboMaster** Engineer robot (2026 cycle).

This repository contains ROS 2 Humble packages for a **6-DOF robotic arm**: URDF description, ros2\_control setup, MoveIt2 motion planning, and teleoperation via AprilTag cube or keyboard.  It targets an OrangePi SBC communicating with an MCU (ESP32/STM32) over UART.

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
│  ros2_control         ──► mock_components  OR  uart_bridge_node → MCU   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

| Path | Description |
|---|---|
| `robotic_arm_v4_urdf/` | Robot URDF, ros2\_control config, MoveIt2 config, core launch files |
| `arm_teleop/` | `socket_teleop_node` (TCP receiver + P-controller) and `keyboard_teleop_node` |
| `arm_hardware/` | UART bridge node, homing node — communicate with the MCU over UART |
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

**Simulation only — Plan+Execute via RViz (no hardware, no vision):**

```bash
source install/setup.bash
ros2 launch arm_bringup arm_bringup.launch.py use_teleop:=false
```

**Real robot — Plan+Execute via RViz (no vision pipeline):**

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true use_teleop:=false uart_port:=/dev/yourdevice
```

**Real robot + arm\_vision socket teleop (full system):**

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true uart_port:=/dev/yourdevice
```

Headless (no RViz):

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_moveit_rviz:=false use_teleop:=false
```

### 2 — Start the arm\_vision client (separate machine or terminal)

```bash
cd arm_vision
python main.py run --host <ros-host-ip> --show
```

The client opens the webcam, detects the AprilTag cube, maps its pose to a
target EE position, and streams it to the ROS `socket_teleop_node` at 30 Hz.

### 3 — Homing (real robot only)

Homing moves each joint to 0 rad in a collision-aware order.  The group sequence is chosen **dynamically** based on Joint2's current angle at startup:

| J2 angle | Sequence | Rationale |
|----------|----------|-----------|
| J2 ≥ 0° (forward) | `[J2] → [J3,J4,J5,J6] → [J1]` | Retract shoulder first to clear space |
| J2 < 0° (back) | `[J3] → [J2] → [J4,J5,J6] → [J1]` | Fold elbow first to avoid body collision |

Run automatically at startup by adding `run_homing:=true` to the bringup:

```bash
ros2 launch arm_bringup arm_bringup.launch.py use_real_robot:=true uart_port:=/dev/yourdevice run_homing:=true
```

Or run it as a standalone node while the stack is already running:

```bash
source install/setup.bash
ros2 run arm_hardware homing_node
```

Key parameters (override with `--ros-args -p name:=value`):

| Parameter | Default | Description |
|---|---|---|
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

### 5 — View-only mode (real arm angles in RViz, no motion commands)

Shows live encoder feedback from the MCU as a moving robot model in RViz.  No motion commands are ever sent to the hardware in this mode.

```bash
bash scripts/run_view_only.sh --port /dev/yourdevice
# with raw-frame debug logging:
bash scripts/run_view_only.sh --port /dev/yourdevice --debug
```

If `--port` is not given the script prints available `/dev/tty*` devices and exits.

This starts only: `robot_state_publisher` + `uart_bridge_node` (RX only) + RViz(`display.rviz`).

### 6 — Keyboard teleoperation (optional, separate terminal)

```bash
source install/setup.bash
ros2 run arm_teleop keyboard_teleop_node
```

Or:

```bash
bash scripts/run_keyboard_teleop.sh
```

### 7 — MCU Emulator (test without real hardware)

The emulator creates a virtual serial port pair so you can run and test the full ROS stack — homing, Plan+Execute, and vision teleop — without the real robot connected.  It simulates the MCU's motor PID: positions approach the commanded angles with a configurable time constant.

**Terminal 1 — start the emulator:**

```bash
# All joints at zero (default):
python3 scripts/mcu_emulator.py

# Start at non-zero positions to test homing:
python3 scripts/mcu_emulator.py --start 30,-20,15,-10,50,5

# Slower motor response (default tau=0.2s):
python3 scripts/mcu_emulator.py --start 30,-20,15,-10,50,5 --tau 0.5
```

The emulator prints the virtual serial port path:

```
╔══════════════════════════════════════════════════════════════╗
║  MCU Emulator Ready                                        ║
║  Connect the bridge with:                                   ║
║    uart_port:=/dev/pts/7                                   ║
╚══════════════════════════════════════════════════════════════╝
```

**Terminal 2 — launch the full ROS stack with that port:**

```bash
# Plan+Execute from RViz only (no vision):
ros2 launch arm_bringup arm_bringup.launch.py \
  use_real_robot:=true uart_port:=/dev/pts/7 use_teleop:=false

# With homing on startup:
ros2 launch arm_bringup arm_bringup.launch.py \
  use_real_robot:=true uart_port:=/dev/pts/7 use_teleop:=false run_homing:=true

# Full pipeline — vision teleop → MoveIt → emulated arm:
ros2 launch arm_bringup arm_bringup.launch.py \
  use_real_robot:=true uart_port:=/dev/pts/7
# then in a third terminal:
python arm_vision/main.py run --host localhost --show
```

**Emulator output legend:**

| Color | Tag | Meaning |
|-------|-----|---------|
| Cyan | `[MCU RX TRAJ]` | Command frame received from bridge during a trajectory |
| Cyan | `[MCU RX HOLD]` | Command frame received during idle-hold |
| Yellow | `[MCU TX MOVE]` | Encoder feedback sent back — motors still converging |
| Yellow | `[MCU TX HOLD]` | Encoder feedback sent back — motors at target |
| Red | `[MCU WATCHDOG]` | No command received within timeout — TX gap detected |

**Emulator options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--start J1,J2,J3,J4,J5,J6` | `0,0,0,0,0,0` | Initial motor positions in degrees |
| `--rate Hz` | `20` | Feedback send rate (should match `send_rate_hz`) |
| `--tau s` | `0.2` | Motor time constant — how fast position approaches target |
| `--watchdog s` | `1.0` | Warn if no command received for this long |
| `--quiet` | off | Only print changes > 0.5° and warnings |

### 8 — Pipeline test (mock arm\_vision sender)

Sends fake target poses over TCP without a camera, to test the full post-vision pipeline:

```bash
# Interactive mode — type poses manually:
python scripts/test_pose_sender.py

# Demo mode — loops through a preset waypoint sequence:
python scripts/test_pose_sender.py --demo

# Hold a fixed pose:
python scripts/test_pose_sender.py --hold 0.3 0.0 0.2
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
| `use_real_robot` | `false` | `true` to start the UART bridge to the MCU |
| `uart_port` | *(required when `use_real_robot:=true`)* | Serial device for the MCU connection, e.g. `/dev/ttyUSB0` |
| `baud_rate` | `115200` | UART baud rate |
| `use_teleop` | `true` | `false` to skip socket\_teleop and use MoveIt Plan+Execute directly |
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

## UART Protocol & Joint Mapping

### Physical layer

- **Device:** configured via `uart_port:=/dev/yourdevice` at launch time (or set `port` in `hardware_params.yaml`)
- **Baud rate:** 115200, 8N1

To find the correct device:
```bash
ls /dev/ttyUSB* /dev/ttyACM* /dev/ttyCH341* 2>/dev/null
# plug in the USB-UART cable, then run again and note what appeared
```

### Frame format

Fixed 16-byte binary frame, same structure in both directions:

```
Offset  Bytes  Field
──────  ─────  ──────────────────────────────────────────────
0       1      SOF = 0xA5  (sync sentinel)
1       1      LEN = 0x0C  (payload byte count = 12)
2–13    12     6 × int16_t, little-endian, centidegrees (1/100 °)
               order: J1 J2 J3 J4 J5 J6
14–15   2      CRC16-MODBUS (poly 0xA001, init 0xFFFF) over bytes [0..13]
               little-endian (lo byte first)
──────  ─────  ──────────────────────────────────────────────
Total   16 bytes
```

Values are in **motor degrees** (after sign-flip and gear-ratio).  Resolution is 0.01°, range ±327.67°.  Frames failing SOF, LEN, or CRC checks are dropped with a `[WARN]` log; the bridge re-syncs by scanning for the next `0xA5` byte.

Suggested MCU DMA setup (STM32 HAL):
```c
HAL_UART_Receive_DMA(&huart, rx_buf, 16);  // fires once per complete frame
```

### Joint sign & gear ratio

The MCU reports raw **motor** angles.  `uart_bridge_node` converts to/from URDF **joint** angles using two per-joint parameters in `arm_hardware/config/hardware_params.yaml`:

```
TX: motor_angle = sign × joint_angle × gear_ratio
RX: joint_angle = sign × motor_angle ÷ gear_ratio
```

| Parameter | Default | Description |
|---|---|---|
| `joint_sign_flip` | `[-1, -1, 1, -1, -1, 1]` | `+1` = same direction as URDF, `-1` = flip |
| `joint_gear_ratio` | `[2, 1, 1, 1, 1, 1]` | Motor turns per joint turn (J1 has 2:1 reduction) |

Order for both lists: `[Joint1, Joint2, Joint3, Joint4, Joint5, Joint6]`.

To tune these values: run view-only mode, move each joint by hand, and adjust until the RViz model matches the physical arm.

### `hardware_params.yaml` key parameters

| Parameter | Default | Description |
|---|---|---|
| `send_rate_hz` | `20.0` | TX rate to MCU (Hz) — keep ≤ 20 Hz to avoid MCU DMA overrun |
| `command_timeout_s` | `0.5` | Idle-hold refresh interval — re-TX last position to keep MCU watchdog alive |
| `override_joint_states` | `false` | Set `true` (real robot) to publish MCU encoder feedback as `/joint_states` |
| `debug_rx` | `false` | Log every raw RX frame and parsed values to the ROS terminal |
| `debug_tx` | `false` | Log every TX frame with source tag (`TRAJ` or `HOLD`) and frame counter |
| `servo_hold_after_traj_s` | `2.0` | Block servo topic messages for this long after a trajectory ends |

---

## Hardware

- **SBC:** OrangePi (RK3588, ARM64)
- **MCU:** ESP32/STM32 connected via USB-UART dongle (pass device path as `uart_port:=...`)
- **Camera:** USB webcam (any OpenCV-compatible device)
- **Arm:** 6-DOF (Joint1 – Joint6)

> **Serial port access:** add your user to the `dialout` group if the device permission is denied:
> ```bash
> sudo usermod -aG dialout $USER   # log out and back in to apply
> sudo systemctl stop ModemManager  # prevents ModemManager hijacking ttyACM* devices
> ```
