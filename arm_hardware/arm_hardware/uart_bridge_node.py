#!/usr/bin/env python3
"""
UART bridge node: forwards joint position commands to the STM32 controller
over a serial port, and publishes STM32 encoder feedback as ROS joint states.

TX frame (Orange Pi → STM32):
  "$d1.ddd,d2.ddd,d3.ddd,d4.ddd,d5.ddd,d6.ddd*CCCC\n"
  — '$' start-of-frame, CSV payload in degrees (3 dp), '*' separator,
    CCCC = CRC16-MODBUS of the CSV payload (uppercase hex, 4 chars), then LF.

RX frame (STM32 → Orange Pi):
  Same framed format — actual joint angles in degrees read from encoders.
  Frames failing CRC validation are silently dropped with a WARN log.

Command source:
  Subscribes to /arm_controller/controller_state (JointTrajectoryControllerState,
  100 Hz) and forwards `desired.positions` to the STM32.  This unified source
  works for BOTH MoveIt Servo (streamed joint velocity → controller → state) and
  MoveIt Plan+Execute (action goal → controller → state).  Commands are held
  off until the first non-trivial motion is detected, so the node cannot
  accidentally command zero-degrees at startup.
"""

import math
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


# ── CRC16-MODBUS (poly=0x8005, init=0xFFFF) ───────────────────────────────────
def _crc16_modbus(data: bytes) -> int:
    """Compute CRC16-MODBUS checksum used for TX/RX frame validation."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


class UartBridgeNode(Node):

    def __init__(self):
        super().__init__('uart_bridge_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('port', '/dev/ttyS4')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('serial_timeout', 0.05)
        self.declare_parameter('joint_names',
                               ['Joint1', 'Joint2', 'Joint3',
                                'Joint4', 'Joint5', 'Joint6'])
        self.declare_parameter('send_rate_hz', 50.0)
        self.declare_parameter('command_timeout_s', 0.5)
        self.declare_parameter('publish_real_joint_states', True)
        self.declare_parameter('override_joint_states', False)
        self.declare_parameter('debug_rx', False)
        self.declare_parameter('use_crc_framing', True)

        self._port       = self.get_parameter('port').value
        self._baud       = self.get_parameter('baud_rate').value
        self._ser_to     = self.get_parameter('serial_timeout').value
        self._joints     = self.get_parameter('joint_names').value
        self._send_rate  = self.get_parameter('send_rate_hz').value
        self._cmd_to     = self.get_parameter('command_timeout_s').value
        self._pub_real   = self.get_parameter('publish_real_joint_states').value
        self._override   = self.get_parameter('override_joint_states').value
        self._debug_rx   = self.get_parameter('debug_rx').value
        self._use_crc    = self.get_parameter('use_crc_framing').value

        # ── State ─────────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        # Latest desired joint positions in radians, keyed by joint name
        self._latest_cmd: Optional[dict] = None
        self._cmd_time: float = 0.0
        # Safety gate: don't forward until controller has shown non-zero motion.
        # Prevents commanding 0° at startup before a trajectory is ever sent.
        self._ctrl_enabled: bool = False        # Last joint positions received from hardware (radians).
        # Seeded with zeros so /joint_states is always published in override
        # mode, giving robot_state_publisher a valid TF chain even when no
        # hardware data has arrived yet.
        self._last_hw_positions: list = [0.0] * len(self._joints)
        # ── Publishers ────────────────────────────────────────────────────────
        self._real_js_pub = self.create_publisher(
            JointState, '/real_joint_states', 10)
        if self._override:
            self._js_pub = self.create_publisher(
                JointState, '/joint_states', 10)

        # ── Subscriber ────────────────────────────────────────────────────────
        # Unified command source for both MoveIt Servo and Plan+Execute paths.
        # JointTrajectoryController publishes desired positions here at the
        # ros2_control update rate (100 Hz) whenever it is executing a trajectory
        # (or holding the final position after one completes).
        self.create_subscription(
            JointTrajectoryControllerState,
            '/arm_controller/controller_state',
            self._controller_state_cb, 10)

        # ── Serial port ───────────────────────────────────────────────────────
        self._ser: Optional['serial.Serial'] = None
        if not _SERIAL_AVAILABLE:
            self.get_logger().error(
                'pyserial not installed.  Run: pip install pyserial')
        else:
            self._open_serial()

        # ── Background read thread ────────────────────────────────────────────
        self._running = True
        if self._ser is not None:
            self._read_thread = threading.Thread(
                target=self._read_loop, daemon=True)
            self._read_thread.start()

        # ── Send timer ────────────────────────────────────────────────────────
        period = 1.0 / self._send_rate
        self.create_timer(period, self._send_tick)

        self.get_logger().info(
            f'UART bridge ready — using port={self._port}, baud={self._baud}')
        self.get_logger().info(
            f'  override_joint_states={self._override}  '
            f'use_crc_framing={self._use_crc}  '
            f'debug_rx={self._debug_rx}')

    # ── Serial helpers ────────────────────────────────────────────────────────

    def _open_serial(self):
        try:
            self._ser = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=self._ser_to,
            )
            self.get_logger().info(f'Opened serial port {self._port}')
        except serial.SerialException as e:
            self.get_logger().error(
                f'Failed to open {self._port}: {e}\n'
                f'Set the correct port in hardware_params.yaml and rebuild.')
            self._ser = None

    # ── Controller state callback ─────────────────────────────────────────────

    def _controller_state_cb(self, msg: JointTrajectoryControllerState) -> None:
        """Forward desired joint positions from the JointTrajectoryController.

        Covers both paths:
          • MoveIt Servo  — publishes topic → controller → controller_state
          • Plan+Execute  — action goal → controller → controller_state

        Safety gate: withholds forwarding until at least one trajectory step
        with non-trivial velocity has been observed.  This prevents the node
        from immediately sending the controller's startup-zero state to the
        STM32 before the operator has deliberately commanded any motion.
        """
        if not msg.desired.positions:
            return

        if not self._ctrl_enabled:
            # Enable once the controller is actively executing a trajectory
            # (any joint moving faster than 0.1 deg/s).
            vels = msg.desired.velocities
            if not (vels and any(abs(v) > math.radians(0.1) for v in vels)):
                return
            self._ctrl_enabled = True
            self.get_logger().info(
                'Controller motion detected — UART TX enabled.')

        positions = {
            name: pos
            for name, pos in zip(msg.joint_names, msg.desired.positions)
        }
        with self._lock:
            self._latest_cmd = positions
            self._cmd_time = time.monotonic()

    # ── Send timer ────────────────────────────────────────────────────────────

    def _send_tick(self):
        # ── Heartbeat: always publish /joint_states when in override mode ────
        # This keeps robot_state_publisher's TF chain alive even when the
        # serial port is unavailable or no UART data has arrived yet.
        # _last_hw_positions is seeded with zeros and updated by _read_loop.
        if self._override:
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(self._joints)
            with self._lock:
                js.position = list(self._last_hw_positions)
            self._js_pub.publish(js)

        if self._ser is None:
            return

        with self._lock:
            cmd = self._latest_cmd
            age = time.monotonic() - self._cmd_time if self._latest_cmd else 999

        if cmd is None or age > self._cmd_to:
            return   # no command or stale

        # Build framed TX: $payload*CCCC\n  (or plain CSV if use_crc_framing=false)
        vals_deg = [
            math.degrees(cmd.get(name, 0.0))
            for name in self._joints
        ]
        payload = ','.join(f'{v:.3f}' for v in vals_deg)
        if self._use_crc:
            crc = _crc16_modbus(payload.encode('ascii'))
            line = f'${payload}*{crc:04X}\n'
        else:
            line = payload + '\n'

        try:
            self._ser.write(line.encode('ascii'))
        except serial.SerialException as e:
            self.get_logger().warn(f'UART write error: {e}')

    # ── Background read loop ──────────────────────────────────────────────────

    def _read_loop(self):
        """Continuously read lines from STM32 and publish joint states."""
        while self._running and rclpy.ok():
            if self._ser is None:
                time.sleep(0.1)
                continue
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode('ascii', errors='replace').strip()

                if self._debug_rx:
                    self.get_logger().info(f'[RX raw] {line!r}')

                # ── Frame parsing ─────────────────────────────────────────
                if self._use_crc:
                    # New framed format: $payload*CCCC
                    if not line.startswith('$') or '*' not in line:
                        self.get_logger().warn(
                            f'RX bad frame (expected $payload*CCCC): {line!r}')
                        continue
                    payload, _, crc_str = line[1:].rpartition('*')
                    try:
                        rx_crc = int(crc_str, 16)
                    except ValueError:
                        self.get_logger().warn(
                            f'RX bad CRC field: {line!r}')
                        continue
                    expected_crc = _crc16_modbus(payload.encode('ascii'))
                    if rx_crc != expected_crc:
                        self.get_logger().warn(
                            f'RX CRC mismatch: received {rx_crc:04X}, '
                            f'computed {expected_crc:04X} — frame dropped')
                        continue
                else:
                    # Legacy plain CSV format (no header/CRC)
                    payload = line
                # ─────────────────────────────────────────────────────────

                parts = payload.split(',')
                if len(parts) != len(self._joints):
                    self.get_logger().warn(
                        f'RX wrong field count: got {len(parts)}, '
                        f'expected {len(self._joints)} — payload={payload!r}')
                    continue
                positions_deg = [float(p) for p in parts]
                positions_rad = [math.radians(d) for d in positions_deg]

                if self._debug_rx:
                    deg_str = '  '.join(
                        f'{n}={d:+.2f}°'
                        for n, d in zip(self._joints, positions_deg))
                    self.get_logger().info(f'[RX parsed] {deg_str}')

                js = JointState()
                js.header.stamp = self.get_clock().now().to_msg()
                js.name = list(self._joints)
                js.position = positions_rad

                # Cache for the startup heartbeat (zero-seed until first frame).
                with self._lock:
                    self._last_hw_positions = positions_rad

                if self._pub_real:
                    self._real_js_pub.publish(js)
                # Publish to /joint_states immediately (real-time model update).
                # The _send_tick heartbeat also publishes at 50 Hz but only uses
                # the cached zeros on startup; this direct call takes over as
                # soon as the first UART frame arrives.
                if self._override:
                    self._js_pub.publish(js)

            except (ValueError, UnicodeDecodeError):
                pass   # malformed line — ignore
            except serial.SerialException as e:
                self.get_logger().warn(f'UART read error: {e}')
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f'Read loop error: {e}')
                time.sleep(0.1)

    def destroy_node(self):
        self._running = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UartBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
