#!/usr/bin/env python3
"""
UART bridge node: forwards joint position commands to the STM32 controller
over a serial port, and publishes STM32 encoder feedback as ROS joint states.

TX frame (Orange Pi → STM32):
  "d1.ddd,d2.ddd,d3.ddd,d4.ddd,d5.ddd,d6.ddd\n"
  — joint angles in degrees, 3 decimal places, Joint1..Joint6 order.

RX frame (STM32 → Orange Pi):
  Same CSV format — actual joint angles in degrees read from encoders.

The node subscribes to /arm_controller/joint_trajectory (produced by MoveIt
Servo or move_group) and sends the instantaneous target positions at
`send_rate_hz`.  A background thread reads STM32 feedback and publishes
/real_joint_states (and optionally /joint_states for MoveIt feedback).
"""

import math
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

try:
    import serial
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


class UartBridgeNode(Node):

    def __init__(self):
        super().__init__('uart_bridge_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('port', '/dev/ttyS3')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('serial_timeout', 0.05)
        self.declare_parameter('joint_names',
                               ['Joint1', 'Joint2', 'Joint3',
                                'Joint4', 'Joint5', 'Joint6'])
        self.declare_parameter('send_rate_hz', 50.0)
        self.declare_parameter('command_timeout_s', 0.5)
        self.declare_parameter('publish_real_joint_states', True)
        self.declare_parameter('override_joint_states', False)

        self._port       = self.get_parameter('port').value
        self._baud       = self.get_parameter('baud_rate').value
        self._ser_to     = self.get_parameter('serial_timeout').value
        self._joints     = self.get_parameter('joint_names').value
        self._send_rate  = self.get_parameter('send_rate_hz').value
        self._cmd_to     = self.get_parameter('command_timeout_s').value
        self._pub_real   = self.get_parameter('publish_real_joint_states').value
        self._override   = self.get_parameter('override_joint_states').value

        # ── State ─────────────────────────────────────────────────────────────
        self._lock = threading.Lock()
        # Latest desired joint positions in radians, keyed by joint name
        self._latest_cmd: Optional[dict] = None
        self._cmd_time: float = 0.0

        # ── Publishers ────────────────────────────────────────────────────────
        self._real_js_pub = self.create_publisher(
            JointState, '/real_joint_states', 10)
        if self._override:
            self._js_pub = self.create_publisher(
                JointState, '/joint_states', 10)

        # ── Subscriber ────────────────────────────────────────────────────────
        self.create_subscription(
            JointTrajectory, '/arm_controller/joint_trajectory',
            self._traj_cb, 10)

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
            f'UART bridge ready: port={self._port}, baud={self._baud}')

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

    # ── Trajectory callback ───────────────────────────────────────────────────

    def _traj_cb(self, msg: JointTrajectory):
        if not msg.points:
            return

        # Use the last trajectory point as the target (appropriate for Servo
        # which publishes short 1-2 point trajectories at ~30 Hz).
        last = msg.points[-1]
        positions = {
            name: pos
            for name, pos in zip(msg.joint_names, last.positions)
        }

        with self._lock:
            self._latest_cmd = positions
            self._cmd_time = time.monotonic()

    # ── Send timer ────────────────────────────────────────────────────────────

    def _send_tick(self):
        if self._ser is None:
            return

        with self._lock:
            cmd = self._latest_cmd
            age = time.monotonic() - self._cmd_time if self._latest_cmd else 999

        if cmd is None or age > self._cmd_to:
            return   # no command or stale

        # Build CSV frame: degrees, Joint1..Joint6 order
        vals_deg = [
            math.degrees(cmd.get(name, 0.0))
            for name in self._joints
        ]
        line = ','.join(f'{v:.3f}' for v in vals_deg) + '\n'

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
                parts = line.split(',')
                if len(parts) != len(self._joints):
                    continue
                positions_deg = [float(p) for p in parts]
                positions_rad = [math.radians(d) for d in positions_deg]

                js = JointState()
                js.header.stamp = self.get_clock().now().to_msg()
                js.name = list(self._joints)
                js.position = positions_rad

                if self._pub_real:
                    self._real_js_pub.publish(js)
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
