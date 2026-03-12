#!/usr/bin/env python3
"""
UART bridge — bidirectional serial link between OrangePi and STM32.

Binary frame format — 16 bytes fixed, BOTH directions:

  Offset  Bytes  Field
  ──────  ─────  ──────────────────────────────────────────────────
  0       1      SOF = 0xA5  (sync sentinel)
  1       1      LEN = 0x0C  (payload byte count = 12, for validation)
  2–13    12     6 × int16_t, little-endian, unit = centidegrees (1/100 °)
                 order: J1 J2 J3 J4 J5 J6
  14–15   2      CRC16-MODBUS (poly 0xA001, init 0xFFFF) over bytes [0..13]
                 little-endian (lo byte first)
  ──────  ─────  ──────────────────────────────────────────────────
  Total   16 bytes

  int16 centidegrees: resolution 0.01°, range ±327.67° (sufficient for all joints).
  Fixed frame size → MCU uses HAL_UART_Receive_DMA(&huart, buf, 16); DMA fires
  exactly once per complete frame — no idle-line, no CRLF, no variable length.
  0xA5 SOF enables byte-hunt resync after any corruption.

TX sources (in priority order):
  1. Action server  /arm_controller/follow_joint_trajectory  (homing / Plan+Execute)
  2. Servo topic    /arm_controller/joint_trajectory         (socket / keyboard teleop)
  3. Idle hold      -> re-TX last commanded position         (preserves trajectory target)
                      (falls back to encoder position only at cold start, before any command)

RX -> publishes /joint_states (and /real_joint_states).
TX is gated on the first valid RX frame: no frames are sent until the
actual encoder position is known, preventing a startup snap to zero.
Once RX is received, idle-hold echoes the encoder position ("hold where
you are") until an explicit command arrives.

Key parameters (hardware_params.yaml):
  port             serial device path (required)
  joint_sign_flip  per-joint sign correction  (default: [-1,-1,1,-1,-1,1])
  joint_gear_ratio per-joint gear ratio        (default: [2,1,1,1,1,1])
  override_joint_states  own /joint_states     (default: false)
"""

import bisect
import math
import struct
import threading
import time
from typing import Optional

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory

try:
    import serial
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False


_FRAME_SOF  = 0xA5
_FRAME_LEN  = 0x0C   # 6 joints × 2 bytes
_FRAME_SIZE = 16     # SOF(1) + LEN(1) + payload(12) + CRC(2)


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _pack_frame(angles_deg: list) -> bytes:
    """Pack 6 angles (degrees) into a 16-byte binary frame."""
    centi = [max(-32767, min(32767, round(a * 100))) for a in angles_deg]
    header_payload = struct.pack('<BB6h', _FRAME_SOF, _FRAME_LEN, *centi)
    return header_payload + struct.pack('<H', _crc16(header_payload))


def _unpack_frame(data: bytes):
    """Unpack a 16-byte binary frame. Returns list of 6 floats (degrees), or None."""
    if len(data) != _FRAME_SIZE:
        return None
    if data[0] != _FRAME_SOF or data[1] != _FRAME_LEN:
        return None
    if _crc16(data[:14]) != struct.unpack_from('<H', data, 14)[0]:
        return None
    return [v / 100.0 for v in struct.unpack_from('<6h', data, 2)]


def _interp(times: list, pts: list, t: float) -> dict:
    """Linearly interpolate over trajectory waypoints at time t (seconds)."""
    if t <= times[0]:  return dict(pts[0])
    if t >= times[-1]: return dict(pts[-1])
    i = bisect.bisect_right(times, t) - 1
    a = (t - times[i]) / (times[i+1] - times[i])
    return {k: (1-a)*pts[i].get(k, 0.0) + a*pts[i+1].get(k, 0.0)
            for k in set(pts[i]) | set(pts[i+1])}


class UartBridgeNode(Node):

    def __init__(self):
        super().__init__('uart_bridge_node')

        def p(name, default): return self.declare_parameter(name, default).value

        self._port      = p('port', '')
        self._baud      = p('baud_rate', 115200)
        self._joints    = p('joint_names', ['Joint1','Joint2','Joint3','Joint4','Joint5','Joint6'])
        self._rate      = p('send_rate_hz', 20.0)
        self._cmd_to    = p('command_timeout_s', 0.5)
        self._override  = p('override_joint_states', False)
        self._debug_rx  = p('debug_rx', False)
        self._debug_tx  = p('debug_tx', False)
        self._sign      = list(p('joint_sign_flip',  [-1.0,-1.0, 1.0,-1.0,-1.0, 1.0]))
        self._ratio     = list(p('joint_gear_ratio', [ 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        ser_to          = p('serial_timeout', 0.05)

        self._servo_hold  = p('servo_hold_after_traj_s', 2.0)

        self._lock        = threading.Lock()
        self._hw_pos      = [0.0] * len(self._joints)
        self._got_hw      = False
        self._cmd: Optional[dict] = None
        self._cmd_t       = 0.0
        self._traj_active = False
        self._traj_end_t  = 0.0   # monotonic time of last trajectory completion
        self._running     = True
        self._tx_count    = 0      # total TX frames sent (for diagnostics)

        self._real_pub = self.create_publisher(JointState, '/real_joint_states', 10)
        if self._override:
            self._js_pub = self.create_publisher(JointState, '/joint_states', 10)

        self.create_subscription(JointTrajectory,
                                 '/arm_controller/joint_trajectory',
                                 self._servo_cb, 10)

        self._as = ActionServer(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            execute_callback=self._exec_cb,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

        self._ser: Optional['serial.Serial'] = None
        if not self._port:
            self.get_logger().error(
                '"port" param not set. Pass uart_port:=/dev/yourdevice at launch.')
        elif not _SERIAL_OK:
            self.get_logger().error('pyserial not installed: pip install pyserial')
        else:
            try:
                self._ser = serial.Serial(self._port, self._baud, timeout=ser_to)
                self.get_logger().info(f'Opened {self._port} @ {self._baud}')
            except serial.SerialException as e:
                self.get_logger().error(f'Cannot open {self._port}: {e}')

        if self._ser:
            threading.Thread(target=self._rx_loop, daemon=True).start()
            threading.Thread(target=self._tx_loop, daemon=True).start()

    # ---- servo topic --------------------------------------------------------

    def _servo_cb(self, msg: JointTrajectory):
        # Check and write _cmd in a single lock to prevent TOCTOU:
        # without this, a servo message could slip through and overwrite
        # a trajectory command that started between the two lock acquisitions.
        if not msg.points:
            return
        pt = msg.points[-1]
        if not pt.positions:
            return
        with self._lock:
            if self._traj_active:
                return
            # Block servo messages for a hold window after a trajectory ends.
            # MoveIt Servo publishes halt messages (containing encoder positions)
            # that would overwrite the trajectory's final target before the MCU
            # has time to converge.
            if time.monotonic() - self._traj_end_t < self._servo_hold:
                return
            self._cmd   = dict(zip(msg.joint_names, pt.positions))
            self._cmd_t = time.monotonic()

    # ---- action server ------------------------------------------------------

    def _exec_cb(self, goal_handle):
        traj = goal_handle.request.trajectory
        if not traj.points:
            goal_handle.abort()
            r = FollowJointTrajectory.Result()
            r.error_code, r.error_string = r.INVALID_GOAL, 'Empty trajectory'
            return r

        times = [pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
                 for pt in traj.points]
        pts   = [dict(zip(traj.joint_names, pt.positions)) for pt in traj.points]
        tick  = 1.0 / self._rate
        t0    = time.monotonic()

        self.get_logger().info(
            f'[Traj] {len(traj.points)} pts / {times[-1]:.2f}s / '
            f'{", ".join(traj.joint_names)}')

        with self._lock:
            self._traj_active = True
        try:
            while time.monotonic() < t0 + times[-1]:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    r = FollowJointTrajectory.Result()
                    r.error_code = r.SUCCESSFUL
                    return r
                with self._lock:
                    self._cmd   = _interp(times, pts, time.monotonic() - t0)
                    self._cmd_t = time.monotonic()
                time.sleep(tick)
            with self._lock:
                self._cmd, self._cmd_t = dict(pts[-1]), time.monotonic()
        finally:
            with self._lock:
                self._traj_active = False
                self._traj_end_t  = time.monotonic()

        self.get_logger().info('[Traj] done.')
        goal_handle.succeed()
        r = FollowJointTrajectory.Result()
        r.error_code = r.SUCCESSFUL
        return r

    # ---- TX thread ----------------------------------------------------------

    def _tx_loop(self):
        """Dedicated TX thread — runs at send_rate_hz, independent of the
        ROS executor.  This guarantees the MCU receives frames at a steady
        cadence even when the executor is busy processing action-server
        callbacks, avoiding watchdog timeouts."""
        tick = 1.0 / self._rate
        next_t = time.monotonic()

        while self._running and rclpy.ok():
            # ── precise cadence: sleep until the next tick ────────────
            now = time.monotonic()
            sleep_s = next_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            next_t += tick
            # If we fell behind (e.g. OS scheduling jitter), skip ticks
            # rather than bursting — the MCU only needs the latest value.
            if next_t < time.monotonic():
                next_t = time.monotonic() + tick

            # ── snapshot shared state under one lock ──────────────────
            with self._lock:
                got    = self._got_hw
                hw     = list(self._hw_pos)
                active = self._traj_active
                cmd    = self._cmd
                age    = (time.monotonic() - self._cmd_t) if self._cmd else 999.0

            # Gate on first valid encoder frame.  Before that we don't
            # know the real arm position — TX-ing zeros would snap the
            # arm and publishing zeros on /joint_states misleads MoveIt.
            if not got and not active:
                continue

            # Publish /joint_states from real encoder data.
            if self._override and got:
                js = JointState()
                js.header.stamp = self.get_clock().now().to_msg()
                js.name, js.position = list(self._joints), hw
                self._js_pub.publish(js)

            # Idle hold: no active trajectory and command is stale.
            # Priority: last commanded position → real encoder position.
            if not active and age > self._cmd_to:
                if cmd is not None:
                    hold_pos = [cmd.get(n, 0.0) for n in self._joints]
                else:
                    hold_pos = hw
                cmd = dict(zip(self._joints, hold_pos))
                with self._lock:
                    self._cmd, self._cmd_t = cmd, time.monotonic()
                age = 0.0

            if not cmd or age > self._cmd_to:
                continue

            vals = [s * r * math.degrees(cmd.get(n, 0.0))
                    for n, s, r in zip(self._joints, self._sign, self._ratio)]
            frame = _pack_frame(vals)
            try:
                self._ser.write(frame)
                self._tx_count += 1
                if self._debug_tx:
                    src = 'TRAJ' if active else 'HOLD'
                    self.get_logger().info(
                        f'[TX #{self._tx_count} {src}] {frame.hex(" ")}  '
                        f'({" ".join(f"{v:+.2f}" for v in vals)}deg)')
            except serial.SerialException as e:
                self.get_logger().warn(f'TX error: {e}')
                time.sleep(0.1)

    # ---- RX loop ------------------------------------------------------------

    def _rx_loop(self):
        first = True
        while self._running and rclpy.ok():
            try:
                # Byte-hunt: scan for SOF byte 0xA5, then read remaining 15 bytes.
                sof = self._ser.read(1)
                if not sof:
                    continue
                if sof[0] != _FRAME_SOF:
                    continue  # discard until aligned
                rest = self._ser.read(_FRAME_SIZE - 1)
                if len(rest) != _FRAME_SIZE - 1:
                    continue  # timeout mid-frame
                frame = bytes(sof) + rest

                if self._debug_rx:
                    self.get_logger().info(f'[RX raw] {frame.hex(" ")}')

                angles_deg = _unpack_frame(frame)
                if angles_deg is None:
                    self.get_logger().warn(f'[RX] bad frame: {frame.hex(" ")}')
                    continue

                pos = [s * math.radians(d) / r
                       for d, s, r in zip(angles_deg, self._sign, self._ratio)]

                if self._debug_rx:
                    self.get_logger().info('[RX] ' + '  '.join(
                        f'{n}={math.degrees(v):+.2f}deg'
                        for n, v in zip(self._joints, pos)))

                js = JointState()
                js.header.stamp = self.get_clock().now().to_msg()
                js.name, js.position = list(self._joints), pos

                with self._lock:
                    self._hw_pos = pos
                    self._got_hw = True

                if first:
                    first = False
                    self.get_logger().info('[RX] First frame — RX nominal.')

                self._real_pub.publish(js)
                if self._override:
                    self._js_pub.publish(js)

            except serial.SerialException as e:
                self.get_logger().warn(f'RX error: {e}')
                time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f'RX loop: {e}')
                time.sleep(0.1)

    def destroy_node(self):
        self._running = False
        if self._ser:
            try: self._ser.close()
            except Exception: pass
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
