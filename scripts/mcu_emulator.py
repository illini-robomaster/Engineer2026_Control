#!/usr/bin/env python3
"""
Simulated STM32 MCU for testing the UART bridge without real hardware.

By default this creates a virtual serial port pair (pty), prints the
device path to connect the uart_bridge to, then:

  1. Sends encoder feedback (RX from bridge's perspective) at 20 Hz.
  2. Reads command frames (TX from bridge's perspective) as they arrive.
  3. Simulates motor PID: current position moves toward target with a
     first-order time constant (default ~200 ms per step at 20 Hz).
  4. Watchdog: warns if no command frame received within timeout.

Usage:
    python3 scripts/mcu_emulator.py [--start J1,J2,J3,J4,J5,J6]
    python3 scripts/mcu_emulator.py [--start ...] [--uart /dev/ttyUSB0]

  In the default PTY mode, launch the bridge with:
    uart_port:=<printed device path>

  With --uart, the emulator opens that serial device directly instead of
  creating a virtual PTY pair.

Frame format (same both directions, 16 bytes):
  [0xA5] [0x0C] [6 × int16 LE centideg] [CRC16 LE]
  Values are in *motor* degrees (after sign-flip and gear-ratio applied
  by the bridge).
"""

import argparse
import os
import select
import struct
import sys
import tty
import time

# ── Frame constants (must match uart_bridge_node.py) ──────────────────────

SOF        = 0xA5
LEN        = 0x0C
FRAME_SIZE = 16
NUM_JOINTS = 6

# ── Per-joint limits (motor-angle space, degrees) ─────────────────────────
# Must match ARM_CMD_MIN_DEG / ARM_CMD_MAX_DEG in arm_mc02.h.
#   J1 (gear 2:1, sign -1):  ±180° joint → ±360° motor
#   J2 (gear 1:1, sign -1):   ±90° joint →  ±90° motor
#   J3 (gear 1:1, sign +1):  -68.75°…+229.18° motor
#   J4 (gear 1:1, sign -1):  ±180° motor
#   J5 (gear 1:1, sign -1):  ±150° motor
#   J6 (gear 1:1, sign +1):  ±180° motor
JOINT_MIN_DEG = [-360.0, -90.0, -68.75, -180.0, -150.0, -180.0]
JOINT_MAX_DEG = [ 360.0,  90.0, 229.18,  180.0,  150.0,  180.0]

# ── Per-joint velocity limits (deg/s, converted from ARM_VEL_LIM rad/s) ──
# ARM_VEL_LIM = {1.2, 1.2, 1.2, 3.0, 1.5, 3.6} rad/s
import math as _math
JOINT_VEL_LIM_DEG_S = [v * 180.0 / _math.pi for v in [1.2, 1.2, 1.2, 3.0, 1.5, 3.6]]

# ── CRC-16 MODBUS (poly 0xA001, init 0xFFFF) ─────────────────────────────

def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def pack_frame(angles_deg: list[float]) -> bytes:
    centi = [max(-32767, min(32767, round(a * 100))) for a in angles_deg]
    hdr = struct.pack('<BB6h', SOF, LEN, *centi)
    return hdr + struct.pack('<H', crc16(hdr))


def unpack_frame(data: bytes) -> list[float] | None:
    if len(data) != FRAME_SIZE:
        return None
    if data[0] != SOF or data[1] != LEN:
        return None
    if crc16(data[:14]) != struct.unpack_from('<H', data, 14)[0]:
        return None
    return [v / 100.0 for v in struct.unpack_from('<6h', data, 2)]

# ── Colours for terminal output ───────────────────────────────────────────

C_RESET  = '\033[0m'
C_RX     = '\033[92m'   # bright green — frames we receive (bridge TX)
C_TX     = '\033[33m'   # yellow — frames we send   (bridge RX)
C_WARN   = '\033[31m'   # red
C_INFO   = '\033[32m'   # green
C_DIM    = '\033[90m'   # grey

# ── Motor simulation ─────────────────────────────────────────────────────

def fmt_joints(vals: list[float]) -> str:
    labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    return '  '.join(f'{l}={v:+8.2f}°' for l, v in zip(labels, vals))


def set_raw_mode(fd: int) -> None:
    tty.setraw(fd)


def main():
    parser = argparse.ArgumentParser(description='STM32 MCU emulator for UART bridge testing')
    parser.add_argument('--start', type=str, default='0,0,0,0,0,0',
                        help='Initial motor positions in degrees (comma-separated, default: all zeros)')
    parser.add_argument('--uart', type=str, default=None,
                        help='Directly use this UART device instead of creating a virtual PTY pair')
    parser.add_argument('--baud', type=int, default=115200,
                        help='UART baud rate for --uart mode (default: 115200)')
    parser.add_argument('--rate', type=float, default=50.0,
                        help='Feedback send rate in Hz (default: 50, matches MCU)')
    parser.add_argument('--startup-wait', type=float, default=2.0,
                        help='Grace period to wait for the first valid command frame before warning (default: 2.0)')
    parser.add_argument('--watchdog', type=float, default=2.0,
                        help='Watchdog timeout in seconds (default: 2.0, matches MCU WATCHDOG_MS=2000)')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print warnings and position changes > 0.5°')
    args = parser.parse_args()

    start_pos = [float(x) for x in args.start.split(',')]
    if len(start_pos) != NUM_JOINTS:
        print(f'ERROR: --start needs exactly {NUM_JOINTS} values, got {len(start_pos)}')
        sys.exit(1)

    serial_dev = None
    slave_fd = None

    if args.uart:
        try:
            import serial
        except ImportError:
            print('ERROR: --uart requires pyserial (pip install pyserial)')
            sys.exit(1)

        try:
            serial_dev = serial.Serial(args.uart, args.baud, timeout=0)
        except serial.SerialException as exc:
            print(f'ERROR: failed to open UART {args.uart}: {exc}')
            sys.exit(1)

        comm_fd = serial_dev.fileno()
        slave_path = args.uart
    else:
        # Create virtual serial port pair via pty
        master_fd, slave_fd = os.openpty()
        set_raw_mode(master_fd)
        set_raw_mode(slave_fd)
        comm_fd = master_fd
        slave_path = os.ttyname(slave_fd)

    print(f'{C_INFO}╔══════════════════════════════════════════════════════════════╗{C_RESET}')
    print(f'{C_INFO}║  MCU Emulator Ready                                        ║{C_RESET}')
    print(f'{C_INFO}║                                                             ║{C_RESET}')
    if args.uart:
        print(f'{C_INFO}║  Using UART device:                                         ║{C_RESET}')
        print(f'{C_INFO}║    {slave_path:<56s}║{C_RESET}')
        print(f'{C_INFO}║  Baud: {args.baud:<52d}║{C_RESET}')
    else:
        print(f'{C_INFO}║  Connect the bridge with:                                   ║{C_RESET}')
        print(f'{C_INFO}║    uart_port:={slave_path:<40s}║{C_RESET}')
    print(f'{C_INFO}║                                                             ║{C_RESET}')
    print(f'{C_INFO}║  Rate: {args.rate:5.1f} Hz   Watchdog: {args.watchdog:.1f}s                        ║{C_RESET}')
    print(f'{C_INFO}║  Waiting up to {args.startup_wait:4.1f}s for first valid command frame          ║{C_RESET}')
    print(f'{C_INFO}╚══════════════════════════════════════════════════════════════╝{C_RESET}')
    print()

    current_pos = list(start_pos)          # simulated encoder position (motor deg)
    target_pos  = list(start_pos)          # commanded target (motor deg)
    last_rx_t   = 0.0                      # last time we received a command
    watchdog_warned = False
    tick = 1.0 / args.rate
    start_t = time.monotonic()
    startup_warned = False

    rx_buf = bytearray()
    frame_count_rx = 0  # commands received from bridge
    frame_count_tx = 0  # feedback frames sent to bridge

    print(f'{C_INFO}Initial position (motor deg):{C_RESET}')
    print(f'  {fmt_joints(current_pos)}')
    print()

    try:
        while True:
            loop_start = time.monotonic()

            # ── Read any available command frames from the bridge ──────────
            while True:
                ready, _, _ = select.select([comm_fd], [], [], 0)
                if not ready:
                    break
                chunk = os.read(comm_fd, 256)
                if not chunk:
                    break
                rx_buf.extend(chunk)

            # Parse complete frames from buffer
            while len(rx_buf) >= FRAME_SIZE:
                # Hunt for SOF
                sof_idx = -1
                for i in range(len(rx_buf)):
                    if rx_buf[i] == SOF:
                        sof_idx = i
                        break
                if sof_idx < 0:
                    rx_buf.clear()
                    break
                if sof_idx > 0:
                    print(f'{C_WARN}[MCU] Discarded {sof_idx} bytes before SOF{C_RESET}')
                    del rx_buf[:sof_idx]
                if len(rx_buf) < FRAME_SIZE:
                    break

                frame_data = bytes(rx_buf[:FRAME_SIZE])
                del rx_buf[:FRAME_SIZE]

                angles = unpack_frame(frame_data)
                if angles is None:
                    print(f'{C_WARN}[MCU RX] Bad CRC: {frame_data.hex(" ")}{C_RESET}')
                    continue

                frame_count_rx += 1
                old_target = list(target_pos)
                # Clamp to per-joint limits (matches MCU arm_uart_task.cc sanity clamp)
                target_pos = [
                    max(JOINT_MIN_DEG[i], min(JOINT_MAX_DEG[i], angles[i]))
                    for i in range(NUM_JOINTS)
                ]
                last_rx_t = time.monotonic()
                watchdog_warned = False

                if frame_count_rx == 1 and not args.quiet:
                    wait_s = last_rx_t - start_t
                    print(f'{C_INFO}[MCU] First valid command frame after {wait_s:.2f}s{C_RESET}')

                # Print if target changed
                changed = any(abs(a - b) > 0.05 for a, b in zip(old_target, target_pos))
                if changed or not args.quiet:
                    print(f'{C_RX}[MCU RX #{frame_count_rx:5d}] CMD  {fmt_joints(target_pos)}{C_RESET}')

            # ── Watchdog check ────────────────────────────────────────────
            now = time.monotonic()
            if last_rx_t == 0.0:
                if not startup_warned and (now - start_t) > args.startup_wait:
                    print(f'{C_WARN}[MCU] No valid command frame received after {args.startup_wait:.1f}s; '
                          f'continuing to wait while sending feedback.{C_RESET}')
                    startup_warned = True
            elif (now - last_rx_t) > args.watchdog and not watchdog_warned:
                gap = now - last_rx_t
                print(f'{C_WARN}[MCU WATCHDOG] No command for {gap:.2f}s '
                      f'(timeout={args.watchdog:.1f}s) — SIGNAL LOSS!{C_RESET}')
                watchdog_warned = True

            # ── Simulate motor movement ───────────────────────────────────
            # Velocity-capped integrator matching ARM_VEL_LIM on the MCU.
            # Each joint moves toward target at most vel_lim * dt per tick,
            # then the result is clamped to per-joint position limits.
            old_pos = list(current_pos)
            for i in range(NUM_JOINTS):
                error = target_pos[i] - current_pos[i]
                max_step = JOINT_VEL_LIM_DEG_S[i] * tick
                step = max(-max_step, min(max_step, error))
                current_pos[i] = max(JOINT_MIN_DEG[i],
                                     min(JOINT_MAX_DEG[i],
                                         current_pos[i] + step))

            # ── Send encoder feedback to the bridge ───────────────────────
            tx_frame = pack_frame(current_pos)
            try:
                os.write(comm_fd, tx_frame)
                frame_count_tx += 1
            except OSError as e:
                print(f'{C_WARN}[MCU TX] Write error: {e}{C_RESET}')

            # Print feedback (with movement indication)
            moved = any(abs(a - b) > 0.05 for a, b in zip(old_pos, current_pos))
            if moved or (not args.quiet and frame_count_tx % 20 == 0):
                converged = all(abs(t - c) < 0.1 for t, c in zip(target_pos, current_pos))
                tag = 'HOLD' if converged else 'MOVE'
                print(f'{C_TX}[MCU TX #{frame_count_tx:5d}] {tag} {fmt_joints(current_pos)}{C_RESET}')

            # ── Sleep for remainder of tick ────────────────────────────────
            elapsed = time.monotonic() - loop_start
            remaining = tick - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print(f'\n{C_INFO}[MCU] Shutting down. Sent {frame_count_tx} frames, received {frame_count_rx} commands.{C_RESET}')
    finally:
        if serial_dev is not None:
            serial_dev.close()
        else:
            os.close(comm_fd)
            if slave_fd is not None:
                os.close(slave_fd)


if __name__ == '__main__':
    main()
