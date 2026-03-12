#!/usr/bin/env python3
"""
Simulated STM32 MCU for testing the UART bridge without real hardware.

Creates a virtual serial port pair (pty), prints the device path to
connect the uart_bridge to, then:

  1. Sends encoder feedback (RX from bridge's perspective) at 20 Hz.
  2. Reads command frames (TX from bridge's perspective) as they arrive.
  3. Simulates motor PID: current position moves toward target with a
     first-order time constant (default ~200 ms per step at 20 Hz).
  4. Watchdog: warns if no command frame received within timeout.

Usage:
    python3 scripts/mcu_emulator.py [--start J1,J2,J3,J4,J5,J6]

  Then launch the bridge with:
    uart_port:=<printed device path>

Frame format (same both directions, 16 bytes):
  [0xA5] [0x0C] [6 × int16 LE centideg] [CRC16 LE]
  Values are in *motor* degrees (after sign-flip and gear-ratio applied
  by the bridge).
"""

import argparse
import math
import os
import select
import struct
import sys
import time

# ── Frame constants (must match uart_bridge_node.py) ──────────────────────

SOF        = 0xA5
LEN        = 0x0C
FRAME_SIZE = 16
NUM_JOINTS = 6

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
C_RX     = '\033[36m'   # cyan  — frames we receive (bridge TX)
C_TX     = '\033[33m'   # yellow — frames we send   (bridge RX)
C_WARN   = '\033[31m'   # red
C_INFO   = '\033[32m'   # green
C_DIM    = '\033[90m'   # grey

# ── Motor simulation ─────────────────────────────────────────────────────

def fmt_joints(vals: list[float]) -> str:
    labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    return '  '.join(f'{l}={v:+8.2f}°' for l, v in zip(labels, vals))


def main():
    parser = argparse.ArgumentParser(description='STM32 MCU emulator for UART bridge testing')
    parser.add_argument('--start', type=str, default='0,0,0,0,0,0',
                        help='Initial motor positions in degrees (comma-separated, default: all zeros)')
    parser.add_argument('--rate', type=float, default=20.0,
                        help='Feedback send rate in Hz (default: 20)')
    parser.add_argument('--tau', type=float, default=0.2,
                        help='Motor time constant in seconds — how fast position approaches target (default: 0.2)')
    parser.add_argument('--watchdog', type=float, default=1.0,
                        help='Watchdog timeout in seconds (default: 1.0)')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print warnings and position changes > 0.5°')
    args = parser.parse_args()

    start_pos = [float(x) for x in args.start.split(',')]
    if len(start_pos) != NUM_JOINTS:
        print(f'ERROR: --start needs exactly {NUM_JOINTS} values, got {len(start_pos)}')
        sys.exit(1)

    # Create virtual serial port pair via pty
    master_fd, slave_fd = os.openpty()
    slave_path = os.ttyname(slave_fd)

    print(f'{C_INFO}╔══════════════════════════════════════════════════════════════╗{C_RESET}')
    print(f'{C_INFO}║  MCU Emulator Ready                                        ║{C_RESET}')
    print(f'{C_INFO}║                                                             ║{C_RESET}')
    print(f'{C_INFO}║  Connect the bridge with:                                   ║{C_RESET}')
    print(f'{C_INFO}║    uart_port:={slave_path:<40s}║{C_RESET}')
    print(f'{C_INFO}║                                                             ║{C_RESET}')
    print(f'{C_INFO}║  Rate: {args.rate:5.1f} Hz   Tau: {args.tau:.2f}s   Watchdog: {args.watchdog:.1f}s           ║{C_RESET}')
    print(f'{C_INFO}╚══════════════════════════════════════════════════════════════╝{C_RESET}')
    print()

    current_pos = list(start_pos)          # simulated encoder position (motor deg)
    target_pos  = list(start_pos)          # commanded target (motor deg)
    last_rx_t   = 0.0                      # last time we received a command
    watchdog_warned = False
    tick = 1.0 / args.rate
    alpha = 1.0 - math.exp(-tick / args.tau)  # first-order smoothing coefficient

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
                ready, _, _ = select.select([master_fd], [], [], 0)
                if not ready:
                    break
                chunk = os.read(master_fd, 256)
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
                target_pos = list(angles)
                last_rx_t = time.monotonic()
                watchdog_warned = False

                # Print if target changed
                changed = any(abs(a - b) > 0.05 for a, b in zip(old_target, target_pos))
                if changed or not args.quiet:
                    print(f'{C_RX}[MCU RX #{frame_count_rx:5d}] CMD  {fmt_joints(target_pos)}{C_RESET}')

            # ── Watchdog check ────────────────────────────────────────────
            now = time.monotonic()
            if last_rx_t > 0 and (now - last_rx_t) > args.watchdog and not watchdog_warned:
                gap = now - last_rx_t
                print(f'{C_WARN}[MCU WATCHDOG] No command for {gap:.2f}s '
                      f'(timeout={args.watchdog:.1f}s) — SIGNAL LOSS!{C_RESET}')
                watchdog_warned = True

            # ── Simulate motor movement ───────────────────────────────────
            old_pos = list(current_pos)
            for i in range(NUM_JOINTS):
                error = target_pos[i] - current_pos[i]
                current_pos[i] += alpha * error

            # ── Send encoder feedback to the bridge ───────────────────────
            tx_frame = pack_frame(current_pos)
            try:
                os.write(master_fd, tx_frame)
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
        os.close(master_fd)
        os.close(slave_fd)


if __name__ == '__main__':
    main()
