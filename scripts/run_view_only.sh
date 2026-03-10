#!/usr/bin/env bash
# View real arm joint angles in RViz — without any motion commands.
#
# Starts: robot_state_publisher + uart_bridge (RX only) + RViz
# Does NOT start: arm_controller, move_group, servo, socket_teleop
#
# The arm model in RViz will follow the actual joint angles reported
# by the MCU encoder feedback over UART.
#
# Usage:
#   ./scripts/run_view_only.sh
#   ./scripts/run_view_only.sh --port /dev/ttyCH341USB0
#   ./scripts/run_view_only.sh --port /dev/ttyCH341USB0 --debug       # logs every raw frame
#   ./scripts/run_view_only.sh --port /dev/ttyCH341USB0 --crc         # expect $...*CCCC framing
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

UART_PORT="/dev/ttyS3"
DEBUG_RX="false"
USE_CRC="true"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)  UART_PORT="$2"; shift 2 ;;
        --debug) DEBUG_RX="true"; shift ;;
        --crc)   USE_CRC="true";  shift ;;
        *)       UART_PORT="$1";  shift ;;   # positional fallback
    esac
done

export PATH="$HOME/.local/bin:$PATH"
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

echo "View-only mode: arm angles from UART → robot_state_publisher → RViz"
echo "  UART port    : $UART_PORT"
echo "  debug_rx     : $DEBUG_RX   (add --debug to log every raw frame)"
echo "  use_crc      : $USE_CRC    (add --crc if MCU sends \$...*CCCC format)"
echo "No motion commands will be sent to the arm."
echo ""

ros2 launch arm_bringup view.launch.py \
    uart_port:="$UART_PORT" \
    debug_rx:="$DEBUG_RX" \
    use_crc_framing:="$USE_CRC"
