#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST="127.0.0.1"
PORT="9999"
CONTROL_ORIENTATION="true"
DEBUG_LOG="/tmp/ik_workspace_debug.csv"
PROBE_LOG_DIR="/tmp/workspace_probe"
MAX_SHELL_POINTS="8"
TRAVEL_TIME="0.8"
INSIDE_HOLD="0.3"
OUTSIDE_HOLD="0.3"
HOME_HOLD="0.2"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_workspace_probe_test.sh [options]

Starts a headless ROS control stack, starts ik_direct teleop with CSV logging,
then runs the deterministic workspace probe sender.

Options:
  --host IP                 TCP host for the pose sender (default: 127.0.0.1)
  --port PORT               TCP port for the pose sender (default: 9999)
  --6d                      Run 6D IK mode (default)
  --pos-only                Run position-only IK mode
  --debug-log PATH          ik_teleop_node CSV log path (default: /tmp/ik_workspace_debug.csv)
  --probe-log-dir DIR       workspace probe CSV directory (default: /tmp/workspace_probe)
  --max-shell-points N      number of reachable shell rays to test (default: 8)
  --travel-time SEC         move time per segment (default: 0.8)
  --inside-hold SEC         hold time at reachable pose (default: 0.3)
  --outside-hold SEC        hold time at outside pose (default: 0.3)
  --home-hold SEC           hold time at home pose (default: 0.2)
  -h, --help                show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --6d)
      CONTROL_ORIENTATION="true"
      shift
      ;;
    --pos-only)
      CONTROL_ORIENTATION="false"
      shift
      ;;
    --debug-log)
      DEBUG_LOG="$2"
      shift 2
      ;;
    --probe-log-dir)
      PROBE_LOG_DIR="$2"
      shift 2
      ;;
    --max-shell-points)
      MAX_SHELL_POINTS="$2"
      shift 2
      ;;
    --travel-time)
      TRAVEL_TIME="$2"
      shift 2
      ;;
    --inside-hold)
      INSIDE_HOLD="$2"
      shift 2
      ;;
    --outside-hold)
      OUTSIDE_HOLD="$2"
      shift 2
      ;;
    --home-hold)
      HOME_HOLD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[workspace-probe] Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

export HOME="${HOME:-/tmp/codex-ros-home}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/codex-ros-home/log}"
mkdir -p "$ROS_LOG_DIR" "$PROBE_LOG_DIR"

cleanup() {
  local status=$?
  if [[ -n "${PROBE_PID:-}" ]]; then
    kill "$PROBE_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TELEOP_PID:-}" ]]; then
    kill "$TELEOP_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CONTROL_PID:-}" ]]; then
    kill "$CONTROL_PID" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
  return $status
}
trap cleanup EXIT INT TERM

set +u
source /opt/ros/humble/setup.bash
source "$ROOT_DIR/install/setup.bash"
set -u

echo "[workspace-probe] Launching control.launch.py"
ros2 launch robotic_arm_v4_urdf control.launch.py use_rviz:=false use_real_robot:=false \
  >"$ROS_LOG_DIR/workspace_probe_control.log" 2>&1 &
CONTROL_PID=$!

sleep 5

echo "[workspace-probe] Launching teleop.launch.py"
ros2 launch arm_teleop teleop.launch.py \
  teleop_mode:=ik_direct \
  control_orientation:="$CONTROL_ORIENTATION" \
  debug_log:="$DEBUG_LOG" \
  host:="$HOST" \
  port:="$PORT" \
  >"$ROS_LOG_DIR/workspace_probe_teleop.log" 2>&1 &
TELEOP_PID=$!

sleep 5

echo "[workspace-probe] Running deterministic sender"
python3 "$ROOT_DIR/scripts/workspace_pose_sender.py" \
  --host "$HOST" \
  --port "$PORT" \
  --max-shell-points "$MAX_SHELL_POINTS" \
  --travel-time "$TRAVEL_TIME" \
  --inside-hold "$INSIDE_HOLD" \
  --outside-hold "$OUTSIDE_HOLD" \
  --home-hold "$HOME_HOLD" \
  --log-dir "$PROBE_LOG_DIR" &
PROBE_PID=$!

wait "$PROBE_PID"
PROBE_PID=""

echo "[workspace-probe] node CSV  -> $DEBUG_LOG"
echo "[workspace-probe] probe CSV -> $PROBE_LOG_DIR"
echo "[workspace-probe] ROS logs  -> $ROS_LOG_DIR"
