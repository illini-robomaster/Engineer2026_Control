#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${ARM_VISION_VENV:-${REPO_ROOT}/arm_vision/.venv}"
WITH_SPACEMOUSE=0
RECREATE=0
USE_SYSTEM_SITE_PACKAGES="auto"

usage() {
  cat <<EOF
Usage: bash scripts/setup_arm_vision_venv.sh [options]

Options:
  --with-spacemouse          Install pyspacemouse in addition to the base client deps.
  --recreate                 Remove and recreate the venv first.
  --system-site-packages     Reuse host Python packages such as OpenCV.
  --no-system-site-packages  Force a fully isolated venv.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-spacemouse)
      WITH_SPACEMOUSE=1
      shift
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --system-site-packages)
      USE_SYSTEM_SITE_PACKAGES=1
      shift
      ;;
    --no-system-site-packages)
      USE_SYSTEM_SITE_PACKAGES=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required."
  exit 1
fi

if [[ "${USE_SYSTEM_SITE_PACKAGES}" == "auto" ]]; then
  if python3 - <<'PY' >/dev/null 2>&1
import cv2
import numpy
import scipy
import yaml
PY
  then
    USE_SYSTEM_SITE_PACKAGES=1
    echo "Reusing host Python packages (OpenCV/NumPy/SciPy/PyYAML) via --system-site-packages."
  else
    USE_SYSTEM_SITE_PACKAGES=0
  fi
fi

VENV_FLAGS=()
if [[ "${USE_SYSTEM_SITE_PACKAGES}" == 1 ]]; then
  VENV_FLAGS+=(--system-site-packages)
fi

if (( RECREATE )) && [[ -d "${VENV_DIR}" ]]; then
  rm -rf "${VENV_DIR}"
fi

create_venv() {
  if python3 -m venv "${VENV_FLAGS[@]}" "${VENV_DIR}"; then
    return 0
  fi

  echo "python3 -m venv failed; installing a user-local virtualenv fallback."
  rm -rf "${VENV_DIR}"
  python3 -m pip install --user --upgrade virtualenv
  python3 -m virtualenv "${VENV_FLAGS[@]}" "${VENV_DIR}"
}

create_venv

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

BASE_PACKAGES=(
  'numpy>=1.21'
  'scipy>=1.7'
  'pyyaml>=5.4'
  'pupil-apriltags>=1.0.4'
)

if ! python - <<'PY' >/dev/null 2>&1
import cv2
PY
then
  BASE_PACKAGES+=('opencv-python>=4.5')
fi

python -m pip install --upgrade "${BASE_PACKAGES[@]}"

if (( WITH_SPACEMOUSE )); then
  python -m pip install pyspacemouse || {
    echo "Warning: pyspacemouse installation failed. Install host HID libraries and retry."
  }
fi

python - <<'PY'
import sys

modules = [
    ('cv2', 'cv2'),
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('yaml', 'yaml'),
    ('pupil_apriltags', 'pupil_apriltags'),
]

for import_name, label in modules:
    try:
        mod = __import__(import_name)
    except Exception as exc:
        print(f'{label}: FAIL ({exc})')
        sys.exit(1)
    print(f'{label}: OK ({getattr(mod, "__version__", "version-unknown")})')

try:
    import pyspacemouse
    print(f'pyspacemouse: OK ({getattr(pyspacemouse, "__version__", "version-unknown")})')
except Exception:
    print('pyspacemouse: not installed')
PY

cat <<EOF

arm_vision venv is ready at:
  ${VENV_DIR}

Activate it with:
  source ${VENV_DIR}/bin/activate

Then run:
  python arm_vision/main.py run --host 127.0.0.1 --show

EOF
