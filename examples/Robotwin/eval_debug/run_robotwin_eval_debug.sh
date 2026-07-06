#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_DIR="${REPO_ROOT}/examples/Robotwin/eval_files"

# Debug evaluation for RoboTwin WAM / baseline parity checks.
# This script is intentionally isolated under examples/Robotwin/eval_debug/.

CKPT="${CKPT:?Please set CKPT}"
HOST="${HOST:?Please set HOST}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:?Please set ROBOTWIN_PATH}"
BASE_PORT="${BASE_PORT:-6698}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
MODES="${MODES:-demo_clean demo_randomized}"
TASKS="${TASKS:-all}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-robotwin_debug}"
ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-5}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/examples/Robotwin/eval_debug/outputs/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${OUTPUT_ROOT}"
export OUTPUT_ROOT
export ROBOTWIN_EVAL_RESULT_ROOT="${OUTPUT_ROOT}/native"
export ROBOTWIN_DISABLE_EVAL_VIDEO="${ROBOTWIN_DISABLE_EVAL_VIDEO:-0}"
export ROBOTWIN_SKIP_SAPIEN_TEST="${ROBOTWIN_SKIP_SAPIEN_TEST:-0}"
export ROBOTWIN_EXPERT_CHECK="${ROBOTWIN_EXPERT_CHECK:-1}"
export ROBOTWIN_TEST_NUM
export ROBOTWIN_PATH
export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}"

cat <<EOF
[DEBUG] RoboTwin eval debug launch
[DEBUG] CKPT=${CKPT}
[DEBUG] HOST=${HOST}
[DEBUG] BASE_PORT=${BASE_PORT}
[DEBUG] NUM_CLIENTS=${NUM_CLIENTS}
[DEBUG] TASKS=${TASKS}
[DEBUG] MODES=${MODES}
[DEBUG] ROBOTWIN_TEST_NUM=${ROBOTWIN_TEST_NUM}
[DEBUG] OUTPUT_ROOT=${OUTPUT_ROOT}
EOF

env \
  CKPT="${CKPT}" \
  HOST="${HOST}" \
  ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
  BASE_PORT="${BASE_PORT}" \
  NUM_CLIENTS="${NUM_CLIENTS}" \
  MODES="${MODES}" \
  TASKS="${TASKS}" \
  SEED="${SEED}" \
  RUN_NAME="${RUN_NAME}" \
  ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM}" \
  ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash "${EVAL_DIR}/eval_robotwin_8clients_fast.sh"
