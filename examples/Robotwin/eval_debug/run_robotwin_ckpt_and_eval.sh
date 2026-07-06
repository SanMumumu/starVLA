#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/examples/Robotwin/train_files/robotwin_wam_gate_rand2clean.yaml}"
CKPT="${CKPT:?Please set CKPT}"
HOST="${HOST:?Please set HOST}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:?Please set ROBOTWIN_PATH}"
PORT="${PORT:-6698}"

python "${SCRIPT_DIR}/check_robotwin_ckpt_load.py" --config "${CONFIG}" --ckpt "${CKPT}"
python "${SCRIPT_DIR}/check_robotwin_eval_input.py" --ckpt "${CKPT}" --host "${HOST}" --port "${PORT}"
