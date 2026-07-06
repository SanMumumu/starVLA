#!/usr/bin/env bash
set -euo pipefail

# Edit this block after the server job starts. Copy HOST from the server log line
# "CLIENT_HOST=...". CKPT and BASE_PORT must match the server job.
DEFAULT_CKPT="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_compare/0629_cmp_m3_qformer_best/final_model/pytorch_model.pt"
DEFAULT_HOST="10.237.83.208"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_CLIENTS="8"
DEFAULT_TASK_SUITES="libero_spatial libero_object libero_goal libero_10"
DEFAULT_NUM_TRIALS_PER_TASK="50"
DEFAULT_MAX_TASKS="-1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-/running_package/starvla_dev/starVLA}"
if [[ ! -d "${STARVLA_DIR}" ]]; then
  STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

cd "${STARVLA_DIR}"

export CKPT="${CKPT:-${DEFAULT_CKPT}}"
export HOST="${HOST:-${DEFAULT_HOST}}"
export BASE_PORT="${BASE_PORT:-${DEFAULT_BASE_PORT}}"
export NUM_CLIENTS="${NUM_CLIENTS:-${DEFAULT_NUM_CLIENTS}}"
export TASK_SUITES="${TASK_SUITES:-${DEFAULT_TASK_SUITES}}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-${DEFAULT_NUM_TRIALS_PER_TASK}}"
export MAX_TASKS="${MAX_TASKS:-${DEFAULT_MAX_TASKS}}"

if [[ -z "${HOST}" || "${HOST}" == "CHANGE_ME_SERVER_HOST" ]]; then
  echo "[AIDI CLIENT][ERROR] HOST is not configured."
  echo "Set DEFAULT_HOST in examples/LIBERO/eval_horizon/run_aidi_client.sh"
  echo "or submit with HOST=<server-ip>. The server job prints CLIENT_HOST in its log."
  exit 2
fi

echo "[AIDI CLIENT] STARVLA_DIR=${STARVLA_DIR}"
echo "[AIDI CLIENT] CKPT=${CKPT}"
echo "[AIDI CLIENT] HOST=${HOST}"
echo "[AIDI CLIENT] BASE_PORT=${BASE_PORT}"
echo "[AIDI CLIENT] NUM_CLIENTS=${NUM_CLIENTS}"
echo "[AIDI CLIENT] TASK_SUITES=${TASK_SUITES}"
echo "[AIDI CLIENT] NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "[AIDI CLIENT] MAX_TASKS=${MAX_TASKS}"

exec bash examples/LIBERO/eval_files/eval_libero_8clients.sh
