#!/usr/bin/env bash
set -euo pipefail

# Edit this block before submitting the server job, or override the same names
# through the cluster environment if your submission flow supports it.
DEFAULT_CKPT="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_compare/0629_cmp_m3_qformer_best/final_model/pytorch_model.pt"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_SERVERS="8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-/running_package/starvla_dev/starVLA}"
if [[ ! -d "${STARVLA_DIR}" ]]; then
  STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

cd "${STARVLA_DIR}"

export CKPT="${CKPT:-${DEFAULT_CKPT}}"
export BASE_PORT="${BASE_PORT:-${DEFAULT_BASE_PORT}}"
export NUM_SERVERS="${NUM_SERVERS:-${DEFAULT_NUM_SERVERS}}"
export USE_BF16="${USE_BF16:-1}"
export STRIP_DINO_KEYS="${STRIP_DINO_KEYS:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-${STARVLA_DIR}/logs/libero_horizon_servers_${RUN_ID}}"

echo "[AIDI SERVER] STARVLA_DIR=${STARVLA_DIR}"
echo "[AIDI SERVER] CKPT=${CKPT}"
echo "[AIDI SERVER] BASE_PORT=${BASE_PORT}"
echo "[AIDI SERVER] NUM_SERVERS=${NUM_SERVERS}"
echo "[AIDI SERVER] LOG_DIR=${LOG_DIR}"

HOST_CANDIDATES="$(hostname -I 2>/dev/null | xargs || true)"
CLIENT_HOST="$(printf "%s\n" "${HOST_CANDIDATES}" | awk '{print $1}')"
LAST_PORT="$((BASE_PORT + NUM_SERVERS - 1))"
echo "================ AIDI SERVER CONNECTION INFO ================"
echo "CLIENT_HOST=${CLIENT_HOST}"
echo "BASE_PORT=${BASE_PORT}"
echo "PORT_RANGE=${BASE_PORT}-${LAST_PORT}"
echo "NUM_SERVERS=${NUM_SERVERS}"
echo "============================================================="

exec bash examples/LIBERO/eval_files/run_policy_servers_8.sh
