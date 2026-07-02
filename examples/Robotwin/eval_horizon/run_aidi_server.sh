#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CKPT="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt"
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

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export LOG_DIR="${LOG_DIR:-${STARVLA_DIR}/logs/robotwin_horizon_servers_${RUN_ID}}"

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

mkdir -p "${LOG_DIR}"

PIDS=()
cleanup() {
  echo "[8x SERVER] stopping policy servers"
  for PID in "${PIDS[@]:-}"; do
    kill "${PID}" >/dev/null 2>&1 || true
  done
}
trap cleanup INT TERM EXIT

for IDX in $(seq 0 $((NUM_SERVERS - 1))); do
  GPU_ID="${IDX}"
  PORT="$((BASE_PORT + IDX))"
  LOG_FILE="${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log"

  echo "[8x SERVER] launch gpu=${GPU_ID}, port=${PORT}, log=${LOG_FILE}"
  (
    if [[ -z "${STARVLA_PYTHON:-}" ]]; then
      STARVLA_PYTHON="$(command -v python3 || command -v python)"
    fi
    CMD=("${STARVLA_PYTHON}" deployment/model_server/server_policy.py --ckpt_path "${CKPT}" --port "${PORT}")
    [[ "${USE_BF16}" == "1" ]] && CMD+=(--use_bf16)
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
  ) > "${LOG_FILE}" 2>&1 &

  PIDS+=("$!")
  sleep 2
done

echo "[8x SERVER] all launched. Keep this job alive while client evaluation is running."
wait
