#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CKPT="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_compare/0629_cmp_m5_dual_gate_best/checkpoints/steps_10000_pytorch_model.pt"
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
export LOG_DIR="${LOG_DIR:-${STARVLA_DIR}/logs/libero_plus_horizon_servers_${RUN_ID}}"

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

if [[ -z "${STARVLA_PYTHON:-}" ]]; then
  STARVLA_PYTHON="$(command -v python3 || command -v python)"
fi

if [[ "${STRIP_DINO_KEYS}" == "1" ]]; then
  CKPT_FOR_SERVER="$(
    CKPT="${CKPT}" "${STARVLA_PYTHON}" - <<'PYFILTER'
import hashlib
import os
import sys
from pathlib import Path

import torch

ckpt = Path(os.environ["CKPT"])
stat = ckpt.stat()
tag = hashlib.md5(f"{ckpt}:{stat.st_size}:{int(stat.st_mtime)}".encode()).hexdigest()[:12]
out_path = ckpt.parent / f"{ckpt.stem}_strip_dino_{tag}.pt"

if out_path.exists() and out_path.stat().st_size > 0:
    print(str(out_path))
    sys.exit(0)

obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
    sd = obj["state_dict"]
    nested = True
else:
    sd = obj
    nested = False

if not isinstance(sd, dict):
    print(str(ckpt))
    sys.exit(0)

drop_keys = [k for k in list(sd.keys()) if str(k).startswith("dino.") or str(k).startswith("module.dino.")]
if not drop_keys:
    print(str(ckpt))
    sys.exit(0)

filtered = {k: v for k, v in sd.items() if k not in drop_keys}
if nested:
    obj["state_dict"] = filtered
    torch.save(obj, str(out_path))
else:
    torch.save(filtered, str(out_path))
print(str(out_path))
PYFILTER
  )"
else
  CKPT_FOR_SERVER="${CKPT}"
fi

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
    CMD=("${STARVLA_PYTHON}" deployment/model_server/server_policy.py --ckpt_path "${CKPT_FOR_SERVER}" --port "${PORT}")
    [[ "${USE_BF16}" == "1" ]] && CMD+=(--use_bf16)
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
  ) > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
  sleep 2
done

echo "[8x SERVER] all launched. Keep this job alive while client evaluation is running."
wait
