#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/rynn_base_h8_50k_fp32.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
DATA_ROOT="${LIBERO_DATA_ROOT:-}"
BASE_VLM="${RYNN_BASE_VLM:-}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-}"
RUN_ID="${RUN_ID:-}"

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "[LIBERO train][ERROR] config does not exist: ${CONFIG_YAML}" >&2
  exit 1
fi
if [[ "${NUM_PROCESSES}" != "8" ]]; then
  echo "[LIBERO train][ERROR] this YAML is batch-locked to one 8-GPU node (6 x 8 x 16 = 768); got NUM_PROCESSES=${NUM_PROCESSES}." >&2
  echo "Create a separate topology-specific YAML instead of silently changing the training contract." >&2
  exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

args=(
  accelerate launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
)
[[ -n "${DATA_ROOT}" ]] && args+=(--datasets.vla_data.data_root_dir "${DATA_ROOT}")
[[ -n "${BASE_VLM}" ]] && args+=(--framework.qwenvl.base_vlm "${BASE_VLM}")
[[ -n "${RUN_ROOT_DIR}" ]] && args+=(--run_root_dir "${RUN_ROOT_DIR}")
[[ -n "${RUN_ID}" ]] && args+=(--run_id "${RUN_ID}")

echo "[LIBERO train] config=${CONFIG_YAML} GPUs=${NUM_PROCESSES} H8 full-chunk"
"${args[@]}"
