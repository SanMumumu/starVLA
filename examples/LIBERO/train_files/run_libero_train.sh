#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_YAML="${CONFIG_YAML:-examples/LIBERO/train_files/rynn_base_h8_current_dino_fullres_50k.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
DATA_ROOT="${LIBERO_DATA_ROOT:-}"
BASE_VLM="${RYNN_BASE_VLM:-}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-}"
RUN_ID="${RUN_ID:-}"
GLOBAL_BATCH_SIZE=128

case "${NUM_PROCESSES}" in
  16)
    MICRO_BATCH_SIZE=8
    GRAD_ACCUM_STEPS=1
    ;;
  *)
    echo "[LIBERO train][ERROR] this no-accumulation recipe requires one 16-GPU node; got NUM_PROCESSES=${NUM_PROCESSES}." >&2
    exit 1
    ;;
esac

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "[LIBERO train][ERROR] config does not exist: ${CONFIG_YAML}" >&2
  exit 1
fi
if [[ ! -f "starVLA/config/deepseeds/deepspeed_zero2_aidi_safe.yaml" ]]; then
  echo "[LIBERO train][ERROR] missing AIDI-safe DeepSpeed config" >&2
  exit 1
fi

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python \
  examples/LIBERO/train_files/verify_rynn_h8_recipe.py \
  --config "${CONFIG_YAML}" \
  --num-processes "${NUM_PROCESSES}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

args=(
  accelerate launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2_aidi_safe.yaml
  --num_processes "${NUM_PROCESSES}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
  --datasets.vla_data.per_device_batch_size "${MICRO_BATCH_SIZE}"
  --trainer.expected_global_batch_size "${GLOBAL_BATCH_SIZE}"
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
)
[[ -n "${DATA_ROOT}" ]] && args+=(--datasets.vla_data.data_root_dir "${DATA_ROOT}")
[[ -n "${BASE_VLM}" ]] && args+=(--framework.qwenvl.base_vlm "${BASE_VLM}")
[[ -n "${RUN_ROOT_DIR}" ]] && args+=(--run_root_dir "${RUN_ROOT_DIR}")
[[ -n "${RUN_ID}" ]] && args+=(--run_id "${RUN_ID}")

echo "[LIBERO train] config=${CONFIG_YAML} GPUs=${NUM_PROCESSES} H8 full-chunk"
echo "[LIBERO train] batch=${MICRO_BATCH_SIZE} x ${NUM_PROCESSES} x ${GRAD_ACCUM_STEPS} = ${GLOBAL_BATCH_SIZE}"
if [[ "${LIBERO_TRAIN_DRY_RUN:-0}" == "1" ]]; then
  printf '[LIBERO train][dry-run]'
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi
"${args[@]}"
