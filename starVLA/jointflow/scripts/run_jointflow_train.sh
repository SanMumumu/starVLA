#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-starVLA/jointflow/configs/jointflow_libero.yaml}"
[[ $# -gt 0 ]] && shift
GPUS="${1:-${JOINTFLOW_GPUS:-0}}"
[[ $# -gt 0 ]] && shift
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_GPUS="${#GPU_LIST[@]}"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "[jointflow] config=${CONFIG}"
echo "[jointflow] gpus=${GPUS} num_gpus=${NUM_GPUS}"

ACCELERATE_ARGS=()
if (( NUM_GPUS > 1 )); then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  --gpu_ids "${GPUS}" \
  --num_processes "${NUM_GPUS}" \
  --mixed_precision bf16 \
  --main_process_port "${MAIN_PROCESS_PORT:-29547}" \
  -m starVLA.jointflow.train.train_jointflow \
  --config_yaml "${CONFIG}" \
  "$@"
