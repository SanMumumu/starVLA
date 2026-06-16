#!/usr/bin/env bash
set -euo pipefail

#######
# 中文注释：迁移后默认走 QwenGR00T 原生训练入口和 CED 迁移配置，不再启动独立 train_jointflow。
CONFIG="${1:-starVLA/jointflow/configs/jointflow_libero_ced.yaml}"
#######
[[ $# -gt 0 ]] && shift
#######
# 中文注释：使用 qwen3vl-gr00t 原生训练入口；JointFlow 逻辑由 framework.jointflow.enabled 开关激活。
TRAIN_ENTRY="starVLA/training/train_starvla.py"
#######
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
  "${TRAIN_ENTRY}" \
  --config_yaml "${CONFIG}" \
  "$@"
