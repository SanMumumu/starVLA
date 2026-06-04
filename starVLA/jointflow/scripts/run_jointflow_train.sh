#!/usr/bin/env bash
set -euo pipefail

CONFIG_YAML="${1:-starVLA/jointflow/configs/jointflow_libero.yaml}"
GPUS="${2:-${JOINTFLOW_GPUS:-4,7}}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NUM_PROCESSES="${JOINTFLOW_NUM_PROCESSES:-${#GPU_LIST[@]}}"
MAIN_PROCESS_PORT="${JOINTFLOW_MAIN_PROCESS_PORT:-29547}"
MIXED_PRECISION="${JOINTFLOW_MIXED_PRECISION:-bf16}"
WANDB_PROJECT="${JOINTFLOW_WANDB_PROJECT:-starVLA_JointFlow}"
WANDB_ENTITY="${JOINTFLOW_WANDB_ENTITY:-sanmumumu}"
RUN_ID="${JOINTFLOW_RUN_ID:-qwen_jointflow_libero_gpus_${GPUS//,/}}"
RUN_ROOT_DIR="${JOINTFLOW_RUN_ROOT_DIR:-playground/Checkpoints}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export JOINTFLOW_MIXED_PRECISION="${MIXED_PRECISION}"
export JOINTFLOW_FIND_UNUSED_PARAMETERS="${JOINTFLOW_FIND_UNUSED_PARAMETERS:-true}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export WANDB_MODE="${JOINTFLOW_WANDB_MODE:-online}"

echo "[jointflow] config=${CONFIG_YAML}"
echo "[jointflow] physical CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[jointflow] accelerate --gpu_ids=${GPUS} --num_processes=${NUM_PROCESSES}; logical ranks use cuda:0..$((NUM_PROCESSES - 1))"
echo "[jointflow] mixed_precision=${MIXED_PRECISION}"
echo "[jointflow] wandb mode=${WANDB_MODE} project=${WANDB_PROJECT} entity=${WANDB_ENTITY} run_id=${RUN_ID}"

ACCELERATE_ARGS=()
if (( NUM_PROCESSES > 1 )); then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  --gpu_ids "${GPUS}" \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  -m starVLA.jointflow.train.train_jointflow \
  --config_yaml "${CONFIG_YAML}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
