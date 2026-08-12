#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_YAML="${CONFIG_YAML:-examples/Robotwin/train_files/rynn_base_h50_50k.yaml}"
NUM_MACHINES="${NUM_MACHINES:-${SLURM_NNODES:-8}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_MACHINES * GPUS_PER_NODE))}"
MACHINE_RANK="${MACHINE_RANK:-${SLURM_PROCID:-}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
DATA_ROOT="${ROBOTWIN_DATA_ROOT:-}"
DATASET_STATS="${ROBOTWIN_DATASET_STATS_PATH:-}"
BASE_VLM="${RYNN_BASE_VLM:-}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-}"
RUN_ID="${RUN_ID:-}"

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "[Robotwin train][ERROR] config does not exist: ${CONFIG_YAML}" >&2
  exit 1
fi
if [[ "${NUM_MACHINES}" != "8" || "${GPUS_PER_NODE}" != "8" || "${NUM_PROCESSES}" != "64" ]]; then
  echo "[Robotwin train][ERROR] this YAML is batch-locked to 8 nodes x 8 GPUs (16 x 64 = 1024)." >&2
  echo "Got NUM_MACHINES=${NUM_MACHINES}, GPUS_PER_NODE=${GPUS_PER_NODE}, NUM_PROCESSES=${NUM_PROCESSES}." >&2
  exit 1
fi
if [[ -z "${MACHINE_RANK}" || -z "${MASTER_ADDR}" ]]; then
  echo "[Robotwin train][ERROR] set MACHINE_RANK and MASTER_ADDR, or launch through run_robotwin_train_batch.sh under Slurm." >&2
  exit 1
fi

export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

args=(
  accelerate launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --main_process_ip "${MASTER_ADDR}"
  --main_process_port "${MASTER_PORT}"
  --machine_rank "${MACHINE_RANK}"
  --num_machines "${NUM_MACHINES}"
  --num_processes "${NUM_PROCESSES}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
)
[[ -n "${DATA_ROOT}" ]] && args+=(--datasets.vla_data.data_root_dir "${DATA_ROOT}")
if [[ -z "${DATASET_STATS}" && -n "${DATA_ROOT}" ]]; then
  DATASET_STATS="${DATA_ROOT%/}/dataset_stats.json"
fi
[[ -n "${DATASET_STATS}" ]] && args+=(--datasets.vla_data.fastwam_dataset_stats_path "${DATASET_STATS}")
[[ -n "${BASE_VLM}" ]] && args+=(--framework.qwenvl.base_vlm "${BASE_VLM}")
[[ -n "${RUN_ROOT_DIR}" ]] && args+=(--run_root_dir "${RUN_ROOT_DIR}")
[[ -n "${RUN_ID}" ]] && args+=(--run_id "${RUN_ID}")

echo "[Robotwin train] host=$(hostname) rank=${MACHINE_RANK}/${NUM_MACHINES} config=${CONFIG_YAML} GPUs=${NUM_PROCESSES} global_bs=1024 FastWAM_ABI H50 world_t+50"
"${args[@]}"
