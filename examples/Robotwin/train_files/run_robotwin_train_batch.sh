#!/usr/bin/env bash
#SBATCH --job-name=rynn_robotwin_h50_bs1024
#SBATCH --partition=ebench_t
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${SLURM_NNODES:?Submit this script with sbatch}" != "8" ]]; then
  echo "[Robotwin train][ERROR] expected 8 Slurm nodes, got ${SLURM_NNODES}" >&2
  exit 1
fi

export NUM_MACHINES=8
export GPUS_PER_NODE=8
export NUM_PROCESSES=64
if [[ -z "${MASTER_ADDR:-}" ]]; then
  read -r MASTER_ADDR < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"

# Cluster-specific defaults remain overridable at sbatch submission time.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_4,mlx5_5}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

echo "[Robotwin train] nodes=${NUM_MACHINES} gpus_per_node=${GPUS_PER_NODE} total_gpus=${NUM_PROCESSES}"
echo "[Robotwin train] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[Robotwin train] data=${ROBOTWIN_DATA_ROOT:-YAML default} output=${RUN_ROOT_DIR:-YAML default}"

# One launcher task per node; Accelerate starts eight local workers per task.
srun \
  --nodes="${NUM_MACHINES}" \
  --ntasks="${NUM_MACHINES}" \
  --ntasks-per-node=1 \
  bash "${REPO_ROOT}/examples/Robotwin/train_files/run_robotwin_train.sh"
