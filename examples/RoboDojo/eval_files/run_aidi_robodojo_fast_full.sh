#!/usr/bin/env bash
set -Eeuo pipefail

echo "[AIDI][RoboDojo-fast] entry host=${HOSTNAME:-unknown}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-/opt/robodojo-env/bin/python}"

export STARVLA_ROOT
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

# Original StarVLA rollout topology: one Isaac process per client GPU and
# vectorized environments inside that process. NUM_CLIENTS=8 therefore means
# exactly eight Isaac processes, never several Isaac processes on one GPU.
export NUM_CLIENTS="${NUM_CLIENTS:-8}"
[[ "${NUM_CLIENTS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[RoboDojo-fast][ERROR] NUM_CLIENTS must be a positive integer" >&2
  exit 2
}
if [[ -z "${CLIENT_CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CLIENT_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  else
    client_gpu_ids=""
    for ((gpu = 0; gpu < NUM_CLIENTS; ++gpu)); do
      [[ -z "${client_gpu_ids}" ]] || client_gpu_ids+=","
      client_gpu_ids+="${gpu}"
    done
    export CLIENT_CUDA_VISIBLE_DEVICES="${client_gpu_ids}"
  fi
fi
IFS=',' read -r -a configured_client_gpus <<< "${CLIENT_CUDA_VISIBLE_DEVICES}"
if (( ${#configured_client_gpus[@]} != NUM_CLIENTS )); then
  echo "[RoboDojo-fast][ERROR] NUM_CLIENTS=${NUM_CLIENTS}, but CLIENT_CUDA_VISIBLE_DEVICES exposes ${#configured_client_gpus[@]} entries" >&2
  exit 2
fi
export CLIENT_NUM_WORKERS=1
export TASKS_PER_GPU=1
# Clear variables from the discarded multi-Isaac/two-phase launcher so a
# reused terminal cannot accidentally reactivate that topology.
unset TASKS_PER_GPU_LAYOUT FIRST_PHASE_CONFIGS SECOND_PHASE_TASKS_PER_GPU_LAYOUT
export NUM_SERVERS="${NUM_SERVERS:-8}"
export ROBODOJO_ENVS_PER_CLIENT="${ROBODOJO_ENVS_PER_CLIENT:-6}"
export ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE="${ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE:-16}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-${ROBODOJO_REPLAN_STEPS:-12}}"
export ROBODOJO_REPLAN_STEPS="${N_ACTION_STEPS}"
export NUM_EPISODES="${NUM_EPISODES:-native}"
export SAVE_VIDEO=0
# Do not inherit a visualization choice from a reused client terminal.
export ROBODOJO_SAVE_MODE=none
export PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-60}"
export RESULT_DISCOVERY_TIMEOUT="${RESULT_DISCOVERY_TIMEOUT:-60}"
export SLOT_START_STAGGER_SECONDS="${SLOT_START_STAGGER_SECONDS:-2}"
export TASK_RETRY_BACKOFF_SECONDS="${TASK_RETRY_BACKOFF_SECONDS:-60}"

[[ -n "${STARVLA_CKPT_PATH:-}" ]] || {
  echo "[RoboDojo-fast][ERROR] export STARVLA_CKPT_PATH=/absolute/path/to/checkpoint.pt" >&2
  exit 2
}
[[ -n "${STARVLA_SERVER_HOST:-${HOST:-}}" ]] || {
  echo "[RoboDojo-fast][ERROR] export HOST=<policy-server-IP>" >&2
  exit 2
}
command -v "${ROBODOJO_PYTHON}" >/dev/null 2>&1 || [[ -x "${ROBODOJO_PYTHON}" ]] || {
  echo "[RoboDojo-fast][ERROR] RoboDojo Python is not executable: ${ROBODOJO_PYTHON}" >&2
  exit 1
}

exec "${ROBODOJO_PYTHON}" -u "${SCRIPT_DIR}/robodojo_fast_rollout.py"
