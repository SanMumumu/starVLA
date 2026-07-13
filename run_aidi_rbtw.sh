#!/bin/bash
set -euo pipefail

# ============================================================
# AIDI RoboTwin launcher
# AIDI usually runs RUN_SCRIPTS on the last task.
# Modes:
#   controller: run on last task, ssh fan-out to all tasks
#   worker:     run on each task, start accelerate multi-node
# ============================================================

locate_repo() {
  if [ -n "${WORKING_PATH:-}" ] && [ -d "${WORKING_PATH}/starVLA" ]; then
    echo "${WORKING_PATH}/starVLA"
  elif [ -d /running_package/starvla_dev/starVLA ]; then
    echo /running_package/starvla_dev/starVLA
  elif [ -d /running_package/starvla/starVLA ]; then
    echo /running_package/starvla/starVLA
  elif [ -d /running_package/starvla ]; then
    echo /running_package/starvla
  else
    echo "ERROR_NO_REPO"
  fi
}

resolve_task() {
  local prefix="$1"
  local ns="$2"
  local i="$3"

  local short="${prefix}-task-${i}"
  local fqdn="${prefix}-task-${i}.${prefix}.${ns}.svc.cluster.local"

  if getent hosts "${fqdn}" >/dev/null 2>&1; then
    echo "${fqdn}"
    return 0
  fi

  if getent hosts "${short}" >/dev/null 2>&1; then
    echo "${short}"
    return 0
  fi

  return 1
}

worker_main() {
  CONFIG_ARG="$1"

  echo "========== AIDI StarVLA RoboTwin WORKER =========="
  echo "Host: $(hostname)"
  echo "Date: $(date)"
  echo "Args: ${CONFIG_ARG}"
  echo "WORKING_PATH=${WORKING_PATH:-unset}"
  echo "[rbtw][env] $(env | grep -iE 'rank|world|master|node|worker|machine|nnode|addr|port|mlp|arnold|hostname' | sort | tr '\n' ' ')"

  REPO_ROOT="$(locate_repo)"
  if [ "${REPO_ROOT}" = "ERROR_NO_REPO" ]; then
    echo "ERROR: cannot locate repo"
    exit 1
  fi

  cd "${REPO_ROOT}"

  if [[ "${CONFIG_ARG}" = /* ]]; then
    CONFIG_YAML="${CONFIG_ARG}"
  else
    CONFIG_YAML="${REPO_ROOT}/${CONFIG_ARG}"
  fi

  echo "REPO_ROOT=${REPO_ROOT}"
  echo "CONFIG_YAML=${CONFIG_YAML}"
  echo "PWD=$(pwd)"
  echo "Python=$(which python)"

  if [ ! -f "${CONFIG_YAML}" ]; then
    echo "ERROR: config yaml not found: ${CONFIG_YAML}"
    exit 1
  fi

  # DINOv3 must be loaded from the shared bucket on offline AIDI workers.  An
  # explicit job-level value is propagated by controller_main below.  Refuse a
  # bad path here instead of letting transformers fall back to Hugging Face.
  if [ -n "${DINOV3_WEIGHTS:-}" ]; then
    if [ ! -d "${DINOV3_WEIGHTS}" ]; then
      echo "ERROR: local DINOv3 HF snapshot directory not found: ${DINOV3_WEIGHTS}"
      exit 1
    fi
    if [ ! -f "${DINOV3_WEIGHTS%/}/config.json" ]; then
      echo "ERROR: local DINOv3 HF snapshot is incomplete (missing config.json): ${DINOV3_WEIGHTS}"
      exit 1
    fi
    DINO_WEIGHT_FILE=$(find "${DINOV3_WEIGHTS%/}" -maxdepth 1 -type f \
      \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +100M -print -quit)
    if [ -z "${DINO_WEIGHT_FILE}" ]; then
      echo "ERROR: local DINOv3 HF snapshot has no weight file larger than 100 MiB."
      echo "       model.safetensors may be missing or only a Git-LFS pointer: ${DINOV3_WEIGHTS}"
      exit 1
    fi
    python - "${DINOV3_WEIGHTS%/}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
hidden_size = config.get("hidden_size")
if hidden_size != 1024:
    raise SystemExit(f"ERROR: expected DINOv3 ViT-L hidden_size=1024, got {hidden_size!r} in {sys.argv[1]}")
print(f"DINO config PASS: model_type={config.get('model_type')}, hidden_size={hidden_size}")
PY
    export DINOV3_WEIGHTS
    echo "DINO weights PASS: ${DINO_WEIGHT_FILE}"
    echo "DINOV3_WEIGHTS=${DINOV3_WEIGHTS} (local HF snapshot; network fallback disabled by loader)"
  fi

  # Controlled RoboTwin WAM ablations must stay aligned with the completed
  # QwenGR00T corrnoise baseline. Fail before allocating the training graph if
  # a config drifts in sampling, horizon, optimizer, loader, or prompt fields.
  case "${CONFIG_YAML##*/}" in
    robotwin_wam_*.yaml)
      python examples/Robotwin/train_files/verify_wam_corrnoise_alignment.py \
        --check-files \
        --target "${CONFIG_YAML##*/}"
      python examples/Robotwin/train_files/verify_fastwam_robotwin_data.py \
        --config-yaml "${CONFIG_YAML}"
      ;;
    starvla_qwengroot_robotwin_fastwam_corrnoise.yaml)
      python examples/Robotwin/train_files/verify_fastwam_starvla_alignment.py
      python examples/Robotwin/train_files/verify_fastwam_robotwin_data.py \
        --config-yaml "${CONFIG_YAML}"
      ;;
  esac

  python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
PY

  GPUS_PER_NODE=${GPUS_PER_NODE:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

  HOSTNAME_NOW="$(hostname)"
  AUTO_RANK="$(echo "${HOSTNAME_NOW}" | sed -n 's/.*-task-\([0-9]\+\)$/\1/p')"
  if [ -z "${AUTO_RANK}" ]; then
    echo "ERROR: cannot parse machine rank from hostname=${HOSTNAME_NOW}"
    exit 1
  fi

  if [ -z "${NUM_MACHINES:-}" ] && [ -n "${WORLD_SIZE:-}" ] && [ "${WORLD_SIZE}" -gt "${GPUS_PER_NODE}" ]; then
    NUM_MACHINES=$(( WORLD_SIZE / GPUS_PER_NODE ))
  fi

  NUM_MACHINES=${NUM_MACHINES:?NUM_MACHINES missing: controller did not inject it and WORLD_SIZE is not enough to infer it}
  MACHINE_RANK=${MACHINE_RANK:-${NODE_RANK:-${GROUP_RANK:-${AUTO_RANK}}}}
  MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR missing}
  MASTER_PORT=${MASTER_PORT:-29600}
  TOTAL_GPUS=$(( NUM_MACHINES * GPUS_PER_NODE ))

  echo "NUM_MACHINES=${NUM_MACHINES}"
  echo "MACHINE_RANK=${MACHINE_RANK}"
  echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "TOTAL_GPUS=${TOTAL_GPUS}"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"

  # ============================================================
  # NCCL / network
  # ============================================================
  # Default to IB/RoCE. Do not hard-pin NCCL_IB_HCA here; allow manual override.
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

  # Socket iface is mainly for rendezvous/bootstrap. Keep eth0 unless the platform needs otherwise.
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}

  # PyTorch NCCL failure handling. Use TORCH_* names to avoid deprecated NCCL_* warnings.
  export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}

  export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
  export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-3600000}

  # Default WARN for real training. To verify IB once, launch with NCCL_DEBUG=INFO.
  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
  export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}

  # Avoid slow online version-check warning in offline cluster.
  export ALBUMENTATIONS_DISABLE_VERSION_CHECK=${ALBUMENTATIONS_DISABLE_VERSION_CHECK:-1}
  # Long-lived WAM runs allocate different Qwen/action/world tensors by task;
  # expandable segments prevent small tail allocations from failing after the
  # allocator becomes segmented. This changes allocation only, not numerics.
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

  echo "========== NCCL SETTINGS =========="
  echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
  echo "NCCL_IB_HCA=${NCCL_IB_HCA:-auto}"
  echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
  echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
  echo "NCCL_DEBUG=${NCCL_DEBUG}"
  echo "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
  echo "TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT}"
  echo "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
  echo "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"
  echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "==================================="

  # ============================================================
  # Data path compatibility
  # ============================================================
  ROBOTWIN_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/DATA/RoboTwin
  if [ -d "${ROBOTWIN_ROOT}/Clean" ] && [ ! -e "${ROBOTWIN_ROOT}/clean" ]; then
    ln -s "${ROBOTWIN_ROOT}/Clean" "${ROBOTWIN_ROOT}/clean" || true
  fi
  if [ -d "${ROBOTWIN_ROOT}/Randomized" ] && [ ! -e "${ROBOTWIN_ROOT}/randomized" ]; then
    ln -s "${ROBOTWIN_ROOT}/Randomized" "${ROBOTWIN_ROOT}/randomized" || true
  fi

  ACCEL_CONFIG="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2_aidi_safe.yaml"
  if [ ! -f "${ACCEL_CONFIG}" ]; then
    echo "ERROR: accelerate config not found: ${ACCEL_CONFIG}"
    exit 1
  fi

  echo "ACCEL_CONFIG=${ACCEL_CONFIG}"

  export WANDB_MODE=${WANDB_MODE:-offline}
  export WANDB_SILENT=${WANDB_SILENT:-true}
  export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}
  echo "WANDB_MODE=${WANDB_MODE}"

  echo "========== ACCELERATE STANDARD MULTI-NODE LAUNCH =========="
  echo "NUM_MACHINES=${NUM_MACHINES}"
  echo "MACHINE_RANK=${MACHINE_RANK}"
  echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "TOTAL_GPUS=${TOTAL_GPUS}"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "==========================================================="

  accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --deepspeed_multinode_launcher standard \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --machine_rank "${MACHINE_RANK}" \
    --num_machines "${NUM_MACHINES}" \
    --num_processes "${TOTAL_GPUS}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}"
}

controller_main() {
  if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_yaml_relative_to_repo>"
    exit 1
  fi

  CONFIG_ARG="$1"

  echo "========== AIDI StarVLA RoboTwin CONTROLLER =========="
  echo "Host: $(hostname)"
  echo "Date: $(date)"
  echo "Args: ${CONFIG_ARG}"
  echo "WORKING_PATH=${WORKING_PATH:-unset}"
  echo "LAST_INDEX=${LAST_INDEX:-unset}"
  echo "[rbtw][env] $(env | grep -iE 'rank|world|master|node|worker|machine|nnode|addr|port|mlp|arnold|hostname|last_index' | sort | tr '\n' ' ')"

  HOSTNAME_NOW="$(hostname)"
  PREFIX="$(echo "${HOSTNAME_NOW}" | sed 's/-task-[0-9]\+$//')"
  NS="$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null || echo project-h20-horizon-labs-acloud)"

  if [ -n "${LAST_INDEX:-}" ]; then
    NUM_MACHINES=$(( LAST_INDEX + 1 ))
  else
    NUM_MACHINES=0
    for i in $(seq 0 127); do
      if resolve_task "${PREFIX}" "${NS}" "${i}" >/dev/null 2>&1; then
        NUM_MACHINES=$(( NUM_MACHINES + 1 ))
      else
        break
      fi
    done
  fi

  if [ "${NUM_MACHINES}" -le 1 ]; then
    echo "ERROR: NUM_MACHINES=${NUM_MACHINES}"
    exit 1
  fi

  MASTER_HOST="$(resolve_task "${PREFIX}" "${NS}" 0)"
  MASTER_IP="$(getent hosts "${MASTER_HOST}" | awk '{print $1}' | head -n 1)"
  MASTER_PORT=${MASTER_PORT:-29600}

  if [ -z "${MASTER_IP}" ]; then
    echo "ERROR: cannot resolve MASTER_HOST=${MASTER_HOST}"
    exit 1
  fi

  echo "PREFIX=${PREFIX}"
  echo "NS=${NS}"
  echo "NUM_MACHINES=${NUM_MACHINES}"
  echo "MASTER_HOST=${MASTER_HOST}"
  echo "MASTER_IP=${MASTER_IP}"
  echo "MASTER_PORT=${MASTER_PORT}"

  ROOT="${WORKING_PATH:-/running_package/starvla_dev}"
  Q_ROOT=$(printf "%q" "${ROOT}")
  Q_CONFIG=$(printf "%q" "${CONFIG_ARG}")
  Q_MASTER_IP=$(printf "%q" "${MASTER_IP}")
  Q_DINOV3_WEIGHTS=$(printf "%q" "${DINOV3_WEIGHTS:-}")

  echo "========== SSH fan-out =========="

  pids=()

  for i in $(seq 0 $(( NUM_MACHINES - 1 ))); do
    HOST="$(resolve_task "${PREFIX}" "${NS}" "${i}")"
    echo "Launching worker on task-${i}: ${HOST}"

    CMD="cd ${Q_ROOT} && export NUM_MACHINES=${NUM_MACHINES} MACHINE_RANK=${i} MASTER_ADDR=${Q_MASTER_IP} MASTER_PORT=${MASTER_PORT} GPUS_PER_NODE=8 DINOV3_WEIGHTS=${Q_DINOV3_WEIGHTS} && bash ${Q_ROOT}/run_aidi_rbtw.sh --worker ${Q_CONFIG}"

    if [ "${HOST}" = "${HOSTNAME_NOW}" ] || [ "${HOST}" = "$(hostname)" ]; then
      bash -lc "${CMD}" &
    else
      ssh \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=20 \
        "${HOST}" \
        "bash -lc $(printf "%q" "${CMD}")" &
    fi

    pids+=("$!")
  done

  echo "Launched ${#pids[@]} workers. Waiting..."

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done

  exit "${fail}"
}

_NGPU_NOW="$(nvidia-smi -L 2>/dev/null | wc -l)"

if [ "${1:-}" = "--worker" ]; then
  shift
  worker_main "$@"
elif [ -n "${MASTER_ADDR:-}" ] && { { [ "${NUM_MACHINES:-1}" -gt 1 ] 2>/dev/null; } || { [ -n "${WORLD_SIZE:-}" ] && [ "${WORLD_SIZE}" -gt "${_NGPU_NOW:-8}" ] 2>/dev/null; }; }; then
  echo "[rbtw] platform injected multi-node env -> direct worker mode, skip ssh fan-out"
  worker_main "$@"
else
  controller_main "$@"
fi
