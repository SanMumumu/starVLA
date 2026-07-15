#!/usr/bin/env bash

# Keep the first operation side-effect free so AIDI always has an immediate log.
echo "[AIDI] entry host=${HOSTNAME:-unknown} args=$*"

set -Eeuo pipefail
trap 'rc=$?; echo "[AIDI][ERROR] host=${HOSTNAME:-unknown} line=${BASH_LINENO[0]} rc=${rc}" >&2; exit "${rc}"' ERR

export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_SILENT=${WANDB_SILENT:-true}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-300}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Submission-host layout from 执行脚本/集群路径.txt:
#   starvla_dev/run_aidi_rbtw.sh
#   starvla_dev/starVLA/{examples,deployment,starVLA,...}
# Keep repository discovery compatible with an in-repo launcher too, but all
# workers must execute the launcher that actually exists in the uploaded tree.
if [[ -d "${SCRIPT_DIR}/examples" && -f "${SCRIPT_DIR}/starVLA/training/train_starvla.py" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
  PACKAGE_ROOT="$(dirname "${SCRIPT_DIR}")"
elif [[ -d "${SCRIPT_DIR}/starVLA/examples" && -f "${SCRIPT_DIR}/starVLA/starVLA/training/train_starvla.py" ]]; then
  PACKAGE_ROOT="${SCRIPT_DIR}"
  REPO_ROOT="${SCRIPT_DIR}/starVLA"
else
  PACKAGE_ROOT="${WORKING_PATH:-/running_package/starvla_dev}"
  REPO_ROOT="${PACKAGE_ROOT}/starVLA"
fi
[[ -d "${REPO_ROOT}/examples" && -f "${REPO_ROOT}/starVLA/training/train_starvla.py" ]] || {
  echo "[AIDI][ERROR] cannot locate StarVLA repository from launcher=${SCRIPT_DIR} working_path=${WORKING_PATH:-unset}" >&2
  exit 1
}
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
[[ -f "${SCRIPT_PATH}" ]] || {
  echo "[AIDI][ERROR] launcher does not exist: ${SCRIPT_PATH}" >&2
  exit 1
}
echo "[AIDI] layout package_root=${PACKAGE_ROOT} repo_root=${REPO_ROOT} launcher=${SCRIPT_PATH}"

worker_main() {
  local config_arg="$1"
  local config_yaml
  if [[ "${config_arg}" = /* ]]; then
    config_yaml="${config_arg}"
  else
    config_yaml="${REPO_ROOT}/${config_arg}"
  fi

  cd "${REPO_ROOT}"
  [[ -f "${config_yaml}" ]] || { echo "[AIDI][ERROR] missing config: ${config_yaml}"; exit 1; }

  local accel_config="${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2_aidi_safe.yaml"
  [[ -f "${accel_config}" ]] || { echo "[AIDI][ERROR] missing accelerate config: ${accel_config}"; exit 1; }

  local num_machines="${NUM_MACHINES:?NUM_MACHINES is required}"
  local machine_rank="${MACHINE_RANK:?MACHINE_RANK is required}"
  local master_addr="${MASTER_ADDR:?MASTER_ADDR is required}"
  local master_port="${MASTER_PORT:-29600}"
  local gpus_per_node="${GPUS_PER_NODE:-8}"
  local total_gpus=$((num_machines * gpus_per_node))

  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
  export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

  echo "[AIDI] worker host=${HOSTNAME:-unknown} rank=${machine_rank}/${num_machines}"
  echo "[AIDI] master=${master_addr}:${master_port} gpus=${gpus_per_node} config=${config_yaml}"

  exec accelerate launch \
    --config_file "${accel_config}" \
    --deepspeed_multinode_launcher standard \
    --main_process_ip "${master_addr}" \
    --main_process_port "${master_port}" \
    --machine_rank "${machine_rank}" \
    --num_machines "${num_machines}" \
    --num_processes "${total_gpus}" \
    starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}"
}

resolve_task() {
  local prefix="$1"
  local namespace="$2"
  local index="$3"
  local short="${prefix}-task-${index}"
  local fqdn="${short}.${prefix}.${namespace}.svc.cluster.local"

  if getent hosts "${fqdn}" >/dev/null 2>&1; then
    echo "${fqdn}"
  elif getent hosts "${short}" >/dev/null 2>&1; then
    echo "${short}"
  else
    echo "[AIDI][ERROR] cannot resolve task-${index}" >&2
    return 1
  fi
}

controller_main() {
  local config_arg="${1:?usage: run_aidi_rbtw.sh CONFIG_YAML}"
  local hostname_now="${HOSTNAME:-$(hostname)}"
  local current_index
  current_index="$(sed -n 's/.*-task-\([0-9]\+\)$/\1/p' <<<"${hostname_now}")"
  [[ -n "${current_index}" ]] || { echo "[AIDI][ERROR] cannot parse task index: ${hostname_now}"; exit 1; }

  local num_machines
  if [[ -n "${LAST_INDEX:-}" ]]; then
    num_machines=$((LAST_INDEX + 1))
  else
    # AIDI executes RUN_SCRIPTS on the last task, so its index is N-1.
    num_machines=$((current_index + 1))
  fi

  local prefix="${hostname_now%-task-*}"
  local namespace
  namespace="$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace)"
  local master_host
  master_host="$(resolve_task "${prefix}" "${namespace}" 0)"
  local master_addr
  master_addr="$(getent hosts "${master_host}" | awk 'NR == 1 {print $1}')"
  local master_port="${MASTER_PORT:-29600}"

  echo "[AIDI] controller host=${hostname_now} machines=${num_machines} master=${master_addr}:${master_port}"

  local pids=()
  local index host command
  for ((index = 0; index < num_machines; index++)); do
    host="$(resolve_task "${prefix}" "${namespace}" "${index}")"
    printf -v command \
      'export WORKING_PATH=%q NUM_MACHINES=%q MACHINE_RANK=%q MASTER_ADDR=%q MASTER_PORT=%q GPUS_PER_NODE=8 DINOV3_WEIGHTS=%q; bash %q --worker %q' \
      "${PACKAGE_ROOT}" "${num_machines}" "${index}" "${master_addr}" "${master_port}" \
      "${DINOV3_WEIGHTS:-}" "${SCRIPT_PATH}" "${config_arg}"

    echo "[AIDI] launch task-${index}: ${host}"
    if [[ "${index}" -eq "${current_index}" ]]; then
      bash -lc "${command}" &
    else
      ssh \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=20 \
        "${host}" "bash -lc $(printf '%q' "${command}")" &
    fi
    pids+=("$!")
  done

  local status=0 pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker_main "${1:?missing worker config}"
else
  controller_main "${1:?missing config}"
fi
