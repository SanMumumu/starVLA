#!/usr/bin/env bash
set -euo pipefail

# Edit this block after the server job starts. Copy HOST from the server log line
# "CLIENT_HOST=...". CKPT and BASE_PORT must match the server job.
DEFAULT_CKPT="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_libero/0623_wam_exp6_2b_bigbs_policy_fdm_0.2/final_model/pytorch_model.pt"
DEFAULT_HOST="CHANGE_ME_SERVER_HOST"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_CLIENTS="8"
DEFAULT_TASK_SUITES="libero_spatial libero_object libero_goal libero_10"
DEFAULT_NUM_TRIALS_PER_TASK="50"
DEFAULT_MAX_TASKS="-1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="${STARVLA_DIR:-/running_package/starvla_dev/starVLA}"
if [[ ! -d "${STARVLA_DIR}" ]]; then
  STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

cd "${STARVLA_DIR}"

export CKPT="${CKPT:-${DEFAULT_CKPT}}"
export HOST="${HOST:-${DEFAULT_HOST}}"
export BASE_PORT="${BASE_PORT:-${DEFAULT_BASE_PORT}}"
export NUM_CLIENTS="${NUM_CLIENTS:-${DEFAULT_NUM_CLIENTS}}"
export TASK_SUITES="${TASK_SUITES:-${DEFAULT_TASK_SUITES}}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-${DEFAULT_NUM_TRIALS_PER_TASK}}"
export MAX_TASKS="${MAX_TASKS:-${DEFAULT_MAX_TASKS}}"

if [[ -z "${HOST}" || "${HOST}" == "CHANGE_ME_SERVER_HOST" ]]; then
  echo "[AIDI CLIENT][ERROR] HOST is not configured."
  echo "Set DEFAULT_HOST in examples/LIBERO/eval_horizon/run_aidi_client.sh"
  echo "or submit with HOST=<server-ip>. The server job prints CLIENT_HOST in its log."
  exit 2
fi

echo "[AIDI CLIENT] STARVLA_DIR=${STARVLA_DIR}"
echo "[AIDI CLIENT] CKPT=${CKPT}"
echo "[AIDI CLIENT] HOST=${HOST}"
echo "[AIDI CLIENT] BASE_PORT=${BASE_PORT}"
echo "[AIDI CLIENT] NUM_CLIENTS=${NUM_CLIENTS}"
echo "[AIDI CLIENT] TASK_SUITES=${TASK_SUITES}"
echo "[AIDI CLIENT] NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "[AIDI CLIENT] MAX_TASKS=${MAX_TASKS}"

LIBERO_HOME="${LIBERO_HOME:-/opt/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/usr/local/bin/python3}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"

if [[ "${CKPT}" == *"/final_model/"* ]]; then
  MODEL_ROOT="$(dirname "$(dirname "${CKPT}")")"
  CKPT_TAG="final_model_$(basename "${CKPT}" .pt)"
elif [[ "${CKPT}" == *"/checkpoints/"* ]]; then
  MODEL_ROOT="$(echo "${CKPT}" | awk -F/checkpoints/ '{print $1}')"
  CKPT_TAG="$(basename "${CKPT}" .pt)"
else
  MODEL_ROOT="$(dirname "${CKPT}")"
  CKPT_TAG="$(basename "${CKPT}" .pt)"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${MODEL_ROOT}/eval_results_${CKPT_TAG}_8shard}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/client_logs}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

export LIBERO_HOME LIBERO_CONFIG_PATH
export PYTHONNOUSERSITE=1
export PATH=/usr/local/bin:${PATH}
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NO_ALBUMENTATIONS_UPDATE=1

if [[ ! -s "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  mkdir -p /tmp/libero_config
  cat > /tmp/libero_config/config.yaml <<'YAML'
benchmark_root: /opt/LIBERO/libero/libero
bddl_files: /opt/LIBERO/libero/libero/bddl_files
init_states: /opt/LIBERO/libero/libero/init_files
datasets: /opt/LIBERO/libero/datasets
assets: /opt/LIBERO/libero/libero/assets
YAML
  export LIBERO_CONFIG_PATH=/tmp/libero_config
fi

mkdir -p /tmp/pydeps
cat > /tmp/pydeps/sitecustomize.py <<'SITECUSTOMIZE'
import torch as _torch

_orig_torch_load = _torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

_torch.load = _patched_torch_load
SITECUSTOMIZE
export PYTHONPATH="/tmp/pydeps:${PYTHONPATH:-}"

echo "[8x CLIENT] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[8x CLIENT] waiting for server ports..."
for IDX in $(seq 0 $((NUM_CLIENTS - 1))); do
  PORT="$((BASE_PORT + IDX))"
  until timeout 2 bash -c "cat < /dev/null > /dev/tcp/${HOST}/${PORT}" 2>/dev/null; do
    echo "[8x CLIENT] waiting ${HOST}:${PORT}"
    sleep 3
  done
  echo "[8x CLIENT] port ready: ${HOST}:${PORT}"
done

PIDS=()
for IDX in $(seq 0 $((NUM_CLIENTS - 1))); do
  GPU_ID="${IDX}"
  PORT="$((BASE_PORT + IDX))"
  SHARD_OUT="${OUTPUT_ROOT}/shard_${IDX}"
  LOG_FILE="${LOG_DIR}/client_shard${IDX}_gpu${GPU_ID}_port${PORT}.log"
  mkdir -p "${SHARD_OUT}"

  (
    export CUDA_VISIBLE_DEVICES="${CLIENT_CUDA_VISIBLE_DEVICES:-0}"
    export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

    for SUITE in ${TASK_SUITES}; do
      VIDEO_OUT_PATH="${SHARD_OUT}/${SUITE}"
      mkdir -p "${VIDEO_OUT_PATH}"
      echo "[LIBERO EVAL SHARD] suite=${SUITE} shard=${IDX}/${NUM_CLIENTS} server=${HOST}:${PORT}"
      "${LIBERO_PYTHON}" ./examples/LIBERO/eval_horizon/eval_libero_sharded.py \
        --args.pretrained-path "${CKPT}" \
        --args.host "${HOST}" \
        --args.port "${PORT}" \
        --args.task-suite-name "${SUITE}" \
        --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
        --args.max-tasks "${MAX_TASKS}" \
        --args.task-shard-index "${IDX}" \
        --args.num-task-shards "${NUM_CLIENTS}" \
        --args.video-out-path "${VIDEO_OUT_PATH}" 2>&1 | tee "${VIDEO_OUT_PATH}/eval.log"
    done
  ) > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_horizon/aggregate_libero_shards.py \
  --root "${OUTPUT_ROOT}" \
  --suites ${TASK_SUITES} \
  --num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --max-tasks "${MAX_TASKS}" \
  | tee "${OUTPUT_ROOT}/aggregate.txt"

exit "${FAIL}"
