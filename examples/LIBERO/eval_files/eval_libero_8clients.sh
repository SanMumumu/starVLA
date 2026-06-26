#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"

LIBERO_HOME="${LIBERO_HOME:-/opt/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/usr/local/bin/python3}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"

CKPT="${CKPT:?Please set CKPT}"
HOST="${HOST:?Please set HOST to server task IP}"
BASE_PORT="${BASE_PORT:-6698}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"

TASK_SUITES="${TASK_SUITES:-${TASK_SUITE_NAME:-libero_spatial libero_object libero_goal libero_10}}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"

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

export LIBERO_HOME
export LIBERO_CONFIG_PATH
export PYTHONNOUSERSITE=1
export PATH=/usr/local/bin:${PATH}
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NO_ALBUMENTATIONS_UPDATE=1

# ---- Auto-fix LIBERO config ----
if [[ ! -s "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  echo "[WARN] LIBERO config missing or empty: ${LIBERO_CONFIG_PATH}/config.yaml"
  echo "[WARN] Falling back to /tmp/libero_config/config.yaml"

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

# ---- Patch PyTorch 2.6+ torch.load default for LIBERO init_states ----
mkdir -p /tmp/pydeps

cat > /tmp/pydeps/sitecustomize.py <<'SITECUSTOMIZE'
import torch as _torch

_orig_torch_load = _torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

_torch.load = _patched_torch_load
print("[sitecustomize] torch.load patched: weights_only=False by default")
SITECUSTOMIZE

export PYTHONPATH="/tmp/pydeps:${PYTHONPATH:-}"

echo "[8x CLIENT] STARVLA_DIR=${STARVLA_DIR}"
echo "[8x CLIENT] CKPT=${CKPT}"
echo "[8x CLIENT] HOST=${HOST}"
echo "[8x CLIENT] BASE_PORT=${BASE_PORT}"
echo "[8x CLIENT] NUM_CLIENTS=${NUM_CLIENTS}"
echo "[8x CLIENT] TASK_SUITES=${TASK_SUITES}"
echo "[8x CLIENT] NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "[8x CLIENT] MAX_TASKS=${MAX_TASKS}"
echo "[8x CLIENT] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[8x CLIENT] LIBERO_HOME=${LIBERO_HOME}"
echo "[8x CLIENT] LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "[8x CLIENT] LIBERO_PYTHON=${LIBERO_PYTHON}"

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

  echo "[8x CLIENT] launch shard=${IDX}/${NUM_CLIENTS}, gpu=${GPU_ID}, port=${PORT}, log=${LOG_FILE}"

  (
    # Client only runs LIBERO simulation + RPC; policy inference is on server.
    # This container exposes only one EGL device, so MuJoCo must use device 0.
    export CUDA_VISIBLE_DEVICES=0
    export MUJOCO_EGL_DEVICE_ID=0

    for SUITE in ${TASK_SUITES}; do
      VIDEO_OUT_PATH="${SHARD_OUT}/${SUITE}"
      mkdir -p "${VIDEO_OUT_PATH}"

      echo
      echo "============================================================"
      echo "[LIBERO EVAL SHARD] Running suite: ${SUITE}"
      echo "[LIBERO EVAL SHARD] SHARD=${IDX}/${NUM_CLIENTS}"
      echo "[LIBERO EVAL SHARD] GPU_ID=${GPU_ID}"
      echo "[LIBERO EVAL SHARD] HOST=${HOST}"
      echo "[LIBERO EVAL SHARD] PORT=${PORT}"
      echo "[LIBERO EVAL SHARD] VIDEO_OUT_PATH=${VIDEO_OUT_PATH}"
      echo "============================================================"

      "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero_sharded.py \
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
  sleep 1
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

echo "[8x CLIENT] all client shards finished. Aggregating..."

"${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/aggregate_libero_shards.py \
  --root "${OUTPUT_ROOT}" \
  --suites ${TASK_SUITES} \
  --num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --max-tasks "${MAX_TASKS}" \
  | tee "${OUTPUT_ROOT}/aggregate.txt"

exit "${FAIL}"
