#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"

LIBERO_HOME="${LIBERO_HOME:-/opt/LIBERO}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/usr/local/bin/python3}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"

CKPT="${CKPT:-}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-6694}"

# TASK_SUITE_NAME=libero_goal bash examples/LIBERO/eval_files/eval_libero.sh
# TASK_SUITES="libero_spatial libero_object" bash examples/LIBERO/eval_files/eval_libero.sh
TASK_SUITES="${TASK_SUITES:-${TASK_SUITE_NAME:-libero_spatial libero_object libero_goal libero_10}}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_TASKS="${MAX_TASKS:--1}"

cd "${STARVLA_DIR}"

if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] set CKPT to the same checkpoint used by run_policy_server.sh" >&2
  exit 1
fi

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
# ---------------------------------------------------------------------

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

# .../0618_wam_exp3_policy_tricks/eval_results_steps_40000_pytorch_model/libero_goal/
OUTPUT_ROOT="${OUTPUT_ROOT:-${MODEL_ROOT}/eval_results_${CKPT_TAG}}"

echo "[LIBERO EVAL] CKPT=${CKPT}"
echo "[LIBERO EVAL] MODEL_ROOT=${MODEL_ROOT}"
echo "[LIBERO EVAL] CKPT_TAG=${CKPT_TAG}"
echo "[LIBERO EVAL] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[LIBERO EVAL] HOST=${HOST}"
echo "[LIBERO EVAL] PORT=${PORT}"
echo "[LIBERO EVAL] TASK_SUITES=${TASK_SUITES}"
echo "[LIBERO EVAL] NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "[LIBERO EVAL] MAX_TASKS=${MAX_TASKS}"
echo "[LIBERO EVAL] LIBERO_HOME=${LIBERO_HOME}"
echo "[LIBERO EVAL] LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "[LIBERO EVAL] LIBERO_PYTHON=${LIBERO_PYTHON}"

mkdir -p "${OUTPUT_ROOT}"

for SUITE in ${TASK_SUITES}; do
  VIDEO_OUT_PATH="${OUTPUT_ROOT}/${SUITE}"
  mkdir -p "${VIDEO_OUT_PATH}"

  echo
  echo "============================================================"
  echo "[LIBERO EVAL] Running suite: ${SUITE}"
  echo "[LIBERO EVAL] VIDEO_OUT_PATH=${VIDEO_OUT_PATH}"
  echo "============================================================"

  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${CKPT}" \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.task-suite-name "${SUITE}" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.max-tasks "${MAX_TASKS}" \
    --args.video-out-path "${VIDEO_OUT_PATH}" 2>&1 | tee "${VIDEO_OUT_PATH}/eval.log"
done
