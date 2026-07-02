#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CKPT="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_compare/0629_cmp_m5_dual_gate_best/checkpoints/steps_10000_pytorch_model.pt"
DEFAULT_HOST="CHANGE_ME_SERVER_HOST"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_CLIENTS="8"
DEFAULT_SAVE_VIDEO="0"
DEFAULT_EXPECTED_TOTAL="10030"

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
export SAVE_VIDEO="${SAVE_VIDEO:-${DEFAULT_SAVE_VIDEO}}"
export EXPECTED_TOTAL="${EXPECTED_TOTAL:-${DEFAULT_EXPECTED_TOTAL}}"

if [[ -z "${HOST}" || "${HOST}" == "CHANGE_ME_SERVER_HOST" ]]; then
  echo "[AIDI CLIENT][ERROR] HOST is not configured."
  echo "Copy CLIENT_HOST from the server log into DEFAULT_HOST in run_aidi_client.sh."
  exit 2
fi

echo "[AIDI CLIENT] STARVLA_DIR=${STARVLA_DIR}"
echo "[AIDI CLIENT] CKPT=${CKPT}"
echo "[AIDI CLIENT] HOST=${HOST}"
echo "[AIDI CLIENT] BASE_PORT=${BASE_PORT}"
echo "[AIDI CLIENT] NUM_CLIENTS=${NUM_CLIENTS}"
echo "[AIDI CLIENT] SAVE_VIDEO=${SAVE_VIDEO}"
echo "[AIDI CLIENT] EXPECTED_TOTAL=${EXPECTED_TOTAL}"

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

LIBERO_HOME="${LIBERO_HOME:-/opt/LIBERO-plus}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/opt/LIBERO-plus/.libero_plus_config}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
EVAL_CODE_DIR="examples/LIBERO-plus/eval_horizon"
SEED="${SEED:-7}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-10030}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$(dirname "${CKPT}")")/libero_plus_eval_results_$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}" .pt)_${NUM_CLIENTS}shard}"

export LIBERO_HOME LIBERO_CONFIG_PATH
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/shards"
echo "[LIBERO-PLUS CLIENT] OUTPUT_DIR=${OUTPUT_DIR}"

echo "[LIBERO-PLUS CLIENT] waiting for server ports..."
for ((i = 0; i < NUM_CLIENTS; i++)); do
  PORT=$((BASE_PORT + i))
  until timeout 2 bash -c "cat < /dev/null > /dev/tcp/${HOST}/${PORT}" 2>/dev/null; do
    echo "[LIBERO-PLUS CLIENT] waiting ${HOST}:${PORT}"
    sleep 3
  done
  echo "[LIBERO-PLUS CLIENT] port ready: ${HOST}:${PORT}"
done

PIDS=()
for ((i = 0; i < NUM_CLIENTS; i++)); do
  PORT=$((BASE_PORT + i))
  LOG_FILE="${OUTPUT_DIR}/logs/shard_${i}.log"
  EXTRA=()
  [[ "${SAVE_VIDEO}" == "1" ]] && EXTRA+=(--save-video)

  (
    export CUDA_VISIBLE_DEVICES="${i}"
    export MUJOCO_EGL_DEVICE_ID=0
    "${LIBERO_PYTHON}" "${EVAL_CODE_DIR}/eval_libero_plus_sharded.py" \
      --host "${HOST}" \
      --port "${PORT}" \
      --ckpt "${CKPT}" \
      --output-dir "${OUTPUT_DIR}" \
      --shard-id "${i}" \
      --num-shards "${NUM_CLIENTS}" \
      --seed "${SEED}" \
      --expected-total "${EXPECTED_TOTAL}" \
      "${EXTRA[@]}"
  ) > "${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for ((i = 0; i < NUM_CLIENTS; i++)); do
  if wait "${PIDS[$i]}"; then
    echo "[LIBERO-PLUS CLIENT] shard ${i} finished"
  else
    echo "[LIBERO-PLUS CLIENT][ERROR] shard ${i} failed: ${OUTPUT_DIR}/logs/shard_${i}.log"
    FAILED=1
  fi
done
[[ "${FAILED}" == "0" ]] || exit 1

"${LIBERO_PYTHON}" "${EVAL_CODE_DIR}/aggregate_libero_plus_8.py" \
  --output-dir "${OUTPUT_DIR}" \
  --num-shards "${NUM_CLIENTS}" \
  --expected-total "${EXPECTED_TOTAL}" \
  | tee "${OUTPUT_DIR}/aggregate.txt"
