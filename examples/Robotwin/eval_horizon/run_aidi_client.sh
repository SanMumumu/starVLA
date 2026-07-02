#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CKPT="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt"
DEFAULT_HOST="CHANGE_ME_SERVER_HOST"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_CLIENTS="8"
DEFAULT_ROBOTWIN_PATH="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboTwin"
DEFAULT_MODES="demo_clean demo_randomized"
DEFAULT_TASKS="all"
DEFAULT_RUN_NAME="steps_140000_pytorch_model"
DEFAULT_SEED="0"

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
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-${DEFAULT_ROBOTWIN_PATH}}"
export MODES="${MODES:-${DEFAULT_MODES}}"
export TASKS="${TASKS:-${DEFAULT_TASKS}}"
export RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
export SEED="${SEED:-${DEFAULT_SEED}}"

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
echo "[AIDI CLIENT] ROBOTWIN_PATH=${ROBOTWIN_PATH}"
echo "[AIDI CLIENT] MODES=${MODES}"
echo "[AIDI CLIENT] TASKS=${TASKS}"
echo "[AIDI CLIENT] RUN_NAME=${RUN_NAME}"
echo "[AIDI CLIENT] SEED=${SEED}"

EVAL_CODE_DIR="${EVAL_CODE_DIR:-examples/Robotwin/eval_files}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"

ROBOTWIN_ALL_TASKS=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_block
  handover_mic
  hanging_mug
  lift_pot
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  pick_diverse_bottles
  pick_dual_bottles
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_can_basket
  place_cans_plasticbox
  place_container_plate
  place_dual_shoes
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  put_bottles_dustbin
  put_object_cabinet
  rotate_qrcode
  scan_object
  shake_bottle_horizontally
  shake_bottle
  stack_blocks_three
  stack_blocks_two
  stack_bowls_three
  stack_bowls_two
  stamp_seal
  turn_switch
)

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s\n' "${value}"
}

MODE_ITEMS=()
for MODE in ${MODES}; do
  MODE="$(trim "${MODE}")"
  [[ -n "${MODE}" ]] && MODE_ITEMS+=("${MODE}")
done

TASK_ITEMS=()
if [[ "${TASKS}" == "all" ]]; then
  TASK_ITEMS=("${ROBOTWIN_ALL_TASKS[@]}")
else
  for RAW_TASK in ${TASKS}; do
    IFS=',' read -ra SPLIT_TASKS <<< "${RAW_TASK}"
    for TASK in "${SPLIT_TASKS[@]}"; do
      TASK="$(trim "${TASK}")"
      [[ -n "${TASK}" ]] && TASK_ITEMS+=("${TASK}")
    done
  done
fi

if (( ${#MODE_ITEMS[@]} == 0 || ${#TASK_ITEMS[@]} == 0 )); then
  echo "[ROBOTWIN CLIENT][ERROR] empty MODES or TASKS"
  exit 2
fi

if [[ "${CKPT}" == *"/checkpoints/"* ]]; then
  MODEL_ROOT="$(echo "${CKPT}" | awk -F/checkpoints/ '{print $1}')"
else
  MODEL_ROOT="$(dirname "${CKPT}")"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_ROOT}/robotwin_eval_results_${RUN_NAME}_${NUM_CLIENTS}shard}"

export PYTHONPATH="${ROBOTWIN_PATH}:${STARVLA_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}/logs"

echo "[ROBOTWIN CLIENT] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[ROBOTWIN CLIENT] TASK_COUNT=${#TASK_ITEMS[@]}"

echo "[ROBOTWIN CLIENT] waiting for server ports..."
for ((i = 0; i < NUM_CLIENTS; i++)); do
  PORT=$((BASE_PORT + i))
  until timeout 2 bash -c "cat < /dev/null > /dev/tcp/${HOST}/${PORT}" 2>/dev/null; do
    echo "[ROBOTWIN CLIENT] waiting ${HOST}:${PORT}"
    sleep 3
  done
  echo "[ROBOTWIN CLIENT] port ready: ${HOST}:${PORT}"
done

PIDS=()
for ((SHARD = 0; SHARD < NUM_CLIENTS; SHARD++)); do
  (
    set -euo pipefail
    GPU_ID="${SHARD}"
    PORT="$((BASE_PORT + SHARD))"
    JOB_INDEX=0

    for MODE in "${MODE_ITEMS[@]}"; do
      mkdir -p "${OUTPUT_DIR}/logs/${MODE}"
      for TASK in "${TASK_ITEMS[@]}"; do
        if (( JOB_INDEX % NUM_CLIENTS == SHARD )); then
          SAFE_TASK="${TASK//\//_}"
          LOG_FILE="${OUTPUT_DIR}/logs/${MODE}/${SAFE_TASK}_shard${SHARD}_gpu${GPU_ID}_port${PORT}.log"
          echo "[ROBOTWIN CLIENT] shard=${SHARD} mode=${MODE} task=${TASK} log=${LOG_FILE}"
          bash "${EVAL_CODE_DIR}/eval.sh" \
            "${TASK}" \
            "${MODE}" \
            "${RUN_NAME}" \
            "${SEED}" \
            "${GPU_ID}" \
            "${CKPT}" \
            "${PORT}" \
            "${HOST}" > "${LOG_FILE}" 2>&1
        fi
        JOB_INDEX=$((JOB_INDEX + 1))
      done
    done
  ) &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAILED=1
  fi
done
[[ "${FAILED}" == "0" ]] || exit 1

"${ROBOTWIN_PYTHON}" examples/Robotwin/eval_horizon/aggregate_robotwin_logs.py \
  --output-dir "${OUTPUT_DIR}" \
  | tee "${OUTPUT_DIR}/aggregate.txt"
