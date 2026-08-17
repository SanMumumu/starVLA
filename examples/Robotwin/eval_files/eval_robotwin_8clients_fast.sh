#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:?Please set CKPT}"
HOST="${HOST:?Please set HOST}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:?Please set ROBOTWIN_PATH}"
BASE_PORT="${BASE_PORT:-6698}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
MODES="${MODES:-demo_clean demo_randomized}"
TASKS="${TASKS:-all}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-robotwin_full_eval}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-}"
SERVER_MAX_WAIT="${SERVER_MAX_WAIT:-1800}"
REPLAN_STEPS="${REPLAN_STEPS:-${ROBOTWIN_REPLAN_STEPS:-24}}"

# Final benchmark defaults. Only lower TEST_NUM / disable EXPERT_CHECK for smoke tests.
export ROBOTWIN_DISABLE_EVAL_VIDEO="${ROBOTWIN_DISABLE_EVAL_VIDEO:-1}"
export ROBOTWIN_SKIP_SAPIEN_TEST="${ROBOTWIN_SKIP_SAPIEN_TEST:-1}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-100}"
export ROBOTWIN_EXPERT_CHECK="${ROBOTWIN_EXPERT_CHECK:-1}"

ALL_TASKS=(
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
    click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block
    handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad
    move_playingcard_away move_stapler_pad open_laptop open_microwave
    pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
    place_bread_basket place_bread_skillet place_burger_fries place_can_basket
    place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
    place_fan place_mouse_pad place_object_basket place_object_scale
    place_object_stand place_phone_stand place_shoe press_stapler
    put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
    shake_bottle_horizontally shake_bottle stack_blocks_three stack_blocks_two
    stack_bowls_three stack_bowls_two stamp_seal turn_switch
)

[[ -f "${CKPT}" ]] || { echo "[ERROR] Missing checkpoint: ${CKPT}" >&2; exit 1; }

[[ -f "${ROBOTWIN_PATH}/script/eval_policy.py" ]] || {
    echo "[ERROR] Missing ${ROBOTWIN_PATH}/script/eval_policy.py" >&2; exit 1;
}
[[ "${NUM_CLIENTS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] NUM_CLIENTS must be a positive integer" >&2; exit 1;
}
[[ "${ROBOTWIN_TEST_NUM}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] ROBOTWIN_TEST_NUM must be a positive integer" >&2; exit 1;
}
[[ -z "${REPLAN_STEPS}" || "${REPLAN_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] REPLAN_STEPS must be a positive integer" >&2; exit 1;
}
export ROBOTWIN_REPLAN_STEPS="${REPLAN_STEPS}"

if ! grep -q 'ROBOTWIN_DISABLE_EVAL_VIDEO' "${ROBOTWIN_PATH}/script/eval_policy.py"; then
    echo "[ERROR] Fast-eval patch is not installed." >&2
    echo "Run: python3 patch_robotwin_fast_eval.py ${ROBOTWIN_PATH}/script/eval_policy.py" >&2
    exit 1
fi

# Prefer explicitly visible GPUs. Otherwise use 0..NUM_CLIENTS-1.
declare -a GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
    for ((i=0; i<NUM_CLIENTS; ++i)); do GPU_IDS+=("${i}"); done
fi
(( ${#GPU_IDS[@]} >= NUM_CLIENTS )) || {
    echo "[ERROR] Only ${#GPU_IDS[@]} visible GPUs for ${NUM_CLIENTS} clients" >&2; exit 1;
}
GPU_IDS=("${GPU_IDS[@]:0:NUM_CLIENTS}")

# AIDI bypasses some image entrypoints, so the first `python3` on PATH can be
# different from the environment used to build CuRobo. Resolve and validate an
# interpreter once, and serialize any JIT fallback before eight clients import
# the same extension concurrently.
source "${SCRIPT_DIR}/robotwin_runtime.sh"
prepare_robotwin_runtime "${ROBOTWIN_PATH}" "${GPU_IDS[0]}"

# Resolve tasks.
declare -a SELECTED_TASKS=()
if [[ "${TASKS}" == "all" ]]; then
    SELECTED_TASKS=("${ALL_TASKS[@]}")
elif [[ -f "${TASKS}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%%#*}"
        read -r line <<< "${line}"
        [[ -n "${line}" ]] && SELECTED_TASKS+=("${line}")
    done < "${TASKS}"
else
    TASKS="${TASKS//,/ }"
    read -r -a SELECTED_TASKS <<< "${TASKS}"
fi
(( ${#SELECTED_TASKS[@]} > 0 )) || { echo "[ERROR] No tasks selected" >&2; exit 1; }

read -r -a MODE_LIST <<< "${MODES}"
for mode in "${MODE_LIST[@]}"; do
    [[ "${mode}" == "demo_clean" || "${mode}" == "demo_randomized" ]] || {
        echo "[ERROR] Unsupported mode: ${mode}" >&2; exit 1;
    }
done

if [[ "${CKPT}" == *"/checkpoints/"* ]]; then
    MODEL_ROOT="$(dirname "$(dirname "${CKPT}")")"
else
    MODEL_ROOT="$(dirname "${CKPT}")"
fi
CKPT_STEM="$(basename "${CKPT}")"
CKPT_STEM="${CKPT_STEM%.*}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MODEL_ROOT}/robotwin_eval_results/${RUN_NAME}_${CKPT_STEM}_${TIMESTAMP}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
MANIFEST="${OUTPUT_ROOT}/manifest.tsv"
mkdir -p "${LOG_DIR}"

# All noisy/high-volume runtime output stays on local disk and is deleted.
LOCAL_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/robotwin_fast_eval.XXXXXX")"
RAW_LOG_DIR="${LOCAL_ROOT}/raw"
export ROBOTWIN_EVAL_RESULT_ROOT="${LOCAL_ROOT}/native"
mkdir -p "${RAW_LOG_DIR}" "${ROBOTWIN_EVAL_RESULT_ROOT}"

check_servers() {
    "${ROBOTWIN_PYTHON}" - "${HOST}" "${BASE_PORT}" "${NUM_CLIENTS}" <<'PY'
import socket, sys
host, base, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
missing = []
for port in range(base, base + count):
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError:
        missing.append(str(port))
if missing:
    print(",".join(missing))
    raise SystemExit(1)
PY
}

echo "========== RoboTwin fast result-only evaluation =========="
echo "[INFO] jobs:       $((${#SELECTED_TASKS[@]} * ${#MODE_LIST[@]}))"
echo "[INFO] clients:    ${NUM_CLIENTS}"
echo "[INFO] episodes:   ${ROBOTWIN_TEST_NUM} per task-mode"
echo "[INFO] videos:     disabled"
echo "[INFO] expert seed check: ${ROBOTWIN_EXPERT_CHECK}"
echo "[INFO] replan:     ${REPLAN_STEPS:-full model chunk}"
echo "[INFO] output:     ${OUTPUT_ROOT}"

deadline=$((SECONDS + SERVER_MAX_WAIT))
while ! missing="$(check_servers 2>&1)"; do
    (( SECONDS < deadline )) || { echo "[ERROR] Servers unavailable: ${missing}" >&2; exit 1; }
    sleep 2
done

# Flatten mode × task jobs.
declare -a JOB_MODES=() JOB_TASKS=()
for mode in "${MODE_LIST[@]}"; do
    for task in "${SELECTED_TASKS[@]}"; do
        JOB_MODES+=("${mode}")
        JOB_TASKS+=("${task}")
    done
done
TOTAL_JOBS=${#JOB_TASKS[@]}
printf 'job_id\tmode\ttask\tslot\tgpu\tport\tlog\n' > "${MANIFEST}"

declare -a ACTIVE_PIDS=() ACTIVE_JOB_IDS=() ACTIVE_RAW_LOGS=() ACTIVE_LOGS=()
declare -a FAILED_JOBS=()

kill_tree() {
    local pid="$1" sig="${2:-TERM}" child
    while read -r child; do [[ -n "${child}" ]] && kill_tree "${child}" "${sig}"; done \
        < <(ps -o pid= --ppid "${pid}" 2>/dev/null || true)
    kill -"${sig}" "${pid}" 2>/dev/null || true
}

cleanup() {
    trap - INT TERM EXIT
    for pid in "${ACTIVE_PIDS[@]:-}"; do [[ -n "${pid}" ]] && kill_tree "${pid}" TERM; done
    sleep 1
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && kill_tree "${pid}" KILL || true
    done
    rm -rf "${LOCAL_ROOT}"
}
trap cleanup INT TERM EXIT

launch_job() {
    local slot="$1" job_id="$2"
    local mode="${JOB_MODES[$job_id]}" task="${JOB_TASKS[$job_id]}"
    local gpu="${GPU_IDS[$slot]}" port=$((BASE_PORT + slot))
    local stem="job$(printf '%03d' "${job_id}")_${mode}_${task}_slot${slot}"
    local raw_log="${RAW_LOG_DIR}/${stem}.raw.log"
    local log="${LOG_DIR}/${stem}.log"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${job_id}" "${mode}" "${task}" "${slot}" "${gpu}" "${port}" "${log}" >> "${MANIFEST}"
    echo "[LAUNCH] ${job_id}/${TOTAL_JOBS} ${mode}/${task} -> slot${slot}"

    (
        export ROBOTWIN_PATH ROBOTWIN_PYTHON
        export ROBOTWIN_POLICY_NAME="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
        bash "${SCRIPT_DIR}/eval.sh" \
            "${task}" "${mode}" "${RUN_NAME}" "${SEED}" \
            "${gpu}" "${CKPT}" "${port}" "${HOST}"
    ) > "${raw_log}" 2>&1 &

    ACTIVE_PIDS[$slot]=$!
    ACTIVE_JOB_IDS[$slot]="${job_id}"
    ACTIVE_RAW_LOGS[$slot]="${raw_log}"
    ACTIVE_LOGS[$slot]="${log}"
}

compact_log() {
    local raw="$1" out="$2"
    local cleaned
    cleaned="$(sed -E $'s/\x1B\\[[0-9;?]*[ -\\/]*[@-~]//g' "${raw}")"
    printf '%s\n' "${cleaned}" \
        | grep -E 'Success rate|Traceback|ERROR|Error|Exception' > "${out}" || :
}

next_job=0
completed=0
while (( completed < TOTAL_JOBS )); do
    for ((slot=0; slot<NUM_CLIENTS; ++slot)); do
        pid="${ACTIVE_PIDS[$slot]:-}"
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            job_id="${ACTIVE_JOB_IDS[$slot]}"
            mode="${JOB_MODES[$job_id]}"
            task="${JOB_TASKS[$job_id]}"
            raw_log="${ACTIVE_RAW_LOGS[$slot]}"
            log="${ACTIVE_LOGS[$slot]}"

            status=0
            wait "${pid}" || status=$?
            compact_log "${raw_log}" "${log}"
            rm -f "${raw_log}"

            if (( status == 0 )); then
                final_rate="$(grep 'Success rate:' "${log}" | tail -n 1 || true)"
                echo "[DONE] ${mode}/${task} ${final_rate}"
            else
                FAILED_JOBS+=("${job_id}:${mode}:${task}:status=${status}")
                echo "[FAILED] ${mode}/${task}; log=${log}" >&2
            fi

            ACTIVE_PIDS[$slot]=""
            ACTIVE_JOB_IDS[$slot]=""
            ACTIVE_RAW_LOGS[$slot]=""
            ACTIVE_LOGS[$slot]=""
            completed=$((completed + 1))
        fi

        if [[ -z "${ACTIVE_PIDS[$slot]:-}" ]] && (( next_job < TOTAL_JOBS )); then
            launch_job "${slot}" "${next_job}"
            next_job=$((next_job + 1))
        fi
    done
    (( completed < TOTAL_JOBS )) && sleep 0.2 || true
done

trap - INT TERM EXIT
rm -rf "${LOCAL_ROOT}"

"${ROBOTWIN_PYTHON}" "${SCRIPT_DIR}/aggregate_robotwin_8clients.py" \
    --manifest "${MANIFEST}" --output-dir "${OUTPUT_ROOT}"

if (( ${#FAILED_JOBS[@]} > 0 )); then
    printf '%s\n' "${FAILED_JOBS[@]}" > "${OUTPUT_ROOT}/failed_jobs.txt"
    echo "[ERROR] ${#FAILED_JOBS[@]} jobs failed" >&2
    exit 1
fi

echo "[SUCCESS] ${OUTPUT_ROOT}/summary.txt"
