#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CKPT="${CKPT:?Please set CKPT}"
HOST="${HOST:?Please set HOST to the policy-server job IP}"
BASE_PORT="${BASE_PORT:-6698}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
MODES="${MODES:-demo_clean demo_randomized}"
TASKS="${TASKS:-all}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-robotwin_full_eval}"
SERVER_MAX_WAIT="${SERVER_MAX_WAIT:-1800}"
SERVER_CONNECT_INTERVAL="${SERVER_CONNECT_INTERVAL:-2}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-}"
ROBOTWIN_PATH="${ROBOTWIN_PATH:-}"
# RoboTwin eval_policy.py defaults ROBOTWIN_DISABLE_EVAL_VIDEO=1 (no MP4).
# This script wires eval_result -> OUTPUT_ROOT; enable videos unless caller opts out.
export ROBOTWIN_DISABLE_EVAL_VIDEO="${ROBOTWIN_DISABLE_EVAL_VIDEO:-0}"

ALL_TASKS=(
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

if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint is not visible in the client container: ${CKPT}" >&2
    echo "[ERROR] The client still needs the checkpoint path for StarVLA/RoboTwin runtime configuration." >&2
    exit 1
fi

if [[ ! "${NUM_CLIENTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_CLIENTS must be a positive integer: ${NUM_CLIENTS}" >&2
    exit 1
fi

if [[ -z "${ROBOTWIN_PYTHON}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        ROBOTWIN_PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        ROBOTWIN_PYTHON="$(command -v python)"
    else
        echo "[ERROR] Cannot find python3/python. Set ROBOTWIN_PYTHON." >&2
        exit 1
    fi
fi

if [[ -z "${ROBOTWIN_PATH}" ]]; then
    for candidate in \
        /workspace/RoboTwin \
        /opt/RoboTwin \
        /root/RoboTwin \
        /RoboTwin \
        /running_package/RoboTwin \
        /mnt/data/gaoning/code_repos/RoboTwin
    do
        if [[ -f "${candidate}/script/eval_policy.py" ]]; then
            ROBOTWIN_PATH="${candidate}"
            break
        fi
    done
fi

if [[ -z "${ROBOTWIN_PATH}" || ! -f "${ROBOTWIN_PATH}/script/eval_policy.py" ]]; then
    echo "[ERROR] Cannot locate RoboTwin. Set ROBOTWIN_PATH to the repo inside the image." >&2
    exit 1
fi

if ! grep -q "policy_ckpt_path" "${ROBOTWIN_PATH}/script/eval_policy.py"; then
    echo "[ERROR] ${ROBOTWIN_PATH}/script/eval_policy.py lacks StarVLA's policy_ckpt_path patch." >&2
    echo "[ERROR] Bake the patch documented in examples/Robotwin/README.md into the RoboTwin image." >&2
    exit 1
fi

detect_gpu_ids() {
    local visible="${CUDA_VISIBLE_DEVICES:-}"
    local -a ids=()
    local count

    if [[ -n "${visible}" ]]; then
        IFS=',' read -ra ids <<< "${visible}"
        printf '%s\n' "${ids[@]}"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        count="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
        for ((i = 0; i < count; ++i)); do
            printf '%s\n' "${i}"
        done
        return
    fi

    printf '0\n'
}

mapfile -t GPU_IDS < <(detect_gpu_ids)
if (( ${#GPU_IDS[@]} < NUM_CLIENTS )); then
    echo "[ERROR] Requested ${NUM_CLIENTS} clients, but only ${#GPU_IDS[@]} GPU IDs are visible: ${GPU_IDS[*]}" >&2
    exit 1
fi
GPU_IDS=("${GPU_IDS[@]:0:NUM_CLIENTS}")

declare -a SELECTED_TASKS=()
if [[ "${TASKS}" == "all" ]]; then
    SELECTED_TASKS=("${ALL_TASKS[@]}")
elif [[ -f "${TASKS}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line%%#*}"
        line="$(xargs <<<"${line}")"
        [[ -n "${line}" ]] && SELECTED_TASKS+=("${line}")
    done < "${TASKS}"
else
    TASKS="${TASKS//,/ }"
    read -r -a SELECTED_TASKS <<< "${TASKS}"
fi

if (( ${#SELECTED_TASKS[@]} == 0 )); then
    echo "[ERROR] No tasks selected." >&2
    exit 1
fi

read -r -a MODE_LIST <<< "${MODES}"
for mode in "${MODE_LIST[@]}"; do
    if [[ "${mode}" != "demo_clean" && "${mode}" != "demo_randomized" ]]; then
        echo "[ERROR] Unsupported mode: ${mode}" >&2
        exit 1
    fi
done

if [[ "${CKPT}" == *"/checkpoints/"* ]]; then
    MODEL_ROOT="$(dirname "$(dirname "${CKPT}")")"
else
    MODEL_ROOT="$(dirname "${CKPT}")"
fi
CKPT_NAME="$(basename "${CKPT}")"
CKPT_STEM="${CKPT_NAME%.*}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MODEL_ROOT}/robotwin_eval_results/${RUN_NAME}_${CKPT_STEM}_${TIMESTAMP}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
NATIVE_RESULT_ROOT="${OUTPUT_ROOT}/robotwin_native_eval_result"
MANIFEST="${OUTPUT_ROOT}/manifest.tsv"
mkdir -p "${LOG_DIR}" "${NATIVE_RESULT_ROOT}"

# RoboTwin hard-codes outputs under <ROBOTWIN_PATH>/eval_result. Redirect that
# directory to shared storage so videos and native result files survive the job.
if [[ -L "${ROBOTWIN_PATH}/eval_result" ]]; then
    rm -f "${ROBOTWIN_PATH}/eval_result"
elif [[ -d "${ROBOTWIN_PATH}/eval_result" ]]; then
    if find "${ROBOTWIN_PATH}/eval_result" -mindepth 1 -print -quit | grep -q .; then
        backup="${ROBOTWIN_PATH}/eval_result.pre_starvla_$(date +%s)"
        mv "${ROBOTWIN_PATH}/eval_result" "${backup}"
        echo "[WARN] Existing non-empty eval_result moved to ${backup}"
    else
        rmdir "${ROBOTWIN_PATH}/eval_result"
    fi
elif [[ -e "${ROBOTWIN_PATH}/eval_result" ]]; then
    echo "[ERROR] ${ROBOTWIN_PATH}/eval_result exists and is not a directory/symlink." >&2
    exit 1
fi
ln -s "${NATIVE_RESULT_ROOT}" "${ROBOTWIN_PATH}/eval_result"

check_all_servers_once() {
    "${ROBOTWIN_PYTHON}" - "${HOST}" "${BASE_PORT}" "${NUM_CLIENTS}" <<'PY'
import socket
import sys

host = sys.argv[1]
base = int(sys.argv[2])
count = int(sys.argv[3])
failed = []
for port in range(base, base + count):
    try:
        with socket.create_connection((host, port), timeout=2.0):
            pass
    except OSError:
        failed.append(port)
if failed:
    print(",".join(map(str, failed)))
    raise SystemExit(1)
PY
}

echo "========== StarVLA RoboTwin 8-GPU Remote Evaluation =========="
echo "[INFO] repo:          ${REPO_ROOT}"
echo "[INFO] RoboTwin:      ${ROBOTWIN_PATH}"
echo "[INFO] checkpoint:    ${CKPT}"
echo "[INFO] server:        ${HOST}:${BASE_PORT}-$((BASE_PORT + NUM_CLIENTS - 1))"
echo "[INFO] client GPUs:   ${GPU_IDS[*]}"
echo "[INFO] modes:         ${MODE_LIST[*]}"
echo "[INFO] tasks:         ${#SELECTED_TASKS[@]}"
echo "[INFO] output:        ${OUTPUT_ROOT}"
echo "[INFO] videos:        $([[ "${ROBOTWIN_DISABLE_EVAL_VIDEO}" == "0" ]] && echo enabled || echo disabled)"
echo "[INFO] python:        ${ROBOTWIN_PYTHON}"

deadline=$((SECONDS + SERVER_MAX_WAIT))
while ! missing_ports="$(check_all_servers_once 2>&1)"; do
    if (( SECONDS >= deadline )); then
        echo "[ERROR] Timed out waiting for remote policy servers." >&2
        echo "[ERROR] Last connection error / missing ports: ${missing_ports}" >&2
        exit 1
    fi
    echo "[INFO] Waiting for remote servers; unavailable: ${missing_ports}"
    sleep "${SERVER_CONNECT_INTERVAL}"
done
echo "[READY] All remote policy servers are reachable."

declare -a JOB_MODES=()
declare -a JOB_TASKS=()
for mode in "${MODE_LIST[@]}"; do
    for task in "${SELECTED_TASKS[@]}"; do
        JOB_MODES+=("${mode}")
        JOB_TASKS+=("${task}")
    done
done
TOTAL_JOBS="${#JOB_TASKS[@]}"

printf 'job_id\tmode\ttask\tslot\tgpu\tport\tlog\n' > "${MANIFEST}"

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_JOB_IDS=()
declare -a ACTIVE_LOGS=()
declare -a FAILED_JOBS=()

kill_tree() {
    local pid="$1"
    local sig="${2:-TERM}"
    local child
    while read -r child; do
        [[ -n "${child}" ]] && kill_tree "${child}" "${sig}"
    done < <(ps -o pid= --ppid "${pid}" 2>/dev/null || true)
    kill -"${sig}" "${pid}" 2>/dev/null || true
}

cleanup() {
    trap - INT TERM EXIT
    echo "[INFO] Cleaning up RoboTwin client processes..."
    local pid
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill_tree "${pid}" TERM
    done
    sleep 2
    for pid in "${ACTIVE_PIDS[@]:-}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill_tree "${pid}" KILL
        fi
    done
}
trap cleanup INT TERM EXIT

launch_job() {
    local slot="$1"
    local job_id="$2"
    local mode="${JOB_MODES[$job_id]}"
    local task="${JOB_TASKS[$job_id]}"
    local gpu="${GPU_IDS[$slot]}"
    local port=$((BASE_PORT + slot))
    local log="${LOG_DIR}/job$(printf '%03d' "${job_id}")_${mode}_${task}_slot${slot}_gpu${gpu}_port${port}.log"

    echo "[LAUNCH] job=${job_id}/${TOTAL_JOBS} mode=${mode} task=${task} slot=${slot} gpu=${gpu} server=${HOST}:${port}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${job_id}" "${mode}" "${task}" "${slot}" "${gpu}" "${port}" "${log}" >> "${MANIFEST}"

    (
        export ROBOTWIN_PATH
        export ROBOTWIN_PYTHON
        export ROBOTWIN_POLICY_NAME="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
        export PYTHONUNBUFFERED=1

        bash "${SCRIPT_DIR}/eval.sh" \
            "${task}" \
            "${mode}" \
            "${RUN_NAME}" \
            "${SEED}" \
            "${gpu}" \
            "${CKPT}" \
            "${port}" \
            "${HOST}" \
            > >(
                tee "${log}" \
                | grep --line-buffered -E "Success rate|Traceback|ERROR|Error|Exception" \
                | sed -u "s/^/[slot${slot} ${mode} ${task}] /" \
                || true
            ) 2>&1
    ) &

    ACTIVE_PIDS[$slot]="$!"
    ACTIVE_JOB_IDS[$slot]="${job_id}"
    ACTIVE_LOGS[$slot]="${log}"
}

next_job=0
completed=0

while (( completed < TOTAL_JOBS )); do
    for ((slot = 0; slot < NUM_CLIENTS; ++slot)); do
        pid="${ACTIVE_PIDS[$slot]:-}"

        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            job_id="${ACTIVE_JOB_IDS[$slot]}"
            mode="${JOB_MODES[$job_id]}"
            task="${JOB_TASKS[$job_id]}"
            log="${ACTIVE_LOGS[$slot]}"

            if wait "${pid}"; then
                echo "[DONE] job=${job_id} mode=${mode} task=${task} slot=${slot}"
            else
                status=$?
                FAILED_JOBS+=("${job_id}:${mode}:${task}:status=${status}")
                echo "[FAILED] job=${job_id} mode=${mode} task=${task} slot=${slot} status=${status}" >&2
                echo "[FAILED] log=${log}" >&2
            fi

            ACTIVE_PIDS[$slot]=""
            ACTIVE_JOB_IDS[$slot]=""
            ACTIVE_LOGS[$slot]=""
            completed=$((completed + 1))
            echo "[PROGRESS] ${completed}/${TOTAL_JOBS} task-mode jobs finished."
        fi

        if [[ -z "${ACTIVE_PIDS[$slot]:-}" ]] && (( next_job < TOTAL_JOBS )); then
            launch_job "${slot}" "${next_job}"
            next_job=$((next_job + 1))
        fi
    done

    if (( completed < TOTAL_JOBS )); then
        sleep 3
    fi
done

trap - INT TERM EXIT

"${ROBOTWIN_PYTHON}" "${SCRIPT_DIR}/aggregate_robotwin_8clients.py" \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_ROOT}"

if (( ${#FAILED_JOBS[@]} > 0 )); then
    printf '%s\n' "${FAILED_JOBS[@]}" > "${OUTPUT_ROOT}/failed_jobs.txt"
    echo "[ERROR] ${#FAILED_JOBS[@]} task-mode jobs failed. See ${OUTPUT_ROOT}/failed_jobs.txt" >&2
    exit 1
fi

echo "[SUCCESS] RoboTwin evaluation finished."
echo "[SUCCESS] Summary: ${OUTPUT_ROOT}/summary.txt"
echo "[SUCCESS] CSV:     ${OUTPUT_ROOT}/results.csv"
echo "[SUCCESS] Native:  ${NATIVE_RESULT_ROOT}"
