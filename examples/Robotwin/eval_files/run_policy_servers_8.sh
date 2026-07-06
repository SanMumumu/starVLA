#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CKPT="${CKPT:?Please set CKPT to a StarVLA checkpoint}"
BASE_PORT="${BASE_PORT:-6698}"
NUM_SERVERS="${NUM_SERVERS:-8}"
STARVLA_PYTHON="${STARVLA_PYTHON:-}"
USE_BF16="${USE_BF16:-1}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
READY_CHECK_INTERVAL="${READY_CHECK_INTERVAL:-3}"
ADVERTISE_HOST="${ADVERTISE_HOST:-}"

if [[ ! -f "${CKPT}" ]]; then
    echo "[ERROR] Checkpoint does not exist: ${CKPT}" >&2
    exit 1
fi

if [[ ! "${NUM_SERVERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_SERVERS must be a positive integer: ${NUM_SERVERS}" >&2
    exit 1
fi

if [[ ! "${BASE_PORT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] BASE_PORT must be an integer: ${BASE_PORT}" >&2
    exit 1
fi

if (( BASE_PORT < 1 || BASE_PORT + NUM_SERVERS - 1 > 65535 )); then
    echo "[ERROR] Invalid port range: ${BASE_PORT}-$((BASE_PORT + NUM_SERVERS - 1))" >&2
    exit 1
fi

if [[ -z "${STARVLA_PYTHON}" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        STARVLA_PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        STARVLA_PYTHON="$(command -v python)"
    else
        echo "[ERROR] Cannot find python3/python. Set STARVLA_PYTHON." >&2
        exit 1
    fi
fi

if [[ ! -x "${STARVLA_PYTHON}" ]]; then
    echo "[ERROR] Python is not executable: ${STARVLA_PYTHON}" >&2
    exit 1
fi

# 使用 Python 看到的 CUDA 数量，避免 nvidia-smi 与 CUDA_VISIBLE_DEVICES 不一致。
GPU_COUNT="$(
    "${STARVLA_PYTHON}" - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"

if (( GPU_COUNT < NUM_SERVERS )); then
    echo "[ERROR] Requested ${NUM_SERVERS} servers, but Python sees only ${GPU_COUNT} GPUs." >&2
    exit 1
fi

# 当前 Job 对其他 Job 可见的地址。允许通过 ADVERTISE_HOST 手动覆盖。
if [[ -z "${ADVERTISE_HOST}" ]]; then
    ADVERTISE_HOST="$(
        hostname -I 2>/dev/null |
        awk '{for (i=1; i<=NF; i++) if ($i !~ /^127\./) {print $i; exit}}'
    )"
fi

if [[ -z "${ADVERTISE_HOST}" ]]; then
    ADVERTISE_HOST="$(
        "${STARVLA_PYTHON}" - <<'PY'
import socket

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("10.0.0.1", 1))
    print(sock.getsockname()[0])
    sock.close()
except Exception:
    print(socket.gethostbyname(socket.gethostname()))
PY
)"
fi

if [[ -z "${ADVERTISE_HOST}" ]]; then
    echo "[ERROR] Failed to determine server Job IP. Set ADVERTISE_HOST manually." >&2
    exit 1
fi

CKPT_NAME="$(basename "${CKPT}")"
CKPT_STEM="${CKPT_NAME%.*}"

if [[ "${CKPT}" == *"/checkpoints/"* ]]; then
    MODEL_ROOT="$(dirname "$(dirname "${CKPT}")")"
else
    MODEL_ROOT="$(dirname "${CKPT}")"
fi

LAST_PORT=$((BASE_PORT + NUM_SERVERS - 1))

LOG_ROOT="${LOG_ROOT:-${MODEL_ROOT}/robotwin_server_logs/${CKPT_STEM}_ports_${BASE_PORT}_${LAST_PORT}}"
mkdir -p "${LOG_ROOT}"

PIDS=()
PORTS=()

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
    local status=$?

    trap - INT TERM EXIT

    if (( ${#PIDS[@]} > 0 )); then
        echo "[INFO] Stopping RoboTwin policy servers..."

        local pid
        for pid in "${PIDS[@]}"; do
            [[ -n "${pid}" ]] && kill_tree "${pid}" TERM
        done

        sleep 2

        for pid in "${PIDS[@]}"; do
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                kill_tree "${pid}" KILL
            fi
        done

        for pid in "${PIDS[@]}"; do
            [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
        done
    fi

    exit "${status}"
}

trap cleanup INT TERM EXIT

# 真正执行 WebSocket 握手，不能再用裸 TCP socket 探测。
websocket_is_ready() {
    local host="$1"
    local port="$2"

    env \
        -u http_proxy \
        -u https_proxy \
        -u HTTP_PROXY \
        -u HTTPS_PROXY \
        -u all_proxy \
        -u ALL_PROXY \
        NO_PROXY="${host},127.0.0.1,localhost" \
        no_proxy="${host},127.0.0.1,localhost" \
        "${STARVLA_PYTHON}" - "${host}" "${port}" <<'PY'
import asyncio
import sys

import websockets

host = sys.argv[1]
port = int(sys.argv[2])


async def check() -> int:
    uri = f"ws://{host}:{port}"

    try:
        async with websockets.connect(
            uri,
            open_timeout=2,
            close_timeout=1,
            ping_interval=None,
            max_size=None,
        ):
            return 0
    except Exception:
        return 1


raise SystemExit(asyncio.run(check()))
PY
}

show_failed_server_log() {
    local idx="$1"
    local gpu_id="${idx}"
    local port=$((BASE_PORT + idx))
    local log_file="${LOG_ROOT}/server_${idx}_gpu${gpu_id}_port${port}.log"

    echo "[ERROR] Last 80 lines of ${log_file}:" >&2
    tail -n 80 "${log_file}" >&2 2>/dev/null || true
}

echo "========== StarVLA RoboTwin ${NUM_SERVERS}-GPU Policy Servers =========="
echo "[INFO] repo:           ${REPO_ROOT}"
echo "[INFO] checkpoint:     ${CKPT}"
echo "[INFO] servers:        ${NUM_SERVERS}"
echo "[INFO] ports:          ${BASE_PORT}-${LAST_PORT}"
echo "[INFO] advertise host: ${ADVERTISE_HOST}"
echo "[INFO] logs:           ${LOG_ROOT}"
echo "[INFO] python:         ${STARVLA_PYTHON}"
echo "[INFO] visible GPUs:   ${GPU_COUNT}"

for ((slot = 0; slot < NUM_SERVERS; ++slot)); do
    gpu_id="${slot}"
    port=$((BASE_PORT + slot))
    log_file="${LOG_ROOT}/server_${slot}_gpu${gpu_id}_port${port}.log"

    PORTS+=("${port}")

    echo "[INFO] Launch server ${slot}: gpu=${gpu_id}, port=${port}"

    (
        export STARVLA_PYTHON
        export ROBOTWIN_USE_BF16="${USE_BF16}"
        export NO_ALBUMENTATIONS_UPDATE=1

        exec bash "${SCRIPT_DIR}/run_policy_server.sh" \
            "${CKPT}" \
            "${gpu_id}" \
            "${port}"
    ) >"${log_file}" 2>&1 &

    PIDS+=("$!")
done

deadline=$((SECONDS + SERVER_READY_TIMEOUT))

while true; do
    ready=0

    for port in "${PORTS[@]}"; do
        if websocket_is_ready 127.0.0.1 "${port}"; then
            ready=$((ready + 1))
        fi
    done

    if (( ready == NUM_SERVERS )); then
        break
    fi

    for idx in "${!PIDS[@]}"; do
        pid="${PIDS[$idx]}"

        if ! kill -0 "${pid}" 2>/dev/null; then
            set +e
            wait "${pid}"
            status=$?
            set -e

            echo "[ERROR] Server ${idx} exited before becoming ready; status=${status}." >&2
            show_failed_server_log "${idx}"
            exit 1
        fi
    done

    if (( SECONDS >= deadline )); then
        echo "[ERROR] Only ${ready}/${NUM_SERVERS} servers became ready within ${SERVER_READY_TIMEOUT}s." >&2

        for idx in "${!PIDS[@]}"; do
            show_failed_server_log "${idx}"
        done

        exit 1
    fi

    echo "[INFO] Waiting for servers: ${ready}/${NUM_SERVERS} ready..."
    sleep "${READY_CHECK_INTERVAL}"
done

echo
echo "[READY] All ${NUM_SERVERS} policy servers completed WebSocket handshakes."
echo "[READY] HOST=${ADVERTISE_HOST}"
echo "[READY] BASE_PORT=${BASE_PORT}"
echo "[READY] NUM_CLIENTS=${NUM_SERVERS}"
echo
echo "[READY] Client command:"
echo "HOST=${ADVERTISE_HOST} BASE_PORT=${BASE_PORT} NUM_CLIENTS=${NUM_SERVERS} bash <client-script>"

while true; do
    for idx in "${!PIDS[@]}"; do
        pid="${PIDS[$idx]}"

        if ! kill -0 "${pid}" 2>/dev/null; then
            set +e
            wait "${pid}"
            status=$?
            set -e

            echo "[ERROR] Server ${idx} exited with status ${status}." >&2
            show_failed_server_log "${idx}"

            if (( status == 0 )); then
                status=1
            fi

            exit "${status}"
        fi
    done

    sleep 5
done
