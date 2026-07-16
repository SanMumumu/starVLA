#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CHECKPOINT_PATH="${ROBODOJO_CKPT:-${CKPT:-}}"
BASE_PORT="${BASE_PORT:-7777}"
NUM_SERVERS="${NUM_SERVERS:-8}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python3}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
READY_CHECK_INTERVAL="${READY_CHECK_INTERVAL:-3}"
ADVERTISE_HOST="${ADVERTISE_HOST:-}"

[[ -n "${CHECKPOINT_PATH}" ]] || {
  echo "[RoboDojo][server][ERROR] set ROBODOJO_CKPT=/absolute/path/to/checkpoint.pt" >&2
  exit 2
}
[[ -f "${CHECKPOINT_PATH}" ]] || {
  echo "[RoboDojo][server][ERROR] checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
}
[[ "${NUM_SERVERS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[RoboDojo][server][ERROR] NUM_SERVERS must be a positive integer" >&2
  exit 2
}
[[ "${BASE_PORT}" =~ ^[0-9]+$ ]] || {
  echo "[RoboDojo][server][ERROR] BASE_PORT must be an integer" >&2
  exit 2
}
(( BASE_PORT >= 1 && BASE_PORT + NUM_SERVERS - 1 <= 65535 )) || {
  echo "[RoboDojo][server][ERROR] invalid port range" >&2
  exit 2
}
command -v "${STARVLA_PYTHON}" >/dev/null 2>&1 || [[ -x "${STARVLA_PYTHON}" ]] || {
  echo "[RoboDojo][server][ERROR] Python is not executable: ${STARVLA_PYTHON}" >&2
  exit 1
}

export STARVLA_ROOT
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export NO_ALBUMENTATIONS_UPDATE=1

"${STARVLA_PYTHON}" "${SCRIPT_DIR}/verify_robodojo_checkpoint_contract.py" \
  --checkpoint "${CHECKPOINT_PATH}"

GPU_COUNT="$("${STARVLA_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
[[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] || {
  echo "[RoboDojo][server][ERROR] failed to determine CUDA device count" >&2
  exit 1
}
(( GPU_COUNT >= NUM_SERVERS )) || {
  echo "[RoboDojo][server][ERROR] requested ${NUM_SERVERS} servers, but Python sees ${GPU_COUNT} GPUs" >&2
  exit 1
}

declare -a GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  for ((slot = 0; slot < NUM_SERVERS; ++slot)); do
    GPU_IDS+=("${slot}")
  done
fi
(( ${#GPU_IDS[@]} >= NUM_SERVERS )) || {
  echo "[RoboDojo][server][ERROR] CUDA_VISIBLE_DEVICES exposes only ${#GPU_IDS[@]} IDs" >&2
  exit 1
}
GPU_IDS=("${GPU_IDS[@]:0:NUM_SERVERS}")

if [[ -z "${ADVERTISE_HOST}" ]]; then
  ADVERTISE_HOST="$(hostname -I 2>/dev/null | awk '{for (i=1; i<=NF; ++i) if ($i !~ /^127\./) {print $i; exit}}')"
fi
[[ -n "${ADVERTISE_HOST}" ]] || {
  echo "[RoboDojo][server][ERROR] cannot determine the server IP; set ADVERTISE_HOST" >&2
  exit 1
}

if [[ "${CHECKPOINT_PATH}" == *"/checkpoints/"* ]]; then
  MODEL_ROOT="$(dirname "$(dirname "${CHECKPOINT_PATH}")")"
else
  MODEL_ROOT="$(dirname "${CHECKPOINT_PATH}")"
fi
MODEL_ROOT="$(cd "${MODEL_ROOT}" && pwd -P)"
CKPT_STEM="$(basename "${CHECKPOINT_PATH}")"
CKPT_STEM="${CKPT_STEM%.*}"
LAST_PORT=$((BASE_PORT + NUM_SERVERS - 1))
LOG_ROOT="${ROBODOJO_SERVER_LOG_ROOT:-${MODEL_ROOT}/robodojo_server_logs/${CKPT_STEM}_ports_${BASE_PORT}_${LAST_PORT}}"
mkdir -p "${LOG_ROOT}"

declare -a PIDS=()

kill_tree() {
  local pid="$1" signal="${2:-TERM}" child
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${child}" "${signal}"
  done < <(ps -o pid= --ppid "${pid}" 2>/dev/null || true)
  kill -"${signal}" "${pid}" 2>/dev/null || true
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill_tree "${pid}" TERM
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill_tree "${pid}" KILL
    fi
  done
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
  done
  exit "${rc}"
}
trap cleanup EXIT INT TERM

websocket_is_ready() {
  local host="$1" port="$2"
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
    NO_PROXY="${host},127.0.0.1,localhost" \
    no_proxy="${host},127.0.0.1,localhost" \
    "${STARVLA_PYTHON}" - "${host}" "${port}" <<'PY'
import asyncio
import sys

import websockets


async def main():
    uri = f"ws://{sys.argv[1]}:{int(sys.argv[2])}"
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


raise SystemExit(asyncio.run(main()))
PY
}

show_log_tail() {
  local slot="$1"
  local gpu="${GPU_IDS[$slot]}"
  local port=$((BASE_PORT + slot))
  local log_file="${LOG_ROOT}/server_${slot}_gpu${gpu}_port${port}.log"
  echo "[RoboDojo][server][ERROR] tail of ${log_file}:" >&2
  tail -n 80 "${log_file}" >&2 2>/dev/null || true
}

echo "========== RoboDojo StarVLA policy servers =========="
echo "[RoboDojo][server] checkpoint=${CHECKPOINT_PATH}"
echo "[RoboDojo][server] GPUs=${GPU_IDS[*]}"
echo "[RoboDojo][server] ports=${BASE_PORT}-${LAST_PORT}"
echo "[RoboDojo][server] logs=${LOG_ROOT}"
echo "[RoboDojo][server] advertised-host=${ADVERTISE_HOST}"

for ((slot = 0; slot < NUM_SERVERS; ++slot)); do
  gpu="${GPU_IDS[$slot]}"
  port=$((BASE_PORT + slot))
  log_file="${LOG_ROOT}/server_${slot}_gpu${gpu}_port${port}.log"
  echo "[RoboDojo][server] launch slot=${slot} gpu=${gpu} port=${port}"
  (
    unset DEBUG
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${STARVLA_PYTHON}" -u -m deployment.model_server.server_policy \
      --ckpt_path "${CHECKPOINT_PATH}" \
      --port "${port}" \
      --idle_timeout -1 \
      --use_bf16
  ) >"${log_file}" 2>&1 &
  PIDS+=("$!")
done

deadline=$((SECONDS + SERVER_READY_TIMEOUT))
while :; do
  ready=0
  for ((slot = 0; slot < NUM_SERVERS; ++slot)); do
    port=$((BASE_PORT + slot))
    if websocket_is_ready 127.0.0.1 "${port}"; then
      ready=$((ready + 1))
    fi
  done
  (( ready < NUM_SERVERS )) || break

  for slot in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$slot]}" 2>/dev/null; then
      status=0
      wait "${PIDS[$slot]}" || status=$?
      echo "[RoboDojo][server][ERROR] slot ${slot} exited during startup (rc=${status})" >&2
      show_log_tail "${slot}"
      exit 1
    fi
  done
  (( SECONDS < deadline )) || {
    echo "[RoboDojo][server][ERROR] only ${ready}/${NUM_SERVERS} servers ready after ${SERVER_READY_TIMEOUT}s" >&2
    for ((slot = 0; slot < NUM_SERVERS; ++slot)); do show_log_tail "${slot}"; done
    exit 1
  }
  echo "[RoboDojo][server] waiting: ${ready}/${NUM_SERVERS} ready"
  sleep "${READY_CHECK_INTERVAL}"
done

echo "[RoboDojo][server][READY] all ${NUM_SERVERS} servers passed WebSocket handshake"
echo "[RoboDojo][server][READY] export HOST=${ADVERTISE_HOST}"
echo "[RoboDojo][server][READY] export BASE_PORT=${BASE_PORT}"
echo "[RoboDojo][server][READY] keep this terminal running"

while :; do
  for slot in "${!PIDS[@]}"; do
    if ! kill -0 "${PIDS[$slot]}" 2>/dev/null; then
      status=0
      wait "${PIDS[$slot]}" || status=$?
      echo "[RoboDojo][server][ERROR] slot ${slot} exited (rc=${status})" >&2
      show_log_tail "${slot}"
      exit 1
    fi
  done
  sleep 5
done
