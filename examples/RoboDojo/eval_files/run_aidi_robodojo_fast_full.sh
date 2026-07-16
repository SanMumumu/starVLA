#!/usr/bin/env bash
set -Eeuo pipefail

echo "[AIDI][RoboDojo-full] entry host=${HOSTNAME:-unknown}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboDojo}"
CHECKPOINT_PATH="${STARVLA_CKPT_PATH:?export STARVLA_CKPT_PATH=/absolute/path/to/checkpoint.pt}"
CKPT_NAME="${ROBODOJO_CKPT_NAME:-robodojo_baseline_steps80000_h16_state_zscore}"
HOST="${STARVLA_SERVER_HOST:-${HOST:-}}"
BASE_PORT="${BASE_PORT:-7777}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
LOCAL_BASE_PORT="${ROBODOJO_LOCAL_BASE_PORT:-17777}"
SEED="${ROBODOJO_SEED:-0}"
SERVER_MAX_WAIT="${SERVER_MAX_WAIT:-1200}"
PROGRESS_INTERVAL="${ROBODOJO_PROGRESS_INTERVAL:-60}"
ROBODOJO_PYTHON="${ROBODOJO_PYTHON:-/opt/robodojo-env/bin/python}"

[[ -n "${HOST}" ]] || {
  echo "[RoboDojo-full][ERROR] set HOST to the H20 server job IP" >&2
  exit 2
}
[[ -f "${CHECKPOINT_PATH}" ]] || {
  echo "[RoboDojo-full][ERROR] checkpoint does not exist: ${CHECKPOINT_PATH}" >&2
  exit 1
}
command -v "${ROBODOJO_PYTHON}" >/dev/null 2>&1 || [[ -x "${ROBODOJO_PYTHON}" ]] || {
  echo "[RoboDojo-full][ERROR] RoboDojo Python is not executable: ${ROBODOJO_PYTHON}" >&2
  exit 1
}
[[ "${NUM_CLIENTS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[RoboDojo-full][ERROR] NUM_CLIENTS must be a positive integer" >&2
  exit 2
}
[[ "${BASE_PORT}" =~ ^[0-9]+$ ]] || {
  echo "[RoboDojo-full][ERROR] BASE_PORT must be an integer" >&2
  exit 2
}
[[ "${LOCAL_BASE_PORT}" =~ ^[0-9]+$ ]] || {
  echo "[RoboDojo-full][ERROR] ROBODOJO_LOCAL_BASE_PORT must be an integer" >&2
  exit 2
}
[[ "${PROGRESS_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
  echo "[RoboDojo-full][ERROR] ROBODOJO_PROGRESS_INTERVAL must be a positive integer" >&2
  exit 2
}
(( BASE_PORT >= 1 && BASE_PORT + NUM_CLIENTS - 1 <= 65535 )) || {
  echo "[RoboDojo-full][ERROR] invalid server port range" >&2
  exit 2
}
(( LOCAL_BASE_PORT >= 1 && LOCAL_BASE_PORT + NUM_CLIENTS - 1 <= 65535 )) || {
  echo "[RoboDojo-full][ERROR] invalid local XPolicy bridge port range" >&2
  exit 2
}

for required in \
  "${SCRIPT_DIR}/eval_robodojo.sh" \
  "${SCRIPT_DIR}/run_aidi_robodojo_fast.sh" \
  "${SCRIPT_DIR}/summarize_robodojo_table1.py" \
  "${ROBODOJO_ROOT}/task/RoboDojo/config/_task.yml"; do
  [[ -f "${required}" ]] || {
    echo "[RoboDojo-full][ERROR] missing ${required}" >&2
    exit 1
  }
done

export STARVLA_ROOT ROBODOJO_ROOT ROBODOJO_PYTHON
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export NO_ALBUMENTATIONS_UPDATE=1
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"

# Build the exact paper protocol from the same taxonomy used by the Table-1
# summarizer, and verify that the mounted official RoboDojo checkout declares
# 25 trials for Generalization halves and 50 for all other tasks.
mapfile -t JOB_SPECS < <(
  "${ROBODOJO_PYTHON}" - "${SCRIPT_DIR}" "${ROBODOJO_ROOT}" <<'PY'
from pathlib import Path
import sys

import yaml

script_dir = Path(sys.argv[1])
robodojo_root = Path(sys.argv[2])
sys.path.insert(0, str(script_dir))
from summarize_robodojo_table1 import DIMENSIONS

task_index = yaml.safe_load(
    (robodojo_root / "task/RoboDojo/config/_task.yml").read_text(encoding="utf-8")
)
common = task_index.get("common", {})
overrides = task_index.get("tasks", {})

specs = []
for dimension, tasks in DIMENSIONS.items():
    for task in tasks:
        variants = ((task, 25), (f"{task}_random", 25)) if dimension == "Generalization" else ((task, 50),)
        for variant, expected in variants:
            config = robodojo_root / "task/RoboDojo/config" / f"{variant}.yml"
            implementation = robodojo_root / "task/RoboDojo/tasks" / f"{variant}.py"
            if not config.is_file():
                raise SystemExit(f"missing official task config: {config}")
            if not implementation.is_file():
                raise SystemExit(f"missing official task implementation: {implementation}")
            native = overrides.get(variant, {}).get("eval_nums", common.get("eval_nums", 50))
            if int(native) != expected:
                raise SystemExit(
                    f"official trial contract mismatch for {variant}: native={native}, expected={expected}"
                )
            specs.append((dimension, variant, expected))

if len(specs) != 54 or sum(item[2] for item in specs) != 2100:
    raise SystemExit(f"invalid full benchmark matrix: jobs={len(specs)}, episodes={sum(x[2] for x in specs)}")
for dimension, task, episodes in specs:
    print(f"{dimension}\t{task}\t{episodes}")
PY
)

(( ${#JOB_SPECS[@]} == 54 )) || {
  echo "[RoboDojo-full][ERROR] expected 54 rollout jobs, got ${#JOB_SPECS[@]}" >&2
  exit 1
}

declare -a GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  for ((slot = 0; slot < NUM_CLIENTS; ++slot)); do
    GPU_IDS+=("${slot}")
  done
fi
(( ${#GPU_IDS[@]} >= NUM_CLIENTS )) || {
  echo "[RoboDojo-full][ERROR] only ${#GPU_IDS[@]} client GPU IDs are visible for ${NUM_CLIENTS} clients" >&2
  exit 1
}
GPU_IDS=("${GPU_IDS[@]:0:NUM_CLIENTS}")

if [[ "${CHECKPOINT_PATH}" == *"/checkpoints/"* ]]; then
  MODEL_ROOT="$(dirname "$(dirname "${CHECKPOINT_PATH}")")"
else
  MODEL_ROOT="$(dirname "${CHECKPOINT_PATH}")"
fi
MODEL_ROOT="$(cd "${MODEL_ROOT}" && pwd -P)"
CKPT_STEM="$(basename "${CHECKPOINT_PATH}")"
CKPT_STEM="${CKPT_STEM%.*}"
FULL_RUN_ID="${ROBODOJO_FULL_RUN_ID:-$(date +%Y%m%d_%H%M%S)-$$}"
FULL_RUN_NAME="${ROBODOJO_FULL_RUN_NAME:-full_official_fast}"
OUTPUT_ROOT="${ROBODOJO_FULL_OUTPUT_ROOT:-${MODEL_ROOT}/robodojo_eval_results/${FULL_RUN_NAME}_${CKPT_STEM}_${FULL_RUN_ID}}"
case "${OUTPUT_ROOT}" in
  "${MODEL_ROOT}"/*) ;;
  *)
    echo "[RoboDojo-full][ERROR] output must stay under checkpoint experiment directory ${MODEL_ROOT}" >&2
    exit 2
    ;;
esac
mkdir -p "${OUTPUT_ROOT}/jobs"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd -P)"
MANIFEST="${OUTPUT_ROOT}/manifest.tsv"
STATUS_FILE="${OUTPUT_ROOT}/status.tsv"
printf 'job_id\tdimension\ttask\texpected_episodes\tslot\tgpu\tserver\tlocal_bridge\toutput\tclient_log\n' > "${MANIFEST}"
printf 'job_id\tdimension\ttask\tstatus\texit_code\tepisodes\texpected\tresult\n' > "${STATUS_FILE}"

check_servers() {
  env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
    NO_PROXY="${HOST},127.0.0.1,localhost" \
    no_proxy="${HOST},127.0.0.1,localhost" \
    "${ROBODOJO_PYTHON}" - "${HOST}" "${BASE_PORT}" "${NUM_CLIENTS}" <<'PY'
import sys

from websockets.sync.client import connect

host, base, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
missing = []
for port in range(base, base + count):
    try:
        with connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            open_timeout=3,
            close_timeout=1,
        ) as websocket:
            websocket.recv(timeout=3)
    except Exception:
        missing.append(str(port))
if missing:
    print(",".join(missing))
    raise SystemExit(1)
PY
}

check_local_bridge_ports() {
  "${ROBODOJO_PYTHON}" - "${LOCAL_BASE_PORT}" "${NUM_CLIENTS}" <<'PY'
import socket
import sys

base, count = int(sys.argv[1]), int(sys.argv[2])
busy = []
for port in range(base, base + count):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        busy.append(str(port))
    finally:
        sock.close()
if busy:
    print(",".join(busy))
    raise SystemExit(1)
PY
}

echo "========== RoboDojo official full fast rollout =========="
echo "[RoboDojo-full] protocol=42 tasks / 54 rollout jobs / 2100 episodes"
echo "[RoboDojo-full] Generalization=12 x (25 standard + 25 random)"
echo "[RoboDojo-full] Precision+Long-Horizon+Memory+Open=30 x 50"
echo "[RoboDojo-full] clients=${NUM_CLIENTS} GPUs=${GPU_IDS[*]}"
echo "[RoboDojo-full] servers=${HOST}:${BASE_PORT}-$((BASE_PORT + NUM_CLIENTS - 1))"
echo "[RoboDojo-full] local-bridges=127.0.0.1:${LOCAL_BASE_PORT}-$((LOCAL_BASE_PORT + NUM_CLIENTS - 1))"
echo "[RoboDojo-full] videos=disabled"
echo "[RoboDojo-full] output=${OUTPUT_ROOT}"
echo "[RoboDojo-full] live progress stays here; each complete client log is under jobs/*/logs/client.log"

deadline=$((SECONDS + SERVER_MAX_WAIT))
while ! missing="$(check_servers 2>&1)"; do
  (( SECONDS < deadline )) || {
    echo "[RoboDojo-full][ERROR] policy server ports unavailable: ${missing}" >&2
    exit 1
  }
  echo "[RoboDojo-full] waiting for policy server ports: ${missing}"
  sleep 3
done
if ! busy="$(check_local_bridge_ports 2>&1)"; then
  echo "[RoboDojo-full][ERROR] local XPolicy bridge ports already in use: ${busy}" >&2
  echo "[RoboDojo-full][ERROR] stop the old eval, or set ROBODOJO_LOCAL_BASE_PORT to another free range" >&2
  exit 1
fi

declare -a JOB_DIMENSIONS=() JOB_TASKS=() JOB_EPISODES=()
for spec in "${JOB_SPECS[@]}"; do
  IFS=$'\t' read -r dimension task episodes <<< "${spec}"
  JOB_DIMENSIONS+=("${dimension}")
  JOB_TASKS+=("${task}")
  JOB_EPISODES+=("${episodes}")
done
TOTAL_JOBS=${#JOB_TASKS[@]}

declare -a ACTIVE_PIDS=() ACTIVE_JOB_IDS=() ACTIVE_OUTPUTS=()
declare -a FAILED_JOBS=()

kill_tree() {
  local pid="$1" signal="${2:-TERM}" child
  while read -r child; do
    [[ -n "${child}" ]] && kill_tree "${child}" "${signal}"
  done < <(ps -o pid= --ppid "${pid}" 2>/dev/null || true)
  kill -"${signal}" "${pid}" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill_tree "${pid}" TERM
  done
  sleep 1
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill_tree "${pid}" KILL
    fi
  done
}
trap cleanup INT TERM EXIT

launch_job() {
  local slot="$1" job_id="$2"
  local dimension="${JOB_DIMENSIONS[$job_id]}"
  local task="${JOB_TASKS[$job_id]}"
  local expected="${JOB_EPISODES[$job_id]}"
  local gpu="${GPU_IDS[$slot]}"
  local port=$((BASE_PORT + slot))
  local bridge_port=$((LOCAL_BASE_PORT + slot))
  local display_id=$((job_id + 1))
  local stem="job$(printf '%03d' "${display_id}")_${task}"
  local job_output="${OUTPUT_ROOT}/jobs/${stem}"
  local client_log="${job_output}/logs/client.log"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s:%s\t127.0.0.1:%s\t%s\t%s\n' \
    "${display_id}" "${dimension}" "${task}" "${expected}" "${slot}" "${gpu}" \
    "${HOST}" "${port}" "${bridge_port}" "${job_output}" "${client_log}" >> "${MANIFEST}"
  echo "[LAUNCH] ${display_id}/${TOTAL_JOBS} ${dimension}/${task} (${expected} trials) -> slot=${slot} gpu=${gpu} server=${HOST}:${port}"

  (
    export STARVLA_SERVER_HOST="${HOST}"
    export STARVLA_SERVER_PORT="${port}"
    export STARVLA_CKPT_PATH="${CHECKPOINT_PATH}"
    export ROBODOJO_CKPT_NAME="${CKPT_NAME}"
    export ROBODOJO_TASK="${task}"
    export ROBODOJO_TRIALS=native
    export ROBODOJO_EVAL_MODE=fast
    export ROBODOJO_POLICY_GPU="${gpu}"
    export ROBODOJO_ENV_GPU="${gpu}"
    export ROBODOJO_XPOLICY_PORT="${bridge_port}"
    export ROBODOJO_SEED="${SEED}"
    export ROBODOJO_RUN_ID="${FULL_RUN_ID}_${stem}"
    export ROBODOJO_RUN_NAME="${FULL_RUN_NAME}_${task}"
    export ROBODOJO_OUTPUT_ROOT="${job_output}"
    export ROBODOJO_SKIP_TABLE1_SUMMARY=1
    exec bash "${SCRIPT_DIR}/run_aidi_robodojo_fast.sh"
  ) >/dev/null 2>&1 &

  ACTIVE_PIDS[$slot]=$!
  ACTIVE_JOB_IDS[$slot]="${job_id}"
  ACTIVE_OUTPUTS[$slot]="${job_output}"
}

result_info() {
  local job_output="$1" task="$2"
  "${ROBODOJO_PYTHON}" - "${job_output}" "${task}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "native/eval_result/RoboDojo" / sys.argv[2]
candidates = list(root.rglob("_result.json")) if root.is_dir() else []
if not candidates:
    print("0\t-")
    raise SystemExit(0)
path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
    details = payload.get("details", {})
    count = len(details) if isinstance(details, dict) else 0
except Exception:
    count = 0
print(f"{count}\t{path}")
PY
}

next_job=0
completed=0
last_progress=${SECONDS}
while (( completed < TOTAL_JOBS )); do
  for ((slot = 0; slot < NUM_CLIENTS; ++slot)); do
    pid="${ACTIVE_PIDS[$slot]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      job_id="${ACTIVE_JOB_IDS[$slot]}"
      dimension="${JOB_DIMENSIONS[$job_id]}"
      task="${JOB_TASKS[$job_id]}"
      expected="${JOB_EPISODES[$job_id]}"
      job_output="${ACTIVE_OUTPUTS[$slot]}"
      display_id=$((job_id + 1))

      status=0
      wait "${pid}" || status=$?
      IFS=$'\t' read -r actual result_path <<< "$(result_info "${job_output}" "${task}")"
      if (( status == 0 )) && [[ "${actual}" =~ ^[0-9]+$ ]] && (( actual >= expected )); then
        state=PASS
        echo "[DONE] ${display_id}/${TOTAL_JOBS} ${dimension}/${task} episodes=${actual}/${expected}"
      else
        state=FAIL
        FAILED_JOBS+=("${display_id}:${dimension}:${task}:rc=${status}:episodes=${actual}/${expected}")
        echo "[FAILED] ${display_id}/${TOTAL_JOBS} ${dimension}/${task} rc=${status} episodes=${actual}/${expected}; log=${job_output}/logs/client.log" >&2
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${display_id}" "${dimension}" "${task}" "${state}" "${status}" \
        "${actual}" "${expected}" "${result_path}" >> "${STATUS_FILE}"

      ACTIVE_PIDS[$slot]=""
      ACTIVE_JOB_IDS[$slot]=""
      ACTIVE_OUTPUTS[$slot]=""
      completed=$((completed + 1))
    fi

    if [[ -z "${ACTIVE_PIDS[$slot]:-}" ]] && (( next_job < TOTAL_JOBS )); then
      launch_job "${slot}" "${next_job}"
      next_job=$((next_job + 1))
    fi
  done
  if (( SECONDS - last_progress >= PROGRESS_INTERVAL )); then
    active=0
    for pid in "${ACTIVE_PIDS[@]:-}"; do
      [[ -n "${pid}" ]] && active=$((active + 1))
    done
    echo "[PROGRESS] completed=${completed}/${TOTAL_JOBS} active=${active} queued=$((TOTAL_JOBS - completed - active)); status=${STATUS_FILE}"
    last_progress=${SECONDS}
  fi
  (( completed >= TOTAL_JOBS )) || sleep 1
done

trap - INT TERM EXIT

"${ROBODOJO_PYTHON}" "${SCRIPT_DIR}/summarize_robodojo_table1.py" \
  --eval-root "${MODEL_ROOT}/robodojo_eval_results" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --ckpt-name "${CKPT_NAME}" \
  --seed "${SEED}"

TABLE_PREFIX="${MODEL_ROOT}/robodojo_eval_results/table1_${CKPT_STEM}_seed${SEED}"
pass_count=$((TOTAL_JOBS - ${#FAILED_JOBS[@]}))
{
  echo "RoboDojo official full fast rollout"
  echo "checkpoint=${CHECKPOINT_PATH}"
  echo "jobs_passed=${pass_count}/${TOTAL_JOBS}"
  echo "protocol_episodes=2100"
  echo "videos=disabled"
  echo "table_markdown=${TABLE_PREFIX}.md"
  echo "table_csv=${TABLE_PREFIX}.csv"
  echo "table_json=${TABLE_PREFIX}.json"
} > "${OUTPUT_ROOT}/README_RESULTS.txt"

if (( ${#FAILED_JOBS[@]} > 0 )); then
  printf '%s\n' "${FAILED_JOBS[@]}" > "${OUTPUT_ROOT}/failed_jobs.txt"
  echo "[RoboDojo-full][ERROR] ${#FAILED_JOBS[@]}/${TOTAL_JOBS} jobs failed; see ${OUTPUT_ROOT}/failed_jobs.txt" >&2
  echo "[RoboDojo-full] partial Table-1 summary=${TABLE_PREFIX}.md"
  exit 1
fi

echo "[RoboDojo-full][SUCCESS] all 54 jobs and 2100 episodes completed"
echo "[RoboDojo-full][SUCCESS] Table-1 summary=${TABLE_PREFIX}.md"
echo "[RoboDojo-full][SUCCESS] run manifest=${MANIFEST}"
