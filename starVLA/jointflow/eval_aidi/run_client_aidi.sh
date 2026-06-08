#!/usr/bin/env bash
# AIDI CLIENT：检查 LIBERO，发现 server，起分片 eval，汇总结果。
# object ckpt 默认只测 libero_object；CKPT 必须和 server 脚本一致。
set -euo pipefail

cd "$(dirname "$0")/../../.."
test -f pyproject.toml || { echo "ERROR: not in StarVLA repo root: $PWD"; exit 1; }
export PYTHONPATH="${PWD}${LIBERO_HOME:+:${LIBERO_HOME}}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
[[ -n "${LIBERO_HOME:-}" && -z "${LIBERO_CONFIG_PATH:-}" ]] && export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"

# ===== USER CONFIG =====
CKPT="${CKPT:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python3}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
TASK_SUITES="${TASK_SUITES:-libero_object}"
NUM_TRIALS="${NUM_TRIALS:-50}"
# =======================
if [[ -z "${CKPT}" ]]; then echo "ERROR: set CKPT."; exit 1; fi

for py in "${LIBERO_PYTHON}" /opt/conda/envs/libero/bin/python /opt/conda/bin/python /usr/local/bin/python /usr/bin/python3 python3 python; do
  command -v "${py}" >/dev/null 2>&1 || continue
  if "${py}" - <<'PY' >/dev/null 2>&1
from libero.libero import benchmark  # noqa: F401
PY
  then
    LIBERO_PYTHON="${py}"
    break
  fi
done

# 中文注释：有些镜像用 editable install，pip metadata 在 site-packages，但源码在 /opt/LIBERO。
LIBERO_SRC="$("${LIBERO_PYTHON}" -m pip show libero 2>/dev/null | awk -F': ' '/Editable project location/ {print $2; exit}')"
if [[ -n "${LIBERO_SRC}" ]]; then
  export PYTHONPATH="${LIBERO_SRC}:${LIBERO_SRC}/libero:${PYTHONPATH}"
  [[ -z "${LIBERO_HOME:-}" ]] && export LIBERO_HOME="${LIBERO_SRC}"
  [[ -z "${LIBERO_CONFIG_PATH:-}" ]] && export LIBERO_CONFIG_PATH="${LIBERO_SRC}/libero"
fi

if ! "${LIBERO_PYTHON}" - <<'PY'
from libero.libero import benchmark  # noqa: F401
import imageio, mujoco, msgpack, websockets  # noqa: F401
print("LIBERO eval env OK")
PY
then
  echo "ERROR: cannot import LIBERO with LIBERO_PYTHON=${LIBERO_PYTHON}"
  echo "DEBUG: python paths:"
  command -v python || true
  command -v python3 || true
  "${LIBERO_PYTHON}" -c 'import sys; print(sys.executable); print(sys.version); print(sys.path)' || true
  "${LIBERO_PYTHON}" -m pip show libero || true
  echo "Check job_client.yaml uses docker.hobot.cc/imagesys/starvla-libero:v1.0, or set LIBERO_PYTHON to the env that has LIBERO."
  exit 1
fi

RUN_DIR="$(cd "$(dirname "${CKPT}")/.." && pwd)"
EVAL_ID="${JOINTFLOW_EVAL_ID:-$(basename "${RUN_DIR}")__$(basename "${CKPT}" .pt)}"
RDV_DIR="${RUN_DIR}/eval/${EVAL_ID}"
echo "[client] TASK_SUITES=${TASK_SUITES} NUM_TRIALS=${NUM_TRIALS} NUM_CLIENTS=${NUM_CLIENTS}"

echo "[client] wait servers.json -> ${RDV_DIR}"
mapfile -t SERVERS < <("${LIBERO_PYTHON}" -m starVLA.jointflow.eval.rendezvous wait \
  --rdv_dir "${RDV_DIR}" --min_servers 1 --timeout "${SERVER_MAX_WAIT:-43200}")
if [[ ${#SERVERS[@]} -eq 0 ]]; then echo "ERROR: no servers found."; exit 1; fi

hosts=(); ports=()
for s in "${SERVERS[@]}"; do hosts+=("${s%% *}"); ports+=("${s##* }"); done

pids=()
for ((i = 0; i < NUM_CLIENTS; i++)); do
  host="${hosts[$((i % ${#hosts[@]}))]}"
  port="${ports[$((i % ${#ports[@]}))]}"
  CUDA_VISIBLE_DEVICES="${i}" "${LIBERO_PYTHON}" -m starVLA.jointflow.eval.eval_libero_jointflow \
    --host "${host}" --port "${port}" --task_suites "${TASK_SUITES}" \
    --num_shards "${NUM_CLIENTS}" --shard_id "${i}" --num_trials_per_task "${NUM_TRIALS}" \
    --results_json "${RDV_DIR}/shard${i}.json" >"${RDV_DIR}/client_${i}.log" 2>&1 &
  pids+=("$!")
  echo "[client] shard ${i}/${NUM_CLIENTS} -> ${host}:${port}"
done

rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
"${LIBERO_PYTHON}" -m starVLA.jointflow.eval.collect_results --results_dir "${RDV_DIR}" --out "${RDV_DIR}/summary.json" || rc=1
touch "${RDV_DIR}/CLIENTS_DONE"  # 中文注释：通知 server 退出。
echo "[client] done rc=${rc}, summary=${RDV_DIR}/summary.json"
exit "${rc}"
