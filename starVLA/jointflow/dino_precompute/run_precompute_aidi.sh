#!/usr/bin/env bash
######### // code // ##########
# 中文注释：AIDI 多机多卡 DINOv3 离线特征预计算入口。
# 每个 worker 启动本机每张 GPU 一个 Python 进程；全局 shard id =
# node_rank * gpus_per_worker + local_gpu。dino_precompute.py 以 dataset 粒度分片，
# 适合 RoboTwin robotwin_all: 50 Clean + 50 Randomized 共 100 个 dataset。
######### // code // ##########
set -euo pipefail

cd "$(dirname "$0")/../../.."
test -f pyproject.toml || { echo "ERROR: not in StarVLA repo root, PWD=$PWD"; exit 1; }

if [[ "${JOINTFLOW_SKIP_PIP:-0}" != "1" ]]; then
  python3 -m pip install --user --force-reinstall --no-deps \
    numpy==1.26.4 pandas==2.2.3 pyarrow==14.0.1
  python3 -m pip install --user --no-deps -e .
fi

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

CONFIG="${JOINTFLOW_PRECOMPUTE_CONFIG:-starVLA/jointflow/configs/jointflow_robotwin_aidi.yaml}"
BATCH_SIZE="${JOINTFLOW_PRECOMPUTE_BATCH_SIZE:-64}"
AMP_DTYPE="${JOINTFLOW_PRECOMPUTE_AMP_DTYPE:-bf16}"
PREFETCH="${JOINTFLOW_PRECOMPUTE_PREFETCH:-3}"
# 中文注释：动态认领模式（默认开）。对 worker 没起齐 / rank 撞车 / 个别 shard 崩溃都鲁棒：
# 每个 GPU 进程扫全量 data_mix、原子认领未完成的库，只要还有进程在跑就能把所有库补完。
DYNAMIC="${JOINTFLOW_PRECOMPUTE_DYNAMIC:-1}"
GPUS_PER_WORKER="${JOINTFLOW_PRECOMPUTE_GPUS_PER_WORKER:-${GPU_PER_WORKER:-$(nvidia-smi -L | wc -l)}}"
NUM_NODES="${JOINTFLOW_PRECOMPUTE_NUM_NODES:-${WORLD_SIZE:-${WORKER_NUM:-${AIDI_WORKER_NUM:-1}}}}"
NODE_RANK="${JOINTFLOW_PRECOMPUTE_NODE_RANK:-${RANK:-${NODE_RANK:-${WORKER_RANK:-${AIDI_WORKER_INDEX:-${SLURM_PROCID:-0}}}}}}"
TOTAL_SHARDS="${JOINTFLOW_PRECOMPUTE_TOTAL_SHARDS:-$(( NUM_NODES * GPUS_PER_WORKER ))}"

if [[ -n "${JOINTFLOW_DATA_ROOT:-}" ]]; then
  DATA_ROOT_OVERRIDE=(--config_override "datasets.vla_data.data_root_dir=${JOINTFLOW_DATA_ROOT}")
else
  DATA_ROOT_OVERRIDE=()
fi

DYNAMIC_FLAG=()
if [[ "${DYNAMIC}" == "1" ]]; then
  DYNAMIC_FLAG=(--dynamic)
fi

# 中文注释：日志落到“持久且 dev 节点能直接看”的位置。默认写到数据桶（一定挂载），
# 可用 JOINTFLOW_PRECOMPUTE_LOG_DIR 覆盖（例如指向 exp 目录）。文件名带 hostname，
# 即使多 node 的 NODE_RANK 撞车（都当 0、shard id 相同）也不会互相覆盖，便于排查到底起了几台机。
LOG_DIR="${JOINTFLOW_PRECOMPUTE_LOG_DIR:-${JOINTFLOW_DATA_ROOT:+${JOINTFLOW_DATA_ROOT}/_dino_precompute_logs}}"
LOG_DIR="${LOG_DIR:-/tmp/jointflow_dino_logs}"
if ! mkdir -p "${LOG_DIR}" 2>/dev/null; then
  echo "[precompute-dino-aidi] WARN cannot create LOG_DIR=${LOG_DIR}; fallback to /tmp"
  LOG_DIR="/tmp/jointflow_dino_logs"
  mkdir -p "${LOG_DIR}"
fi
HOST="$(hostname 2>/dev/null || echo node)"

echo "[precompute-dino-aidi] CONFIG=${CONFIG}"
echo "[precompute-dino-aidi] NUM_NODES=${NUM_NODES} NODE_RANK=${NODE_RANK} GPUS_PER_WORKER=${GPUS_PER_WORKER} TOTAL_SHARDS=${TOTAL_SHARDS}"
echo "[precompute-dino-aidi] DATA_ROOT=${JOINTFLOW_DATA_ROOT:-<from config>} BATCH_SIZE=${BATCH_SIZE} AMP_DTYPE=${AMP_DTYPE} PREFETCH=${PREFETCH} DYNAMIC=${DYNAMIC}"
echo "[precompute-dino-aidi] LOG_DIR=${LOG_DIR} HOST=${HOST} (per-shard logs persist here; tail them from the dev node)"

PIDS=()
for LOCAL_GPU in $(seq 0 $(( GPUS_PER_WORKER - 1 ))); do
  SHARD_ID=$(( NODE_RANK * GPUS_PER_WORKER + LOCAL_GPU ))
  LOG="${LOG_DIR}/shard_${SHARD_ID}_of_${TOTAL_SHARDS}_${HOST}.log"
  echo "[precompute-dino-aidi] launch local_gpu=${LOCAL_GPU} shard=${SHARD_ID}/${TOTAL_SHARDS} log=${LOG}"
  CUDA_VISIBLE_DEVICES="${LOCAL_GPU}" python3 -m starVLA.jointflow.dino_precompute.dino_precompute \
    --config_yaml "${CONFIG}" \
    --batch_size "${BATCH_SIZE}" \
    --amp_dtype "${AMP_DTYPE}" \
    --prefetch "${PREFETCH}" \
    --num_shards "${TOTAL_SHARDS}" \
    --shard_id "${SHARD_ID}" \
    "${DYNAMIC_FLAG[@]}" \
    "${DATA_ROOT_OVERRIDE[@]}" \
    >"${LOG}" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAIL=1
  fi
done

if [[ "$FAIL" != "0" ]]; then
  echo "[precompute-dino-aidi] at least one local shard process exited non-zero; recent logs:" >&2
  for LOG in "${LOG_DIR}"/shard_*_of_${TOTAL_SHARDS}_${HOST}.log; do
    echo "===== ${LOG} =====" >&2
    tail -n 80 "$LOG" >&2 || true
  done
fi

# 中文注释：动态模式里单个 dataset 失败会被记进进程内 failed 名单、进程仍可能 exit 0，
# 因此无论 FAIL 与否都把 shard 日志里的报错摘要打到主 job 日志，方便定位“为什么补不齐”。
echo "[precompute-dino-aidi] error summary from this node's shard logs (if any):"
grep -hE '\[dynamic\] (dataset|build).* failed|Unable to decode any frame|Traceback \(most recent' \
  "${LOG_DIR}"/shard_*_of_${TOTAL_SHARDS}_${HOST}.log 2>/dev/null | sort | uniq -c | sort -rn | head -n 40 || true

echo "[precompute-dino-aidi] all local shards on NODE_RANK=${NODE_RANK} (HOST=${HOST}) completed (local FAIL=${FAIL})."

######### // code // ##########
# 中文注释：rank0 在所有 node 都可能还在收尾时，做“全局完整性体检”并打印是否全部处理好。
# --check_only 扫描整个 data_mix（不加载 DINO/不占 GPU），complete → 退出 0，不全 → 退出 2。
# rank0 轮询直到 complete 或超时：这同时让 rank0 活到全局完成，缓解“rank0 先退导致整 job 被结束、
# 其它 node 被中断”的问题。其它 rank 不重复体检，跑完自己分片即退出。
VERIFY="${JOINTFLOW_PRECOMPUTE_VERIFY:-1}"
VERIFY_TIMEOUT="${JOINTFLOW_PRECOMPUTE_VERIFY_TIMEOUT:-3600}"
VERIFY_INTERVAL="${JOINTFLOW_PRECOMPUTE_VERIFY_INTERVAL:-60}"

if [[ "${VERIFY}" == "1" && "${NODE_RANK}" == "0" ]]; then
  echo "[precompute-dino-aidi] ===== rank0 GLOBAL completeness check (timeout=${VERIFY_TIMEOUT}s) ====="
  CHECK_LOG="${LOG_DIR}/check_rank0_${HOST}.log"
  START_TS=$(date +%s)
  VERIFY_OK=0
  while true; do
    if python3 -m starVLA.jointflow.dino_precompute.dino_precompute \
        --config_yaml "${CONFIG}" \
        "${DATA_ROOT_OVERRIDE[@]}" \
        --check_only >"${CHECK_LOG}" 2>&1; then
      RC=0
    else
      RC=$?
    fi
    grep -E '^\[check\] complete=' "${CHECK_LOG}" || true
    if [[ "${RC}" == "0" ]]; then
      VERIFY_OK=1
      break
    fi
    NOW_TS=$(date +%s)
    if (( NOW_TS - START_TS >= VERIFY_TIMEOUT )); then
      break
    fi
    echo "[precompute-dino-aidi] not complete yet; re-check in ${VERIFY_INTERVAL}s ..."
    sleep "${VERIFY_INTERVAL}"
  done

  echo "============================================================"
  if [[ "${VERIFY_OK}" == "1" ]]; then
    echo "[precompute-dino-aidi] ✅ ALL DINO latents are complete and ready for training."
  else
    echo "[precompute-dino-aidi] ⚠️ DINO latents are NOT fully complete. Missing/incomplete datasets:"
    grep -E '^\[check\]\[(INCOMPLETE|NO_DATASET_DIR)\]' "${CHECK_LOG}" | head -n 200 || true
    echo "[precompute-dino-aidi] -> re-submit the same job; resume will fill only the gaps."
  fi
  echo "============================================================"

  if [[ "${VERIFY_OK}" != "1" ]]; then
    exit 1
  fi
fi

if [[ "$FAIL" != "0" ]]; then
  exit 1
fi
######### // code // ##########
