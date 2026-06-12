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
GPUS_PER_WORKER="${JOINTFLOW_PRECOMPUTE_GPUS_PER_WORKER:-${GPU_PER_WORKER:-$(nvidia-smi -L | wc -l)}}"
NUM_NODES="${JOINTFLOW_PRECOMPUTE_NUM_NODES:-${WORLD_SIZE:-${WORKER_NUM:-${AIDI_WORKER_NUM:-1}}}}"
NODE_RANK="${JOINTFLOW_PRECOMPUTE_NODE_RANK:-${RANK:-${NODE_RANK:-${WORKER_RANK:-${AIDI_WORKER_INDEX:-${SLURM_PROCID:-0}}}}}}"
TOTAL_SHARDS="${JOINTFLOW_PRECOMPUTE_TOTAL_SHARDS:-$(( NUM_NODES * GPUS_PER_WORKER ))}"

if [[ -n "${JOINTFLOW_DATA_ROOT:-}" ]]; then
  DATA_ROOT_OVERRIDE=(--config_override "datasets.vla_data.data_root_dir=${JOINTFLOW_DATA_ROOT}")
else
  DATA_ROOT_OVERRIDE=()
fi

echo "[precompute-dino-aidi] CONFIG=${CONFIG}"
echo "[precompute-dino-aidi] NUM_NODES=${NUM_NODES} NODE_RANK=${NODE_RANK} GPUS_PER_WORKER=${GPUS_PER_WORKER} TOTAL_SHARDS=${TOTAL_SHARDS}"
echo "[precompute-dino-aidi] DATA_ROOT=${JOINTFLOW_DATA_ROOT:-<from config>} BATCH_SIZE=${BATCH_SIZE} AMP_DTYPE=${AMP_DTYPE} PREFETCH=${PREFETCH}"

PIDS=()
for LOCAL_GPU in $(seq 0 $(( GPUS_PER_WORKER - 1 ))); do
  SHARD_ID=$(( NODE_RANK * GPUS_PER_WORKER + LOCAL_GPU ))
  LOG="/tmp/jointflow_dino_precompute_shard_${SHARD_ID}_of_${TOTAL_SHARDS}.log"
  echo "[precompute-dino-aidi] launch local_gpu=${LOCAL_GPU} shard=${SHARD_ID}/${TOTAL_SHARDS} log=${LOG}"
  CUDA_VISIBLE_DEVICES="${LOCAL_GPU}" python3 -m starVLA.jointflow.data.dino_precompute \
    --config_yaml "${CONFIG}" \
    --batch_size "${BATCH_SIZE}" \
    --amp_dtype "${AMP_DTYPE}" \
    --prefetch "${PREFETCH}" \
    --num_shards "${TOTAL_SHARDS}" \
    --shard_id "${SHARD_ID}" \
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
  echo "[precompute-dino-aidi] at least one shard failed; recent logs:" >&2
  for LOG in /tmp/jointflow_dino_precompute_shard_*_of_${TOTAL_SHARDS}.log; do
    echo "===== ${LOG} =====" >&2
    tail -n 80 "$LOG" >&2 || true
  done
  exit 1
fi

echo "[precompute-dino-aidi] all local shards completed."
