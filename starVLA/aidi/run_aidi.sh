#!/usr/bin/env bash
#######
# 中文注释：QwenGR00T(wam/jointflow/baseline) AIDI 训练入口。走原生 accelerate + train_starvla.py。
# 一个实验 = experiments/ 下一个 .sh（定义 CONFIG / RUN_NAME / OVERRIDES）。backbone 全量训（不冻结）。
# 用法：集群在 jobs/*.yaml 的 RUN_SCRIPTS 末尾带实验名；本地 SKIP_PIP=1 GPUS=0,1 bash run_aidi.sh e1_wam_4task
#######
set -euo pipefail

cd "$(dirname "$0")/../.."
test -f pyproject.toml || { echo "ERROR: not in StarVLA repo root, PWD=$PWD"; exit 1; }

if [[ "${SKIP_PIP:-0}" != "1" ]]; then
  #######
  # 中文注释：集群 docker 自带 numpy 2.x，但镜像里的 pandas/pyarrow 是用 numpy 1.x 编译的，
  # import 时会 "_ARRAY_API not found" 直接崩。强制把 numpy 降回仓库锁定的 1.26.4，装到 --user site
  # （user-site 在 sys.path 中优先于全局 site，运行时即可 shadow 掉镜像的 numpy 2.x）。
  # 这是作业能在集群启动的前提：不静默、不吞错，并回显实际生效版本，便于在日志里核对。
  python3 -m pip install --user --ignore-installed "numpy==1.26.4"
  python3 -c "import numpy; print('[aidi] numpy ->', numpy.__version__)"
  #######
  python3 -m pip install --user swanlab >/dev/null 2>&1 || true
  python3 -m pip install --user --no-deps -e . >/dev/null 2>&1 || true
fi

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-BJrwNBbMCFHZnnok62oXx}"
#######
# 中文注释：wandb 默认离线（集群无外网/避免 wandb_login 因无 key 崩）；key 已配好，离线 run 可事后
# wandb sync 同步，或临时 export WANDB_MODE=online 直传。
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-offline}"
#######
#######
# 中文注释：A800/H20 上 action head 或其它 fp32 matmul 可走 TF32，加速但不改变 bf16 主干；
# 如需严格复现实验数值，export STARVLA_ALLOW_TF32=0。
export STARVLA_ALLOW_TF32="${STARVLA_ALLOW_TF32:-1}"
export STARVLA_FLOAT32_MATMUL_PRECISION="${STARVLA_FLOAT32_MATMUL_PRECISION:-high}"
#######

EXP="${1:?用法: run_aidi.sh <实验名>}"
# shellcheck disable=SC1090
source "starVLA/aidi/experiments/${EXP}.sh"
: "${CONFIG:?experiment must set CONFIG}"
: "${RUN_NAME:?experiment must set RUN_NAME}"
OVERRIDES=("${OVERRIDES[@]:-}")

#######
# 中文注释：torchvision.io.VideoReader 在 torchvision 0.22 起已弃用，0.24 会移除。
# AIDI 配方默认从旧的 torchvision_av 切到仓库内已有的直接 PyAV 解码路径，避免训练日志刷屏，
# 也避免未来镜像升级后接口消失。需要回退/对比时可 export AIDI_VIDEO_BACKEND=torchvision_av；
# 想试 torchcodec 时也可 export AIDI_VIDEO_BACKEND=torchcodec（需镜像已安装且版本匹配）。
AIDI_VIDEO_BACKEND="${AIDI_VIDEO_BACKEND:-pyav}"
if [[ "$AIDI_VIDEO_BACKEND" != "keep" ]]; then
  OVERRIDES+=(--datasets.vla_data.video_backend "$AIDI_VIDEO_BACKEND")
fi
#######

#######
# 中文注释：8 卡训练时 4 workers/GPU = 32 个 PyAV/FFmpeg 解码 worker，LIBERO 在线读视频容易在
# worker 内触发 av.error.MemoryError: Cannot allocate memory。模型前后向才是瓶颈时，降 worker
# 基本不影响吞吐，但能明显降低容器 CPU 内存峰值。可按需 export AIDI_NUM_WORKERS=2/4 回调。
AIDI_NUM_WORKERS="${AIDI_NUM_WORKERS:-1}"
AIDI_PREFETCH_FACTOR="${AIDI_PREFETCH_FACTOR:-1}"
AIDI_PIN_MEMORY="${AIDI_PIN_MEMORY:-false}"
AIDI_PERSISTENT_WORKERS="${AIDI_PERSISTENT_WORKERS:-false}"
if [[ "$AIDI_NUM_WORKERS" != "keep" ]]; then
  OVERRIDES+=(--datasets.vla_data.num_workers "$AIDI_NUM_WORKERS")
  OVERRIDES+=(--datasets.vla_data.prefetch_factor "$AIDI_PREFETCH_FACTOR")
  OVERRIDES+=(--datasets.vla_data.pin_memory "$AIDI_PIN_MEMORY")
  OVERRIDES+=(--datasets.vla_data.persistent_workers "$AIDI_PERSISTENT_WORKERS")
fi
#######

OUT_ROOT="${OUT_ROOT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_qwenwam}"
GPUS="${GPUS:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
NGPU="$(awk -F, '{print NF}' <<<"$GPUS")"
DS_CONFIG="${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"

#######
# 中文注释：多机支持。AIDI(多 worker)注入标准 PyTorch 分布式变量：MASTER_ADDR / MASTER_PORT /
#   WORLD_SIZE(总进程数=总卡数) / RANK(全局rank) / LOCAL_RANK；每节点 GPU 数 = 本机 NGPU。
#   据此换算 accelerate 多机参数：
#     --num_machines = WORLD_SIZE / NGPU；--machine_rank = 优先 NODE_RANK/GROUP_RANK，否则 RANK/NGPU；
#     --num_processes = WORLD_SIZE(全局总进程)；--main_process_ip/port = MASTER_ADDR/MASTER_PORT。
#   单机(WORLD_SIZE 未设或 ≤ NGPU)退回 --num_processes NGPU，与原行为一致（LIBERO 等单机 job 不受影响）。
#   ⚠️ 若平台把 RANK 设成「节点序号」而非「全局rank」，看下面日志 machine_rank 是否每节点不同；
#      若各节点都是 0，把 --machine_rank 改成 "${RANK}" 即可。
WS="${WORLD_SIZE:-${NGPU}}"
if [[ "${WS}" -gt "${NGPU}" ]]; then
  NNODES=$(( WS / NGPU ))
  NODE_RANK="${NODE_RANK:-${GROUP_RANK:-$(( ${RANK:-0} / NGPU ))}}"
  LAUNCH_ARGS=(
    --num_machines "${NNODES}"
    --machine_rank "${NODE_RANK}"
    --main_process_ip "${MASTER_ADDR:?多机需要 MASTER_ADDR(平台未注入)}"
    --main_process_port "${MASTER_PORT:-29500}"
    --num_processes "${WS}"
  )
  echo "[aidi][multinode] WORLD_SIZE=${WS} NGPU/node=${NGPU} RANK=${RANK:-0} -> num_machines=${NNODES} machine_rank=${NODE_RANK} num_processes=${WS} master=${MASTER_ADDR}:${MASTER_PORT:-29500}"
else
  LAUNCH_ARGS=(--num_processes "${NGPU}")
  echo "[aidi][singlenode] num_processes=${NGPU}"
fi
#######

echo "[aidi] EXP=${EXP} RUN_NAME=${RUN_NAME} CONFIG=${CONFIG} GPUS=${GPUS} OUT_ROOT=${OUT_ROOT} VIDEO_BACKEND=${AIDI_VIDEO_BACKEND} NUM_WORKERS=${AIDI_NUM_WORKERS}"

export CUDA_VISIBLE_DEVICES="$GPUS"
exec accelerate launch \
  --config_file "$DS_CONFIG" \
  "${LAUNCH_ARGS[@]}" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --run_root_dir "$OUT_ROOT" \
  --run_id "$RUN_NAME" \
  ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
  "${@:2}"
