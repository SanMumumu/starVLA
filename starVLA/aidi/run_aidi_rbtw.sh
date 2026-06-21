#!/usr/bin/env bash
#######
# 中文注释：RoboTwin AIDI 多机训练入口（job 的 RUN_SCRIPTS 调它，参数 = config yaml 路径）。
# 关键：AIDI 把 RUN_SCRIPTS 在每个 worker(节点)各跑一次，必须按平台注入的分布式 env 拼出
# accelerate 的多机参数(--num_machines/--machine_rank/--main_process_ip/--main_process_port/--num_processes)，
# 否则每个 worker 各跑「单机 8 卡」(DeepSpeed world_size=8) → 8 个独立 job 写同一 run_dir（污染 ckpt/cache）。
# 仓库无 AIDI 多机约定先例，故本脚本：① 先 dump 相关 env(日志能看到平台到底设了哪些变量)；
# ② 用一条「广覆盖」回退链解析节点数/节点rank/master；③ 打印解析结果，便于核对。
# 真多机日志应出现：[rbtw][multinode] total_gpus=64 + DeepSpeed world_size=64 + Total batch size=512。
#######
set -euo pipefail
cd "$(dirname "$0")/../.."
test -f pyproject.toml || { echo "[rbtw] ERROR: not repo root, PWD=$PWD"; exit 1; }

CONFIG="${1:?用法: run_aidi_rbtw.sh <config_yaml 路径>}"

# ---- 依赖 / 运行期 env（与 run_aidi.sh 同口径）----
if [[ "${SKIP_PIP:-0}" != "1" ]]; then
  python3 -m pip install --user --ignore-installed "numpy==1.26.4" >/dev/null 2>&1 || true
fi
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONUNBUFFERED=1
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-BJrwNBbMCFHZnnok62oXx}"
export STARVLA_ALLOW_TF32="${STARVLA_ALLOW_TF32:-1}"

# ---- ① dump 分布式相关 env（用于发现 AIDI 的真实变量名）----
echo "[rbtw][env] $(env | grep -iE 'rank|world|master|node|worker|machine|nnode|addr|port|mlp|arnold|hostname|gpu' | sort | tr '\n' ' ')"

# ---- ② 解析多机参数（广覆盖回退链）----
NGPU="$(nvidia-smi -L | wc -l)"
# 节点总数：优先平台显式节点变量；否则用 WORLD_SIZE(总进程)/NGPU 推
NNODES="${NNODES:-${NUM_MACHINES:-${WORKER_NUM:-${MLP_WORKER_NUM:-${ARNOLD_WORKER_NUM:-}}}}}"
if [[ -z "${NNODES:-}" ]]; then
  if [[ -n "${WORLD_SIZE:-}" ]] && (( WORLD_SIZE > NGPU )); then NNODES=$(( WORLD_SIZE / NGPU )); else NNODES=1; fi
fi
# 本节点 rank：优先节点级变量；否则用全局 RANK/NGPU 推
NODE_RANK="${NODE_RANK:-${MACHINE_RANK:-${GROUP_RANK:-${WORKER_RANK:-${MLP_ROLE_INDEX:-${ARNOLD_ID:-}}}}}}"
if [[ -z "${NODE_RANK:-}" ]]; then NODE_RANK=$(( ${RANK:-0} / NGPU )); fi
MASTER="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-${ARNOLD_WORKER_0_HOST:-}}}"
MPORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
TOTAL=$(( NNODES * NGPU ))

if (( NNODES > 1 )); then
  [[ -n "$MASTER" ]] || { echo "[rbtw] ERROR: 多机但没拿到 master 地址，照上面 [rbtw][env] 找正确变量名告诉我"; exit 1; }
  LAUNCH_ARGS=(
    --num_machines "$NNODES"
    --machine_rank "$NODE_RANK"
    --main_process_ip "$MASTER"
    --main_process_port "$MPORT"
    --num_processes "$TOTAL"
  )
  echo "[rbtw][multinode] nodes=$NNODES node_rank=$NODE_RANK total_gpus=$TOTAL master=$MASTER:$MPORT (期望 DeepSpeed world_size=$TOTAL)"
else
  LAUNCH_ARGS=(--num_processes "$NGPU")
  echo "[rbtw][singlenode] num_processes=$NGPU  ⚠️ 期望多机却走到单机 = 没识别到节点变量名，把上面 [rbtw][env] 整行发我来锁定"
fi

# ---- ③ 启动（dataloader OOM 防护写死，防 worker 被 Killed → NCCL 连带崩）----
exec accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  "${LAUNCH_ARGS[@]}" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --datasets.vla_data.num_workers 1 \
  --datasets.vla_data.prefetch_factor 1 \
  --datasets.vla_data.pin_memory false \
  --datasets.vla_data.persistent_workers false \
  "${@:2}"
