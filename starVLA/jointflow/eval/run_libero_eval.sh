#!/usr/bin/env bash
######### // code // ##########
# 中文注释：CED ckpt 回传后，一条命令跑完 LIBERO 评测（建配置→资产→N卡 server+client→汇总→收尾）。
# 把踩过的坑全封进来：从 run 自带的 config.full.yaml 建配置（去噪步数随训练=10、ced.enabled 自动对齐
# → strict 加载 probe/proj 不报错）、load_live_backbone=false、fp32 server、libero_env+PYTHONPATH+EGL、
# 端口分配、server 收尾。每个 ckpt 只需把 bucket 的 run 目录整个下载到 Z_ws/CED/<RUN_NAME> 即可。
#
# 用法：
#   bash starVLA/jointflow/eval/run_libero_eval.sh <run_dir> [N] [SUITES] [trials]
#   <run_dir>  含 config.full.yaml 和 ckpt/（或 checkpoints/）下的 steps_*.pt
#   N          用几张卡（默认 8）；SUITES 默认四套件；trials 默认 50
# 例：bash starVLA/jointflow/eval/run_libero_eval.sh /mnt/hwdata/wangsen/starVLA/starVLA/Z_ws/CED/jf_e104_ced_E2_1_ident 8
######### // code // ##########
set -uo pipefail

RUN_DIR="${1:?用法: run_libero_eval.sh <run_dir> [N] [SUITES] [trials]}"
N="${2:-8}"
SUITES="${3:-libero_goal,libero_object,libero_spatial,libero_10}"
TRIALS="${4:-50}"
BASE_PORT="${BASE_PORT:-6500}"

REPO=/mnt/hwdata/wangsen/starVLA/starVLA
STARVLA_PY=/mnt/nodestor/ws/Anaconda3/envs/starVLA/bin/python
LIBERO_PY=/mnt/nodestor/ws/Anaconda3/envs/libero_env/bin/python
LIBERO_DATA=/mnt/hwdata/wangsen/starVLA/DATA/LEBERO/libero
#######
# 中文注释：迁移后 eval 加载 qwen3vl-gr00t 训练时使用的 Qwen3-VL base。
QWEN=/mnt/hwdata/wangsen/starVLA/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
#######
DINO_SNAPSHOT=/mnt/hwdata/wangsen/WAM/LDA/CKPTS/dinov3-vits16-pretrain-lvd1689m   # vits16(384)，CED 全用它
OPENPI_LIBERO=/mnt/nodestor/ws/openpi/third_party/libero

cd "$REPO"
EVAL_DIR="$RUN_DIR/eval"; mkdir -p "$EVAL_DIR"

# ckpt 自动定位（ckpt/ 或 checkpoints/ 下 step 最大的）
CKPT=$(ls -1 "$RUN_DIR"/ckpt/steps_*_pytorch_model.pt "$RUN_DIR"/checkpoints/steps_*_pytorch_model.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
[ -n "$CKPT" ] || { echo "ERROR: 在 $RUN_DIR/{ckpt,checkpoints}/ 找不到 steps_*_pytorch_model.pt"; exit 1; }
[ -f "$RUN_DIR/config.full.yaml" ] || { echo "ERROR: 缺 $RUN_DIR/config.full.yaml（请把 bucket 的 run 目录整个下载）"; exit 1; }
echo "[eval] RUN_DIR=$RUN_DIR"
echo "[eval] CKPT=$CKPT  | N=$N | SUITES=$SUITES | trials=$TRIALS"

# 1) 从 config.full.yaml 建本地 config.yaml（去噪步数/horizon/ced 设置原样保留，只改本地路径 + 关 live DINO 自建）
"$STARVLA_PY" - "$RUN_DIR" "$LIBERO_DATA" "$QWEN" "$DINO_SNAPSHOT" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
rd, data, qwen, dino = sys.argv[1:5]
c = OmegaConf.load(str(Path(rd) / "config.full.yaml"))
c.datasets.vla_data.data_root_dir = data
c.framework.qwenvl.base_vlm = qwen
c.framework.dino.weights = dino          # weights=目录 → 本地 HF 快照加载，不联网
c.framework.dino.loader = "hf"
c.framework.dino.load_live_backbone = False  # eval：先不建 DINO，strict 加载 ckpt 成功后 predict 时 lazy 建
OmegaConf.save(c, str(Path(rd) / "config.yaml"))
print(f"[eval] config.yaml ok | num_inference_timesteps={c.framework.action_model.num_inference_timesteps} "
      f"horizon={c.framework.action_model.action_horizon} "
      f"ced.enabled={c.framework.get('ced',{}).get('enabled') if c.framework.get('ced') else None}")
PY
[ $? -eq 0 ] || { echo "ERROR: 建 config.yaml 失败"; exit 1; }

# 2) 资产（dataset_statistics.json + dino_v3_stats.json）
"$STARVLA_PY" -m starVLA.jointflow.eval.prepare_eval_assets --ckpt_path "$CKPT" || { echo "ERROR: prepare_eval_assets 失败"; exit 1; }

# 3) 起 N 个 server（fp32，不加 --use_bf16）；trap 保证退出时收掉
SERVER_PIDS=()
cleanup() { echo "[eval] 收尾 server..."; for p in "${SERVER_PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i "$STARVLA_PY" -m starVLA.jointflow.eval.server_jointflow \
    --ckpt_path "$CKPT" --port $((BASE_PORT+i)) --idle_timeout 7200 \
    > "$EVAL_DIR/server_${i}.log" 2>&1 &
  SERVER_PIDS+=($!)
done
echo "[eval] 启动 $N 个 server，等待 listening ..."
for _ in $(seq 1 120); do   # 最多等 ~600s
  ready=$(grep -l "server listening" "$EVAL_DIR"/server_*.log 2>/dev/null | wc -l)
  [ "$ready" -ge "$N" ] && break
  sleep 5
done
ready=$(grep -l "server listening" "$EVAL_DIR"/server_*.log 2>/dev/null | wc -l)
[ "$ready" -ge "$N" ] || { echo "ERROR: 仅 $ready/$N 个 server 就绪，查 $EVAL_DIR/server_*.log"; exit 1; }
echo "[eval] $N 个 server 就绪"

# 4) 起 N 个 client 分片（libero_env + PYTHONPATH + EGL）
CLIENT_PIDS=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH="$REPO:$OPENPI_LIBERO" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    "$LIBERO_PY" -m starVLA.jointflow.eval.eval_libero_jointflow \
    --host 127.0.0.1 --port $((BASE_PORT+i)) --task_suites "$SUITES" \
    --num_shards "$N" --shard_id "$i" --num_trials_per_task "$TRIALS" --seed 7 \
    --results_json "$EVAL_DIR/shard${i}.json" \
    > "$EVAL_DIR/client_${i}.log" 2>&1 &
  CLIENT_PIDS+=($!)
done
echo "[eval] 启动 $N 个 client 分片，跑仿真中（看 $EVAL_DIR/client_*.log 进度）..."
# 中文注释：只 wait client（不能裸 wait——server 是 idle_timeout 后台进程，裸 wait 会干等到超时）。
wait "${CLIENT_PIDS[@]}"

# 5) 汇总 + 打印（per-suite + AVG）
"$STARVLA_PY" -m starVLA.jointflow.eval.collect_results --results_dir "$EVAL_DIR" --out "$EVAL_DIR/summary.json"
echo "[eval] 结果 -> $EVAL_DIR/summary.json"
######### // code // ##########
