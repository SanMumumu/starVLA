#!/usr/bin/env bash
######### // code // ##########
# 中文注释：TriFlow AIDI 集群入口（仿 starVLA/jointflow/run_aidi.sh）。
# 实验管理约定：
#   - 一个实验 = experiments/ 下一个 .sh（只定义 CONFIG / RUN_NAME / OVERRIDES 三件事）
#   - RUN_NAME 同时是 run_id、wandb_run_id、bucket 输出目录名 → 三处天然对齐
#   - 基建路径（数据根/输出根）由本脚本统一注入，实验文件只放"科学旋钮"
# 用法：
#   集群:  job.yaml 的 RUN_SCRIPTS 末尾带实验名（见 job.yaml 注释）
#   本地:  TRIFLOW_SKIP_PIP=1 TRIFLOW_DATA_ROOT=/mnt/hwdata/wangsen/starVLA/DATA/LEBERO/libero \
#          TRIFLOW_OUT_ROOT=playground/Debug_Checkpoints GPUS=0,1 \
#          bash starVLA/triflow/aidi/run_aidi.sh e001_libero_object_base
#   续训:  追加 TRIFLOW_RESUME=auto（full-state: model+optim+sched+EMA+RNG）
######### // code // ##########
set -euo pipefail

cd "$(dirname "$0")/../../.."
test -f pyproject.toml || { echo "ERROR: not in StarVLA repo root, PWD=$PWD"; exit 1; }

# ---- 依赖安装（集群镜像内；本地调试用 TRIFLOW_SKIP_PIP=1 跳过） ----
if [[ "${TRIFLOW_SKIP_PIP:-0}" != "1" ]]; then
  python3 -m pip install --user --force-reinstall --no-deps \
    numpy==1.26.4 pandas==2.2.3 pyarrow==14.0.1
  python3 -m pip install --user --no-deps -e .
fi

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export TRIFLOW_TQDM="${TRIFLOW_TQDM:-false}"
export TRIFLOW_MIXED_PRECISION="${TRIFLOW_MIXED_PRECISION:-no}"   # TriFlow 全程 fp32（TF32 自动开）
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ---- 选实验：$1 > $EXP > 默认 ----
EXP="${1:-${EXP:-e001_libero_object_base}}"
[[ $# -gt 0 ]] && shift
EXP_FILE="starVLA/triflow/aidi/experiments/${EXP}.sh"
test -f "$EXP_FILE" || { echo "ERROR: experiment file not found: $EXP_FILE"; ls starVLA/triflow/aidi/experiments/; exit 1; }
# shellcheck disable=SC1090
source "$EXP_FILE"   # 必须定义 CONFIG / RUN_NAME；可选 OVERRIDES 数组
: "${CONFIG:?experiment must set CONFIG}"
: "${RUN_NAME:?experiment must set RUN_NAME}"

# ---- 基建路径（集群默认 bucket；本地用 env 覆盖） ----
DATA_ROOT="${TRIFLOW_DATA_ROOT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/DATA/LEBERO/libero}"
OUT_ROOT="${TRIFLOW_OUT_ROOT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_triflow}"
GPUS="${GPUS:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"

# ---- 紧凑词表自愈：缺失则现场扫全部 LIBERO 套件构建（一份词表覆盖所有 mix） ----
VOCAB_PATH="playground/Pretrained_models/triflow_assets/vocab_map_libero.json"
if [[ ! -f "$VOCAB_PATH" ]]; then
  echo "[triflow-aidi] vocab_map missing -> building from all LIBERO suites"
  python3 -m starVLA.triflow.data.build_vocab --config_yaml "$CONFIG" \
    --data_root_dir "$DATA_ROOT" \
    --data_mix libero_object,libero_goal,libero_spatial,libero_10 \
    --out "$VOCAB_PATH"
fi

echo "[triflow-aidi] EXP=$EXP RUN_NAME=$RUN_NAME CONFIG=$CONFIG GPUS=$GPUS"
echo "[triflow-aidi] DATA_ROOT=$DATA_ROOT OUT_ROOT=$OUT_ROOT"

exec bash starVLA/triflow/scripts/run_triflow_train.sh "$CONFIG" "$GPUS" \
  --run_root_dir="$OUT_ROOT" \
  --run_id="$RUN_NAME" \
  --wandb_project=starVLA_TriFlow \
  --wandb_entity=sanmumumu \
  --wandb_run_id="$RUN_NAME" \
  --datasets.vla_data.data_root_dir="$DATA_ROOT" \
  ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
  "$@"
