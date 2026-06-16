#!/usr/bin/env bash
#######
# 中文注释：JointFlow/CED（原生 QwenGR00T + Qwen3-VL）AIDI 训练入口。走原生 accelerate + train_starvla.py。
# 一个实验 = experiments/ 下一个 .sh（定义 CONFIG / RUN_NAME / OVERRIDES）。backbone 全量训（不冻结）。
# 用法：集群在 jobs/*.yaml 的 RUN_SCRIPTS 末尾带实验名；本地 SKIP_PIP=1 GPUS=0,1 bash run_aidi.sh e2_1_ced_effect
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

EXP="${1:?用法: run_aidi.sh <实验名>}"
# shellcheck disable=SC1090
source "starVLA/aidi/experiments/${EXP}.sh"
: "${CONFIG:?experiment must set CONFIG}"
: "${RUN_NAME:?experiment must set RUN_NAME}"
OVERRIDES=("${OVERRIDES[@]:-}")

OUT_ROOT="${OUT_ROOT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_jointflow_ced}"
GPUS="${GPUS:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"
NGPU="$(awk -F, '{print NF}' <<<"$GPUS")"
DS_CONFIG="${DS_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"

echo "[aidi] EXP=${EXP} RUN_NAME=${RUN_NAME} CONFIG=${CONFIG} GPUS=${GPUS} OUT_ROOT=${OUT_ROOT}"

export CUDA_VISIBLE_DEVICES="$GPUS"
exec accelerate launch \
  --config_file "$DS_CONFIG" \
  --num_processes "$NGPU" \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG" \
  --run_root_dir "$OUT_ROOT" \
  --run_id "$RUN_NAME" \
  ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
  "${@:2}"
