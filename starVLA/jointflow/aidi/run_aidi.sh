#!/usr/bin/env bash
######### // code // ##########
# 中文注释：JointFlow AIDI 集群训练入口（实验管理范式同 starVLA/triflow/aidi/run_aidi.sh）。
# 实验约定：
#   - 一个实验 = experiments/ 下一个 .sh（只定义 CONFIG / RUN_NAME / OVERRIDES 三件事）
#   - RUN_NAME 同时是 run_id、wandb_run_id、bucket 输出目录名 → 三处天然对齐
# 用法：
#   集群:  job.yaml 的 RUN_SCRIPTS 末尾带实验名（见 aidi/job.yaml 注释）
#   本地:  JOINTFLOW_SKIP_PIP=1 PATH=<starVLA env>/bin:$PATH GPUS=0,1 \
#          bash starVLA/jointflow/aidi/run_aidi.sh e201_robotwin_task1_sanity
#   续训:  追加 JOINTFLOW_RESUME=auto
#   兼容:  旧用法仍可用 —— CONFIG=<yaml> bash run_aidi.sh <run_name>
#          （$1 不是 experiments/ 里的文件名时按旧语义当 RUN_NAME）
######### // code // ##########
set -euo pipefail

cd "$(dirname "$0")/../../.."
test -f pyproject.toml || { echo "ERROR: not in StarVLA repo root, PWD=$PWD"; exit 1; }

if [[ "${JOINTFLOW_SKIP_PIP:-0}" != "1" ]]; then
  python3 -m pip install --user --force-reinstall --no-deps \
    numpy==1.26.4 pandas==2.2.3 pyarrow==14.0.1
  python3 -m pip install --user swanlab
  python3 -m pip install --user --no-deps -e .
fi

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export JOINTFLOW_TQDM="${JOINTFLOW_TQDM:-false}"
# 中文注释：训练指标默认上 SwanLab（wandb 上行不稳）；WANDB_MODE 仍然是模式开关
# （online/offline/disabled，trainer 的 tracker 适配层会映射给 swanlab）。
# 切回 wandb：JOINTFLOW_LOGGER=wandb（此时下面的 WANDB_API_KEY 生效）。
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-BJrwNBbMCFHZnnok62oXx}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE="${WANDB_MODE:-online}"

EXP_DIR="starVLA/jointflow/aidi/experiments"
ARG1="${1:-${EXP:-e201_robotwin_task1_sanity}}"
[[ $# -gt 0 ]] && shift

OVERRIDES=()
if [[ -f "${EXP_DIR}/${ARG1}.sh" ]]; then
  # ---- 新范式：source 实验文件（定义 CONFIG / RUN_NAME / OVERRIDES） ----
  # shellcheck disable=SC1090
  source "${EXP_DIR}/${ARG1}.sh"
  : "${CONFIG:?experiment must set CONFIG}"
  : "${RUN_NAME:?experiment must set RUN_NAME}"
else
  # ---- 旧范式兼容：$1 当 RUN_NAME，CONFIG 走 env（不破坏既有提交习惯） ----
  CONFIG="${CONFIG:-starVLA/jointflow/configs/jointflow_libero_highscore_b24.yaml}"
  RUN_NAME="${ARG1}"
  echo "[jointflow-aidi] '${ARG1}' is not an experiment file; legacy mode (CONFIG=${CONFIG})"
fi

OUT_ROOT="${JOINTFLOW_OUT_ROOT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_jointflow}"
GPUS="${GPUS:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}"

# 中文注释：本地复现 *_aidi 配置时用 env 覆盖 bucket 路径（与 triflow/aidi 同款钩子）：
#   JOINTFLOW_DATA_ROOT  → datasets.vla_data.data_root_dir
#   JOINTFLOW_DINO_WEIGHTS_TRAIN → framework.dino.weights（本地走 torch.hub/HF 缓存时设 null）
EXTRA_OVERRIDES=()
if [[ -n "${JOINTFLOW_DATA_ROOT:-}" ]]; then
  EXTRA_OVERRIDES+=(--datasets.vla_data.data_root_dir="${JOINTFLOW_DATA_ROOT}")
fi
if [[ -n "${JOINTFLOW_DINO_WEIGHTS_TRAIN:-}" ]]; then
  EXTRA_OVERRIDES+=(--framework.dino.weights="${JOINTFLOW_DINO_WEIGHTS_TRAIN}")
fi

echo "[jointflow-aidi] EXP=${ARG1} RUN_NAME=${RUN_NAME} CONFIG=${CONFIG} GPUS=${GPUS}"
echo "[jointflow-aidi] OUT_ROOT=${OUT_ROOT} DATA_ROOT=${JOINTFLOW_DATA_ROOT:-<from config>}"

exec bash starVLA/jointflow/scripts/run_jointflow_train.sh "$CONFIG" "$GPUS" \
  --run_root_dir "$OUT_ROOT" \
  --run_id "$RUN_NAME" \
  --wandb_project starVLA_JointFlow \
  --wandb_entity sanmumumu \
  --wandb_run_id "$RUN_NAME" \
  ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
  ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
  "$@"
