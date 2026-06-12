#!/usr/bin/env bash
######### // code // ##########
# 中文注释：TriFlow 训练启动脚本。
# 用法: bash starVLA/triflow/scripts/run_triflow_train.sh [config_yaml] [gpus] [extra overrides...]
#   例: bash starVLA/triflow/scripts/run_triflow_train.sh starVLA/triflow/configs/triflow_libero.yaml 0,1
# 注意:
#   - 覆盖参数必须用 --key=value 形式 (normalize_dotlist_args 会丢弃裸 key=value)。
#   - TriFlow 默认 fp32 (TRIFLOW_MIXED_PRECISION=no)，TF32 自动开启。
#   - 训练前先构建词表: python -m starVLA.triflow.data.build_vocab --config_yaml <cfg> \
#         --data_mix libero_object,libero_goal,libero_spatial,libero_10
######### // code // ##########
set -euo pipefail

CONFIG_YAML=${1:-starVLA/triflow/configs/triflow_libero.yaml}
GPUS=${2:-0}
shift $(( $# > 2 ? 2 : $# )) || true

NUM_GPUS=$(awk -F',' '{print NF}' <<< "${GPUS}")
PORT=${TRIFLOW_MASTER_PORT:-29571}

# 中文注释：TRIFLOW_PYTHON 指定解释器（本地未 pip install -e . 时指向 starVLA conda env；
# 集群镜像里 run_aidi.sh 已 pip install，默认 python3 即可）。PYTHONPATH 兜底加 repo 根，
# 保证 `starVLA`/`deployment` 包在未安装环境下也可导入。
PYTHON_BIN=${TRIFLOW_PYTHON:-python3}
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD"

CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${PORT}" \
  --mixed_precision no \
  starVLA/triflow/train/train_triflow.py \
  --config_yaml "${CONFIG_YAML}" \
  "$@"
