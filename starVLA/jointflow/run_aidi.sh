#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
test -f pyproject.toml || { echo "ERROR: not in StarVLA repo root, PWD=$PWD"; exit 1; }

python3 -m pip install --user --force-reinstall --no-deps \
  numpy==1.26.4 pandas==2.2.3 pyarrow==14.0.1
python3 -m pip install --user --no-deps -e .

export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export JOINTFLOW_TQDM="${JOINTFLOW_TQDM:-false}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_MODE=online

CONFIG=${CONFIG:-starVLA/jointflow/configs/jointflow_libero_highscore_b24.yaml}
GPUS=${GPUS:-$(seq -s, 0 $(( $(nvidia-smi -L | wc -l) - 1 )))}
RUN_NAME=${1:-qwen_jointflow_libero_highscore_b24}
[[ $# -gt 0 ]] && shift

exec bash starVLA/jointflow/scripts/run_jointflow_train.sh "$CONFIG" "$GPUS" \
  --run_root_dir /horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_jointflow \
  --run_id "$RUN_NAME" \
  --wandb_project starVLA_JointFlow \
  --wandb_entity sanmumumu \
  --wandb_run_id "$RUN_NAME" \
  "$@"
