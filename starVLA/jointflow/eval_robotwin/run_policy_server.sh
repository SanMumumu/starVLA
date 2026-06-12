#!/usr/bin/env bash
######### // code // ##########
# 中文注释：JointFlow 版 RoboTwin policy server 启动脚本。
# 与 examples/Robotwin/eval_files/run_policy_server.sh 同名同 CLI（<ckpt> [gpu] [port]），
# 因此本目录里"原样拷贝"的 start_eval.sh 会经 SCRIPT_DIR 自动调到本脚本——
# 唯一差异：起的是 starVLA/jointflow/eval/server_jointflow.py（注册 QwenJointFlow、
# 训练同款 state 归一化(min_max+binary，读 config 的 state_norm_modes)、live DINO-L）。
# 注意：
#   - JointFlow recipe 是 fp32 原生（bf16 会 dtype 失配，v1 血泪教训），默认不开 bf16；
#     除非显式 ROBOTWIN_USE_BF16=1（不建议）。
#   - DINOv3-L 权重 gated/离线时，先 export：
#       JOINTFLOW_DINO_LOADER=hf
#       JOINTFLOW_DINO_WEIGHTS=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/dinov3-vitl16-pretrain-lvd1689m/
######### // code // ##########
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash starVLA/jointflow/eval_robotwin/run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"

use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-0}" == "1" ]]; then
    echo "[WARN] JointFlow recipe is fp32-native; bf16 may cause dtype mismatch." >&2
    use_bf16_flag+=(--use_bf16)
fi

echo "[INFO] Starting JointFlow RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}  port: ${port}"
echo "[INFO] JOINTFLOW_DINO_LOADER=${JOINTFLOW_DINO_LOADER:-<unset>} JOINTFLOW_DINO_WEIGHTS=${JOINTFLOW_DINO_WEIGHTS:-<unset>}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" "${REPO_ROOT}/starVLA/jointflow/eval/server_jointflow.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    "${use_bf16_flag[@]}"
