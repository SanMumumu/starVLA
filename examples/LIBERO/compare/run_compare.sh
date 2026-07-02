#!/bin/bash
# World→Action guidance 对比矩阵统一启动脚本（plan §9）。
# 用法：
#   bash examples/LIBERO/compare/run_compare.sh <exp_name> [init_checkpoint]
# 例：
#   bash examples/LIBERO/compare/run_compare.sh m0q_dual_query_control
#   bash examples/LIBERO/compare/run_compare.sh warmup_predicted_absolute /path/to/cmp_warmup_oracle_absolute   # Stage2 接 Stage1
# 所有实验共享 base_compare.yaml 的数据/seed/batch/步数/lr/DiT/query 数/评测；只改 World 信号或注入结构。
set -e

unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE=offline

EXP_NAME=${1:?"用法: run_compare.sh <exp_name> [init_checkpoint]；exp_name 见 experiment_manifest.yaml"}
INIT_CKPT=${2:-}
# 训练步数可经 env 覆盖（结构后训练从 oracle 热启时常用 50000=原来一半）。默认 100000。
MAX_STEPS=${MAX_STEPS:-100000}

REPO_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
cd "${REPO_DIR}"
CMP_DIR=examples/LIBERO/compare
CONFIG_YAML=${CMP_DIR}/configs/${EXP_NAME}.yaml

# 配置不存在则自动生成（base + manifest → 完整 yaml）
if [ ! -f "${CONFIG_YAML}" ]; then
  echo "[run_compare] 未找到 ${CONFIG_YAML}，先生成..."
  python ${CMP_DIR}/generate_configs.py "${EXP_NAME}"
fi

# run_id：配置里的 run_id 加日期前缀
DATE=$(date +%m%d)
BASE_RUN_ID=$(python -c "from omegaconf import OmegaConf;print(OmegaConf.load('${CONFIG_YAML}').get('run_id','${EXP_NAME}'))")
RUN_ID=${DATE}_${BASE_RUN_ID}

base_vlm=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/Qwen3-VL-2B-Instruct/
libero_data_root=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/DATA/LEBERO/libero/
run_root_dir=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_compare

output_dir=${run_root_dir}/${RUN_ID}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

# warm-up 链接：给了 init_checkpoint 就经 CLI 注入 pretrained_checkpoint（strict=False，新 guidance 子模块 fresh init）
PRETRAIN_ARG=""
if [ -n "${INIT_CKPT}" ]; then
  # 允许传「run 目录」：自动定位 ckpt 文件（load_pretrained_backbones 要文件、不要目录）。
  # 优先 final_model，其次 checkpoints/ 里步数最大的 steps_*。
  if [ -d "${INIT_CKPT}" ]; then
    CKPT_FILE=""
    for cand in "${INIT_CKPT}/final_model/pytorch_model.pt" "${INIT_CKPT}/final_model/model.safetensors"; do
      [ -f "$cand" ] && CKPT_FILE="$cand" && break
    done
    if [ -z "$CKPT_FILE" ]; then
      CKPT_FILE=$(ls -1 "${INIT_CKPT}"/checkpoints/steps_*_pytorch_model.pt "${INIT_CKPT}"/checkpoints/steps_*_model.safetensors 2>/dev/null \
        | sed -E 's/.*steps_([0-9]+).*/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    [ -z "$CKPT_FILE" ] && { echo "[run_compare] ❌ 目录 ${INIT_CKPT} 下找不到 ckpt 文件（final_model/ 或 checkpoints/steps_*）"; exit 1; }
    INIT_CKPT="$CKPT_FILE"
  fi
  PRETRAIN_ARG="--trainer.pretrained_checkpoint ${INIT_CKPT}"
  echo "[run_compare] 从 init checkpoint 续训: ${INIT_CKPT}"
fi

echo "[run_compare] exp=${EXP_NAME} run_id=${RUN_ID} config=${CONFIG_YAML}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${CONFIG_YAML} \
  --framework.name QwenGR00T \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 20 \
  --trainer.max_train_steps ${MAX_STEPS} \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100000 \
  ${PRETRAIN_ARG} \
  --run_root_dir ${run_root_dir} \
  --run_id ${RUN_ID} \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye
