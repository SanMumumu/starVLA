unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/Qwen3-VL-4B-Instruct/
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/DATA/LEBERO/libero/
data_mix=libero_all
run_root_dir=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_qwenwam/gr00t
run_id=0618_libero4in1_qwen3gr00t
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


# num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --num_processes ${num_processes} \
#   starVLA/training/train_starvla.py \
#   --config_yaml ${config_yaml} \
#   --framework.name ${Framework_name} \
#   --framework.qwenvl.base_vlm ${base_vlm} \
#   --datasets.vla_data.data_root_dir ${libero_data_root}\
#   --datasets.vla_data.data_mix ${data_mix} \
#   --datasets.vla_data.per_device_batch_size 8 \
#   --trainer.vla_data.video_backend torchvision_av \
#   --trainer.freeze_modules ${freeze_module_list} \
#   --trainer.max_train_steps 100000 \
#   --trainer.save_interval 10000 \
#   --trainer.logging_frequency 100 \
#   --trainer.eval_interval 100000 \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project starVLA_Libero \
#   --wandb_entity jinhuiye
#   # --is_debug True



##### Multi-Server Multi-GPU training script #####

# 2 machines x 8 GPUs
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
NUM_MACHINES=${SLURM_NNODES:-2}
MACHINE_RANK=${SLURM_NODEID:-${MACHINE_RANK:-0}}
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_MACHINES))

# master address
if [ -z "${MASTER_ADDR}" ]; then
  if [ -n "${SLURM_JOB_NODELIST}" ]; then
    MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
  else
    echo "[ERROR] MASTER_ADDR is not set and SLURM_JOB_NODELIST is empty."
    echo "Please manually export MASTER_ADDR on all nodes."
    exit 1
  fi
fi

MASTER_PORT=${MASTER_PORT:-29500}

# NCCL over TCP, disable IB
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1

# Auto-pick network interface for TCP NCCL
if [ -z "${NCCL_SOCKET_IFNAME}" ]; then
  NCCL_SOCKET_IFNAME=$(ip route get ${MASTER_ADDR} | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1); exit}')
  export NCCL_SOCKET_IFNAME
fi

# NCCL debug / timeout
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export NCCL_DEBUG=INFO

echo "========== MULTI-NODE INFO =========="
echo "HOSTNAME=$(hostname)"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NUM_MACHINES=${NUM_MACHINES}"
echo "MACHINE_RANK=${MACHINE_RANK}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "TOTAL_GPUS=${TOTAL_GPUS}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "====================================="

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${MACHINE_RANK} \
  --num_machines ${NUM_MACHINES} \
  --num_processes ${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 50000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye

##### Multi-Server Multi-GPU training script #####