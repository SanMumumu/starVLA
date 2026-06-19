unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# exp2 = WAM 只有 policy（消融：原生视觉 + act-query metaquery 单独打 action，无世界模型监督）。
# 与官方 run_libero_train.sh(exp1) 同结构，只换 config_yaml / run_id。base_vlm 用 4B 与 exp1 对齐。
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/Qwen3-VL-4B-Instruct/
config_yaml=./examples/LIBERO/train_files/starvla_wam_exp2_policy_libero.yaml
libero_data_root=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/DATA/LEBERO/libero/
data_mix=libero_all
run_root_dir=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_libero
run_id=0618_wam_exp2_policy
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye
  # --is_debug True
