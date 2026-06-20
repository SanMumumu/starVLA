unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# exp5 = WAM policy + tricks，换 Qwen3-VL-2B + 大 batch（per_device=16）。
# 与官方 run_libero_train.sh(exp1) 同结构，只换 config_yaml / base_vlm(2B) / batch / steps。
# CLI override 与 yaml 一致：per_device=16、max_train_steps=60000、save_interval=5000（大 batch 略少步 + 留饱和测试）。
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/Qwen3-VL-2B-Instruct/
config_yaml=./examples/LIBERO/train_files/starvla_wam_exp5_2b_bigbs_policy_libero.yaml
libero_data_root=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/DATA/LEBERO/libero/
data_mix=libero_all
run_root_dir=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_libero
run_id=0619_wam_exp5_2b_bigbs_policy
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
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 60000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye
  # --is_debug True
