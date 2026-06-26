unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
export WANDB_MODE=offline
###########################################################################################
# add1 = exp5（2B + tricks + 大 batch + policy-only）唯一改动：动作头换成 Wan-初始化版
#        （framework.action_model.backbone=wan + Wan2.2-TI2V-5B 骨干截断/插值初始化，参考 FastWAM）。
# 单变量对照 exp5——其余 CLI override 与 run_libero_train_exp5.sh 完全一致（per_device=16 / steps=60000 / save=5000）。
# 用前：先跑 Z_ws/wan_init/preprocess_wan_action_backbone.py 生成骨干 .pt，上传 bucket，
#       并把 config 里 framework.action_model.wan.init_path 改成该 bucket 路径。
Framework_name=QwenGR00T
freeze_module_list=''
base_vlm=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/Qwen3-VL-2B-Instruct/
config_yaml=./examples/LIBERO/train_files/starvla_wam_add1_2b_wan_init_policy_libero.yaml
libero_data_root=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/DATA/LEBERO/libero/
data_mix=libero_all
run_root_dir=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_libero_w_grad
run_id=0625_wam_add1_2b_wan_init_policy
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
