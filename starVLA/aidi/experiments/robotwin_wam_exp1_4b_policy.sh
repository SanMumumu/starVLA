#######
# 中文注释：RoboTwin exp1 = WAM policy + tricks + 大bs + Qwen3-VL-4B。与 OFT baseline 比 method。
# tricks(correlated noise + multi-step FM) 已在 yaml 开；大 batch 靠多机多卡（grad_accum=1）。
#######
CONFIG=examples/Robotwin/train_files/starvla_wam_robotwin_exp1_4b_policy.yaml
RUN_NAME=robotwin_wam_exp1_4b_policy
OUT_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.action_model.use_correlated_noise=true
  --framework.action_model.flow_matching_steps=20
  --datasets.vla_data.data_mix=robotwin_all_32
  --datasets.vla_data.per_device_batch_size=4
  --trainer.gradient_accumulation_steps=1
  --trainer.max_train_steps=60000
  --trainer.save_interval=5000
)
