#######
# 中文注释：RoboTwin exp2 = WAM policy + tricks + 大bs + Qwen3-VL-2B。与 exp1 比 size（grad_accum=1，global=per_device×总卡）。
#######
CONFIG=examples/Robotwin/train_files/starvla_wam_robotwin_exp2_2b_policy.yaml
RUN_NAME=robotwin_wam_exp2_2b_policy
OUT_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.action_model.use_correlated_noise=true
  --framework.action_model.flow_matching_steps=20
  --datasets.vla_data.data_mix=robotwin_all_32
  --datasets.vla_data.per_device_batch_size=8
  --trainer.gradient_accumulation_steps=1
  --trainer.max_train_steps=60000
  --trainer.save_interval=5000
)
