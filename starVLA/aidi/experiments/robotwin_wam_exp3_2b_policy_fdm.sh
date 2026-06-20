#######
# 中文注释：RoboTwin exp3 = exp2(2B+tricks+大bs+policy) 再加 FDM(预测 delta-DINO)。与 exp2 比「加 fdm 世界模型监督」增益。
# fdm + delta-DINO 在 yaml 里（tasks.weights.fdm=0.5、wam.fdm_delta_dino=true）。fdm 在线跑 DINO 更吃显存，OOM 就把 per_device 降到 4。
#######
CONFIG=examples/Robotwin/train_files/starvla_wam_robotwin_exp3_2b_policy_fdm.yaml
RUN_NAME=robotwin_wam_exp3_2b_policy_fdm
OUT_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.fdm=0.5
  --framework.wam.fdm_delta_dino=true
  --framework.action_model.use_correlated_noise=true
  --framework.action_model.flow_matching_steps=20
  --datasets.vla_data.data_mix=robotwin_all_32
  --datasets.vla_data.per_device_batch_size=8
  --trainer.gradient_accumulation_steps=1
  --trainer.max_train_steps=60000
  --trainer.save_interval=5000
)
