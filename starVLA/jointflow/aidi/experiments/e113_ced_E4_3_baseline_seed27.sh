# CED E4.3：E1.0 baseline 补种子 27（配方与 e102 完全一致，仅 seed 不同）。
CONFIG=starVLA/jointflow/configs/jointflow_libero_highscore_b24.yaml
RUN_NAME=jf_e113_ced_E4_3_baseline_seed27
OVERRIDES=(
  --trainer.max_train_steps=100000
  --framework.action_model.action_horizon=8
  --framework.action_model.num_inference_timesteps=10
  --datasets.vla_data.action_horizon=8
  --datasets.vla_data.world_model.future_stride=8
  --datasets.vla_data.online_dino=auto
  --framework.dino.repo_or_dir=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/dinov3_repo
  --framework.dino.weights=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/dinov3-vits16-pretrain-lvd1689m/
  --seed=27
)
