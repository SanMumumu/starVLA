# CED E1.0：baseline 重跑（对照锚点）。
# 现行 highscore 配方 + 与 CED 各臂对齐的统一改动（步数 100k、horizon 8、去噪 10 步）。评测 libero_10。
CONFIG=starVLA/jointflow/configs/jointflow_libero_highscore_b24.yaml
RUN_NAME=jf_e102_ced_E1_0_baseline
OVERRIDES=(
  --trainer.max_train_steps=100000
  --framework.action_model.action_horizon=8
  --framework.action_model.num_inference_timesteps=10
  --datasets.vla_data.action_horizon=8
  --datasets.vla_data.world_model.future_stride=8
  --datasets.vla_data.online_dino=auto
  --framework.dino.repo_or_dir=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/dinov3_repo
  --framework.dino.weights=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/dinov3-vits16-pretrain-lvd1689m/
)
