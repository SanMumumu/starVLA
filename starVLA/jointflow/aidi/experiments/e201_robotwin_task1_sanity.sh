######### // code // ##########
# 中文注释：e201 —— RoboTwin 单任务 sanity（adjust_bottle, Clean）。
# 目的：RoboTwin recipe（DINO-L 在线 + 14 维双臂 + chunk16 + 10 步去噪）的最快失败信号：
#       20k 步单任务必须学出"非零成功率"，否则先排查再上全量。
# 通过门：训练 loss 收敛无 NaN；RoboTwin 仿真 adjust_bottle SR > 0（demo_clean）。
######### // code // ##########
CONFIG=starVLA/jointflow/configs/jointflow_robotwin_aidi.yaml
RUN_NAME=jf_e201_robotwin_task1_sanity
OVERRIDES=(
  --datasets.vla_data.data_mix=robotwin_clean_task1
  --trainer.max_train_steps=20000
  --trainer.save_interval=5000
  --trainer.num_warmup_steps=500
)
