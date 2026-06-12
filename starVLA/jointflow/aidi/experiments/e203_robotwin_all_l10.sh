######### // code // ##########
# 中文注释：e203 —— RoboTwin 全量主实验（Clean + Randomized 共 100 数据集，DINO-L，10 步去噪）。
# 目的：含域随机化数据的完整训练；demo_clean 与 demo_randomized 双轨评测。
# 前置：e202 Clean SR 合理（量级对得上同配方基线）。数据 ~27.5k episodes，步数加倍。
######### // code // ##########
CONFIG=starVLA/jointflow/configs/jointflow_robotwin_aidi.yaml
RUN_NAME=jf_e203_robotwin_all_l10
OVERRIDES=(
  --trainer.max_train_steps=200000
)
