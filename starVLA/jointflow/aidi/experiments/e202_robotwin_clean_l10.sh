######### // code // ##########
# 中文注释：e202 —— RoboTwin Clean 50 任务主训（DINO-L，动作去噪 10 步）。
# 目的：标准设置（demo_clean 同分布）的 50 任务平均 SR 基线。
# 前置：e201 sanity 通过。评测：start_eval.sh -m demo_clean ... all
######### // code // ##########
CONFIG=starVLA/jointflow/configs/jointflow_robotwin_aidi.yaml
RUN_NAME=jf_e202_robotwin_clean_l10
OVERRIDES=(
  --datasets.vla_data.data_mix=robotwin_clean
)
