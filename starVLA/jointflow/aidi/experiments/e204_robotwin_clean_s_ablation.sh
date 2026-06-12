######### // code // ##########
# 中文注释：e204（可选消融）—— DINO-S 对照（其余与 e202 完全一致）。
# 目的：量化 DINO-L(1024) 相对 DINO-S(384) 在 RoboTwin 上的收益，回答"L 是否值得 3× 提特征开销"。
# 跑法：与 e202 同期或之后；只比较 demo_clean SR 与收敛曲线。
######### // code // ##########
CONFIG=starVLA/jointflow/configs/jointflow_robotwin_aidi.yaml
RUN_NAME=jf_e204_robotwin_clean_s_ablation
OVERRIDES=(
  --datasets.vla_data.data_mix=robotwin_clean
  --framework.dino.model_size=vits16
)
