# CED E2.2a：lam_i 小扫（0.05）。可选，与 e106 一起判读：取 probe 最优且 loss/act 无恶化者。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e105_ced_E2_2a_ident_lo
OVERRIDES=(
  --framework.losses.lam_i=0.05
)
