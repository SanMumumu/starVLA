# CED E2.2b：lam_i 小扫（0.5）。可选，与 e105 对照。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e106_ced_E2_2b_ident_hi
OVERRIDES=(
  --framework.losses.lam_i=0.5
)
