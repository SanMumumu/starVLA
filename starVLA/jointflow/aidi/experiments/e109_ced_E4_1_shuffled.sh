# CED E4.1：负对照——batch 内打乱的 Δ 当 teacher，应 ≈ E3.3（=E2.1，不另训），排除"任意正则"解释。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e109_ced_E4_1_shuffled
OVERRIDES=(
  --framework.losses.lam_i=0.1
  --framework.losses.lam_a=0.1
  --framework.losses.teacher=delta_shuffled
)
