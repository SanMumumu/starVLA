# CED E4.3：胜出臂补种子 27（同 e110 说明）。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e111_ced_E4_3_winner_seed27
OVERRIDES=(
  --framework.losses.lam_i=0.1
  --framework.losses.lam_a=0.1
  --framework.losses.teacher=delta
  --seed=27
)
