# CED E4.3：胜出臂补种子 17（默认假设胜出臂=E3.1；若不是，把 OVERRIDES 改成胜出臂的）。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e110_ced_E4_3_winner_seed17
OVERRIDES=(
  --framework.losses.lam_i=0.1
  --framework.losses.lam_a=0.1
  --framework.losses.teacher=delta
  --seed=17
)
