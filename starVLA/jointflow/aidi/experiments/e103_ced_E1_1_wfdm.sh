# CED E1.1：change 加权 FDM + 去 passive（C1–C3+C6），不开 effect 任务。
# 门：SR ≥ E1.0；wandb 上 loss/fdm_dynamic 占比上升。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e103_ced_E1_1_wfdm
OVERRIDES=(
  --framework.ced.enabled=false
  --framework.tasks.weights.effect=0.0
  --framework.tasks.weights.fdm=0.25
)
