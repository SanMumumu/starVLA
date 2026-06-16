#######
# 中文注释：E1.2 —— 去掉 passive，把辅助权重并到 fdm/idm（总量≈持平）。不开 CED/effect、不做对齐。
# 这是 E2.1/E2.2/E2.3 三个对齐实验的【共同底座】。
#######
CONFIG=examples/LIBERO/train_files/starvla_jointflow_ced_libero.yaml
RUN_NAME=jf_e1_2_no_passive
OVERRIDES=(
  --framework.ced.enabled=false
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.fdm=0.15
  --framework.tasks.weights.idm=0.15
  --framework.tasks.weights.passive=0.0
  --framework.tasks.weights.effect=0.0
  --framework.losses.lam_a=0.0
  --framework.losses.lam_i=0.0
)
