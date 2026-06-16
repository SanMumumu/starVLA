#######
# 中文注释：E1.1 —— 四任务联合训练（policy/fdm/idm/passive），不开 CED/effect、不做对齐。
# 复用原生基座 config（examples/LIBERO/train_files/starvla_jointflow_ced_libero.yaml），仅用 OVERRIDES 改"科学旋钮"。
#######
CONFIG=examples/LIBERO/train_files/starvla_jointflow_ced_libero.yaml
RUN_NAME=jf_e1_1_fourtask
OVERRIDES=(
  --framework.ced.enabled=false
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.fdm=0.1
  --framework.tasks.weights.idm=0.1
  --framework.tasks.weights.passive=0.1
  --framework.tasks.weights.effect=0.0
  --framework.losses.lam_a=0.0
  --framework.losses.lam_i=0.0
)
