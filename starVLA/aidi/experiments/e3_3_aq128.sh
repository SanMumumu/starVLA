#######
# 中文注释：E3.3（action query num 消融，与 E1.2 对齐：四任务、无 CED）—— 本实验 n_action_query=128。
# 仅放大 action query token 数（conditioning 容量，与 chunk=8 解耦），其余与 E1.2 完全一致。
# baseline=16（E1.2 默认/隐式 baseline，不单列）；本系列从 32 往上扫 32/64/128（看是否饱和）。
#######
CONFIG=examples/LIBERO/train_files/starvla_jointflow_ced_libero.yaml
RUN_NAME=jf_e3_3_aq128
OVERRIDES=(
  --framework.ced.enabled=false
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.fdm=0.15
  --framework.tasks.weights.idm=0.15
  --framework.tasks.weights.passive=0.0
  --framework.tasks.weights.effect=0.0
  --framework.losses.lam_a=0.0
  --framework.losses.lam_i=0.0
  --framework.action_model.n_action_query=128
)
