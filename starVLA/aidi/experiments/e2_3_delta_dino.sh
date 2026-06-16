#######
# 中文注释：E2.3（对照）—— 不用 CED effect，改用 delta DINO feature 做对齐：future_dino − current_dino。
# teacher=delta_dino（原始 (z_h − z_t) 的范数 softmax 池化，detach）。
# 与 E2.1/E2.2 严格同构：同底座(E1.2)/同 effect 权重(.15)/同 lam_a(.5)/同 projection/同对齐位置/同步数/同 eval，
# 【仅 teacher 不同】。注意：仍开 effect 任务以承载对齐与共享前向，只是对齐目标换成 delta DINO。
#######
CONFIG=examples/LIBERO/train_files/starvla_jointflow_ced_libero.yaml
RUN_NAME=jf_e2_3_delta_dino
OVERRIDES=(
  --framework.ced.enabled=true
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.fdm=0.15
  --framework.tasks.weights.idm=0.15
  --framework.tasks.weights.passive=0.0
  --framework.tasks.weights.effect=0.15
  --framework.losses.teacher=delta_dino
  --framework.losses.lam_a=0.5
  --framework.losses.lam_i=0.0
)
