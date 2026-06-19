#######
# 中文注释：E1 WAM 四范式（创新点1）——policy + passive + fdm + idm 全开（config 默认权重）。
#   policy(图→动作) / passive(图→未来DINO) / fdm(图+动作→未来DINO) / idm(图+未来图→动作)。
# 两组 query 依托不同范式（act-query→policy/idm，flow-query→passive/fdm）。tricks 关。vs E0 看多范式增益。
#######
CONFIG=examples/LIBERO/train_files/starvla_qwenwam_libero.yaml
RUN_NAME=wam_e1_4task
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.passive=0.5
  --framework.tasks.weights.fdm=0.5
  --framework.tasks.weights.idm=0.5
  --framework.action_model.use_correlated_noise=false
  --framework.action_model.flow_matching_steps=1
)
