#######
# 中文注释：E0 WAM policy-only 验证——先只看 act-query→action 能不能拟合好。
# passive/fdm/idm 全关，不让未来 DINO 监督干扰 policy 拟合判断；tricks 关。
#######
CONFIG=examples/LIBERO/train_files/starvla_qwenwam_libero.yaml
RUN_NAME=wam_e0_base
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.passive=0.0
  --framework.tasks.weights.fdm=0.0
  --framework.tasks.weights.idm=0.0
  --framework.action_model.use_correlated_noise=false
  --framework.action_model.flow_matching_steps=1
)
