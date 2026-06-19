#######
# 中文注释：E2 WAM 四范式 + trick（behavior-1k 冠军方案）——E1 四范式底座 + correlated noise + multi-step FM。
#   correlated noise：use_correlated_noise=true + correlation_beta=0.5（Σ trainer 启动从数据算 Cholesky 落盘+注入）。
#   multi-step FM：flow_matching_steps=20（behavior-1k 用 15，适当增大）。
# 消融：flow_matching_steps=1（只留 correlated）或 use_correlated_noise=false（只留 multi-step）。vs E1 看 trick 增益。
#######
CONFIG=examples/LIBERO/train_files/starvla_qwenwam_libero.yaml
RUN_NAME=wam_e2_trick
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.passive=0.5
  --framework.tasks.weights.fdm=0.5
  --framework.tasks.weights.idm=0.5
  --framework.action_model.use_correlated_noise=true
  --framework.action_model.correlation_beta=0.5
  --framework.action_model.flow_matching_steps=20
)
