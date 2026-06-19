#######
# 中文注释：RoboTwin E2 WAM 四范式 + trick——correlated noise + multi-step FM(20)。RoboTwin 双臂(14D/horizon16/abs)。
#######
CONFIG=examples/Robotwin/train_files/starvla_qwenwam_robotwin.yaml
RUN_NAME=robotwin_wam_e2_trick
OUT_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_qwenwam_robotwin
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.passive=0.5
  --framework.tasks.weights.fdm=0.5
  --framework.tasks.weights.idm=0.5
  --framework.action_model.use_correlated_noise=true
  --framework.action_model.correlation_beta=0.5
  --framework.action_model.flow_matching_steps=20
)
