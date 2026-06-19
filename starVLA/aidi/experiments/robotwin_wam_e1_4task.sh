#######
# 中文注释：RoboTwin E1 WAM 四范式（创新点1）——policy+passive+fdm+idm 全开，无 trick。RoboTwin 双臂(14D/horizon16/abs)。
#######
CONFIG=examples/Robotwin/train_files/starvla_qwenwam_robotwin.yaml
RUN_NAME=robotwin_wam_e1_4task
OUT_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_qwenwam_robotwin
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.passive=0.5
  --framework.tasks.weights.fdm=0.5
  --framework.tasks.weights.idm=0.5
  --framework.action_model.use_correlated_noise=false
  --framework.action_model.flow_matching_steps=1
)
