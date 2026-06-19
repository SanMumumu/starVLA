#######
# 中文注释：RoboTwin E0 WAM 最简——policy + passive（fdm/idm=0），无 trick。RoboTwin 双臂(14D/horizon16/abs)。
# OUT_ROOT 单独指向 RoboTwin wam 输出目录。
#######
CONFIG=examples/Robotwin/train_files/starvla_qwenwam_robotwin.yaml
RUN_NAME=robotwin_wam_e0_base
OUT_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_qwenwam_robotwin
OVERRIDES=(
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.passive=0.5
  --framework.tasks.weights.fdm=0.0
  --framework.tasks.weights.idm=0.0
  --framework.action_model.use_correlated_noise=false
  --framework.action_model.flow_matching_steps=1
)
