# CED E3.2：对照臂——不减法的 pooled-future teacher（原 RepAlign 式），lam_i=0。
# 判读：E3.1 > E3.2 ⇒ "对齐的是动作效果而非未来本身" claim 成立。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e108_ced_E3_2_future_teacher
OVERRIDES=(
  --framework.losses.lam_i=0.0
  --framework.losses.lam_a=0.1
  --framework.losses.teacher=pooled_future
)
