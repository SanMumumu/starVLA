# CED E3.1：核心臂——Δ teacher 蒸馏（lam_a=0.1，teacher=delta）。
# 前置：E2.x 判读后把 lam_i 改成胜出值（默认 0.1）。Phase 3 前不要提交本实验（杀手#4）。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e107_ced_E3_1_effect_teacher
OVERRIDES=(
  --framework.losses.lam_i=0.1
  --framework.losses.lam_a=0.1
  --framework.losses.teacher=delta
)
