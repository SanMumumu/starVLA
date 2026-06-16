#######
# 中文注释：E2.1（核心）—— CED / counterfactual effect distillation。
# 用有效性控制表征 effect representation（Δ=v̂(fdm⁺)−v̂(fdm⁰) 的池化向量）与 action DiT 早期层表征对齐。
# 底座=E1.2（policy/fdm/idm/passive=1/.15/.15/0）+ 开 effect 任务承载对齐；teacher=delta（CED effect）。
# 与 E2.2/E2.3 严格同构：同 effect 权重(.15)/同 lam_a(.5)/同 projection(repa_proj)/同对齐位置(repa_layer)/
# 同步数/同 eval 频率（后四者来自共享基座 config），【仅 teacher 不同】。
#######
CONFIG=examples/LIBERO/train_files/starvla_jointflow_ced_libero.yaml
RUN_NAME=jf_e2_1_ced_effect
OVERRIDES=(
  --framework.ced.enabled=true
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.fdm=0.15
  --framework.tasks.weights.idm=0.15
  --framework.tasks.weights.passive=0.0
  --framework.tasks.weights.effect=0.15
  --framework.losses.teacher=delta
  --framework.losses.lam_a=0.5
  --framework.losses.lam_i=0.0
)
