#######
# 中文注释：E1.3 trick —— 在 E1.2 底座（四任务去 passive、无 CED）上叠加 behavior-1k 冠军方案的两个 trick，
# 合并成一个实验、做成开关方便消融：
#   correlated noise：use_correlated_noise=true + correlation_beta=0.5（Σ 训练启动时从数据算）
#   multi-step Flow Matching：flow_matching_steps=15（每 VLM step 跑 15 次 FM 预测平均，降方差）
# 消融：单独关某个 trick 即可——
#   只留 correlated noise：复制本文件，把 flow_matching_steps 改 1
#   只留 multi-step：把 use_correlated_noise 改 false
#######
CONFIG=examples/LIBERO/train_files/starvla_jointflow_ced_libero.yaml
RUN_NAME=jf_e1_3_trick
OVERRIDES=(
  --framework.ced.enabled=false
  --framework.tasks.weights.policy=1.0
  --framework.tasks.weights.fdm=0.15
  --framework.tasks.weights.idm=0.15
  --framework.tasks.weights.passive=0.0
  --framework.tasks.weights.effect=0.0
  --framework.losses.lam_a=0.0
  --framework.losses.lam_i=0.0
  --framework.action_model.use_correlated_noise=true
  --framework.action_model.correlation_beta=0.5
  --framework.action_model.flow_matching_steps=15
)
