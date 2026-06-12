# CED E2.1：机制引擎臂——effect 任务 + loss_ident（lam_i=0.1，lam_a=0）。
# 门：metric/probe_mse_val 显著低于随机基线；diag_action_gap 的 gap_dyn 随训练上升；SR 不降。
CONFIG=starVLA/jointflow/configs/jointflow_libero_ced.yaml
RUN_NAME=jf_e104_ced_E2_1_ident
OVERRIDES=()
