######### // code // ##########
# 中文注释：e001 —— libero_object 单套件基线（首个全量训练）。
# 目的：验证 6 任务联合训练在真实数据上的收敛与 LIBERO SR 基线。
# 旋钮：全部用 triflow_libero.yaml 默认（60k 步；8 GPU × b16 × accum2 = 全局 256）。
######### // code // ##########
CONFIG=starVLA/triflow/configs/triflow_libero.yaml
RUN_NAME=triflow_e001_libero_object_base
OVERRIDES=()
