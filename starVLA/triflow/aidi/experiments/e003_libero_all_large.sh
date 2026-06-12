######### // code // ##########
# 中文注释：e003 —— 大模型（ELF-M 档：24×1056×16 ≈330M）四套件合训。
# 目的：验证全生成式配方随规模的收益（对照 e002 基线 88M）。
# 旋钮：全部在 triflow_libero_large.yaml 里（lr 8e-5 / warmup 4k / 100k 步 / 全局 batch 256）。
######### // code // ##########
CONFIG=starVLA/triflow/configs/triflow_libero_large.yaml
RUN_NAME=triflow_e003_libero_all_large
OVERRIDES=()
