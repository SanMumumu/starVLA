######### // code // ##########
# 中文注释：e002 —— LIBERO 四套件合训（object+goal+spatial+10，triflow 运行期注册的混合）。
# 目的：多套件指令/场景多样性下的全生成式联合训练；与 e001 对比泛化。
# 前置：四套件 latents 均已预计算（本地与 bucket 已确认 object/goal/spatial/10 齐全）。
######### // code // ##########
CONFIG=starVLA/triflow/configs/triflow_libero.yaml
RUN_NAME=triflow_e002_libero_all_b256
OVERRIDES=(
  --datasets.vla_data.data_mix=triflow_libero_all
  --trainer.max_train_steps=120000
)
