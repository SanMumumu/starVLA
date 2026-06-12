######### // code // ##########
# 中文注释：TriFlow 实验文件模板（被 run_aidi.sh source，不直接执行）。
# 复制本文件 → eNNN_<内容>.sh，编号递增、永不复用；改完在 aidi/README.md 实验台账登记。
# 只定义"科学旋钮"三件事；数据根/输出根/wandb 等基建由 run_aidi.sh 统一注入。
######### // code // ##########

# 1) 基础配置 yaml
CONFIG=starVLA/triflow/configs/triflow_libero.yaml

# 2) 运行名 = run_id = wandb_run_id = 输出目录名（约定: triflow_<实验文件名>）
RUN_NAME=triflow_eNNN_template

# 3) 覆盖项（--key=value 形式；裸 key=value 会被 normalize_dotlist_args 丢弃！）
OVERRIDES=(
  # --datasets.vla_data.data_mix=triflow_libero_all
  # --trainer.max_train_steps=120000
  # --framework.tasks.weights.joint=1.0
  # --framework.flow.noise_scale.text=1.0
)
