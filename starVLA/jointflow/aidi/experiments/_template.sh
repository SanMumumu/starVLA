######### // code // ##########
# 中文注释：JointFlow 实验文件模板（被 aidi/run_aidi.sh source，不直接执行）。
# 复制本文件 → eNNN_<内容>.sh，编号递增、永不复用；改完在 aidi/README.md 实验台账登记。
# 编号段约定：e1xx = LIBERO；e2xx = RoboTwin；e3xx+ = 新基准。
# 只定义"科学旋钮"三件事；输出根/wandb 由 run_aidi.sh 统一注入。
######### // code // ##########

# 1) 基础配置 yaml（集群实验用 *_aidi/bucket 路径版本）
CONFIG=starVLA/jointflow/configs/jointflow_robotwin_aidi.yaml

# 2) 运行名 = run_id = wandb_run_id = bucket 输出目录名（约定: jf_<实验文件名>）
RUN_NAME=jf_eNNN_template

# 3) 覆盖项（--key=value 形式；裸 key=value 会被 normalize_dotlist_args 丢弃！）
OVERRIDES=(
  # --datasets.vla_data.data_mix=robotwin_clean
  # --trainer.max_train_steps=200000
  # --framework.dino.model_size=vits16
  # --framework.action_model.num_inference_timesteps=10
)
