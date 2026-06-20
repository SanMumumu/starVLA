#######
# 中文注释：RoboTwin baseline = QwenOFT 纯 VLA（关 llava co-train）。robotwin_all_32 (100任务 @ horizon32)。
# 与 WAM exp1/2/3 对照 method。配置自洽（per_device=4 / grad_accum=1 在 yaml 里），OVERRIDES 仅显式钉关键项防漂移。
#######
CONFIG=examples/Robotwin/train_files/starvla_qwenoft_robotwin.yaml
RUN_NAME=robotwin_oft_baseline
OUT_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin
OVERRIDES=(
  --framework.name=QwenOFT
  --datasets.vla_data.data_mix=robotwin_all_32
  --datasets.vla_data.per_device_batch_size=4
  --trainer.gradient_accumulation_steps=1
  --trainer.max_train_steps=100000
  --trainer.save_interval=5000
)
