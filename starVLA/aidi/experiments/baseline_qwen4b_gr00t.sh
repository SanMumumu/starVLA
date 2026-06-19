#######
# 中文注释：官方 QwenGR00T 基线（纯 VLA + 全量FT）——Qwen3-VL-4B + GR00T flow-matching action head 原生路径。
# 不开 jointflow/wam，单一 LIBERO VLA 数据（dataset_py=lerobot_datasets）。作为 jointflow(query-mask)/wam 的公平对照基线。
# OUT_ROOT 单独指向 baseline 输出目录（run_aidi.sh 默认是 wam 目录）。
#######
CONFIG=examples/LIBERO/train_files/starvla_qwen4b_gr00t_libero.yaml
RUN_NAME=baseline_qwen4b_gr00t
OUT_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_qwen4b_gr00t
#######
# 中文注释：baseline 当前显存约 47GiB/80GiB，Qwen3 gradient checkpointing 会大量重算，
# 默认关掉换速度；若 OOM，提交前 export AIDI_BASELINE_GRAD_CKPT=true 回退。
AIDI_BASELINE_GRAD_CKPT="${AIDI_BASELINE_GRAD_CKPT:-false}"
AIDI_BASELINE_BATCH="${AIDI_BASELINE_BATCH:-keep}"
AIDI_BASELINE_REPEATED_DIFFUSION_STEPS="${AIDI_BASELINE_REPEATED_DIFFUSION_STEPS:-1}"
AIDI_BASELINE_FREEZE_QWEN="${AIDI_BASELINE_FREEZE_QWEN:-false}"
#######
OVERRIDES=(
  --framework.qwenvl.enable_gradient_checkpointing "$AIDI_BASELINE_GRAD_CKPT"
  --trainer.gradient_checkpointing "$AIDI_BASELINE_GRAD_CKPT"
  --framework.action_model.repeated_diffusion_steps "$AIDI_BASELINE_REPEATED_DIFFUSION_STEPS"
)

if [[ "$AIDI_BASELINE_BATCH" != "keep" ]]; then
  OVERRIDES+=(--datasets.vla_data.per_device_batch_size "$AIDI_BASELINE_BATCH")
fi

if [[ "$AIDI_BASELINE_FREEZE_QWEN" == "true" || "$AIDI_BASELINE_FREEZE_QWEN" == "1" ]]; then
  OVERRIDES+=(--trainer.freeze_modules qwen_vl_interface)
fi
