#!/usr/bin/env bash

# AIDI calls this file directly from the uploaded repository.  Keep an
# immediate first log so a package/path failure is visible without waiting for
# model or IsaacSim initialization.
echo "[AIDI][RoboDojo] entry host=${HOSTNAME:-unknown}"
set -Eeuo pipefail
trap 'rc=$?; echo "[AIDI][RoboDojo][ERROR] line=${BASH_LINENO[0]} rc=${rc}" >&2; exit "${rc}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export STARVLA_ROOT
export ROBODOJO_ROOT="${ROBODOJO_ROOT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboDojo}"
export STARVLA_CKPT_PATH="${STARVLA_CKPT_PATH:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robodojo/starvla_qwengroot_robodojo_baseline_h16_jitx_corrnoise_zscore/checkpoints/steps_80000_pytorch_model.pt}"
export ROBODOJO_REPLAN_STEPS="${ROBODOJO_REPLAN_STEPS:-12}"
if [[ -z "${ROBODOJO_CKPT_NAME:-}" ]]; then
  checkpoint_stem="$(basename "${STARVLA_CKPT_PATH}")"
  checkpoint_stem="${checkpoint_stem%.*}"
  checkpoint_run="$(basename "$(dirname "$(dirname "${STARVLA_CKPT_PATH}")")")"
  export ROBODOJO_CKPT_NAME="${checkpoint_run}_${checkpoint_stem}"
fi

exec bash "${SCRIPT_DIR}/eval_robodojo.sh" \
  RoboDojo \
  "${ROBODOJO_TASK:-stack_bowls}" \
  "${ROBODOJO_CKPT_NAME}" \
  "${ROBODOJO_ENV_CFG:-arx_x5}" \
  joint \
  "${ROBODOJO_SEED:-0}" \
  "${ROBODOJO_POLICY_GPU:-0}" \
  "${ROBODOJO_ENV_GPU:-1}" \
  "${ROBODOJO_POLICY_ENV:-starVLA}" \
  "${ROBODOJO_EVAL_ENV:-RoboDojo}"
