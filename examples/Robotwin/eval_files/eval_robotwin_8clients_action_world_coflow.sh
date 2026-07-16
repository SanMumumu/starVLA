#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:?Please set CKPT}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
VERIFY_PYTHON="${STARVLA_PYTHON:-${ROBOTWIN_PYTHON:-python3}}"

"${VERIFY_PYTHON}" "${SCRIPT_DIR}/verify_action_world_coflow_checkpoint_contract.py" \
    --checkpoint "${CKPT}" \
    --replan-steps "${REPLAN_STEPS}"

export ROBOTWIN_POLICY_NAME="model2robotwin_fastwam_interface"
export DEPLOY_POLICY_TEMPLATE_PATH="${SCRIPT_DIR}/deploy_policy_fastwam.yml"
export REPLAN_STEPS
export RUN_NAME="${RUN_NAME:-robotwin_action_world_coflow_replan${REPLAN_STEPS}}"

exec bash "${SCRIPT_DIR}/eval_robotwin_8clients_replan.sh" "$@"
