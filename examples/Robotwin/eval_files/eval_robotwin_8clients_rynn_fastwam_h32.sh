#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Rynn keeps its own model implementation, while its RoboTwin observations,
# state/action order, normalization, and composite image follow FastWAM.
export ROBOTWIN_POLICY_NAME="model2robotwin_fastwam_interface"
export DEPLOY_POLICY_TEMPLATE_PATH="${SCRIPT_DIR}/deploy_policy_fastwam.yml"
export REPLAN_STEPS="${REPLAN_STEPS:-${ROBOTWIN_REPLAN_STEPS:-24}}"
export ROBOTWIN_REPLAN_STEPS="${REPLAN_STEPS}"
export RUN_NAME="${RUN_NAME:-robotwin_rynn_fastwam_h32_replan${REPLAN_STEPS}}"

# These controls apply only to the archived two-stage WAM/Co-Flow models.
unset WAM_EXPECTED_PHASE WAM_EXPECTED_WORLD_TO_ACTION
unset COFLOW_INFERENCE_MODE COFLOW_INFERENCE_HORIZON COFLOW_INFERENCE_SEED

exec bash "${SCRIPT_DIR}/eval_robotwin_8clients_fast.sh" "$@"
