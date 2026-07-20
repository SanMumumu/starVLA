#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:?Please set CKPT}"
REPLAN_STEPS="${REPLAN_STEPS:-16}"
COFLOW_INFERENCE_MODE="${COFLOW_INFERENCE_MODE:-diagonal}"
COFLOW_INFERENCE_HORIZON="${COFLOW_INFERENCE_HORIZON:-16}"
COFLOW_INFERENCE_SEED="${COFLOW_INFERENCE_SEED:-0}"

[[ "${COFLOW_INFERENCE_MODE}" == "policy" || "${COFLOW_INFERENCE_MODE}" == "diagonal" ]] || {
    echo "[ERROR] COFLOW_INFERENCE_MODE must be policy or diagonal" >&2
    exit 1
}
[[ "${COFLOW_INFERENCE_HORIZON}" == "16" ]] || {
    echo "[ERROR] single-bridge COFLOW_INFERENCE_HORIZON must be 16" >&2
    exit 1
}
[[ "${COFLOW_INFERENCE_SEED}" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] COFLOW_INFERENCE_SEED must be a non-negative integer" >&2
    exit 1
}

# The H20 server launch performs the full checkpoint/config audit.  Keep the
# RoboTwin client launcher free of training-only Python dependencies; the live
# adapter still rejects a wrong framework, mode, state ABI, or causal horizon
# from the server metadata before the first action is executed.

export ROBOTWIN_POLICY_NAME="model2robotwin_fastwam_interface"
export DEPLOY_POLICY_TEMPLATE_PATH="${SCRIPT_DIR}/deploy_policy_fastwam.yml"
export REPLAN_STEPS
export COFLOW_INFERENCE_MODE
export COFLOW_INFERENCE_HORIZON
export COFLOW_INFERENCE_SEED
export RUN_NAME="${RUN_NAME:-robotwin_action_world_coflow_${COFLOW_INFERENCE_MODE}_h${COFLOW_INFERENCE_HORIZON}_replan${REPLAN_STEPS}}"

exec bash "${SCRIPT_DIR}/eval_robotwin_8clients_replan.sh" "$@"
