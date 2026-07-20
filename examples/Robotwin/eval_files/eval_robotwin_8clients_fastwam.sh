#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:?Please set CKPT}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
VERIFY_PYTHON="${STARVLA_PYTHON:-${ROBOTWIN_PYTHON:-python3}}"
WAM_EXPECTED_PHASE="${WAM_EXPECTED_PHASE:-}"
WAM_EXPECTED_WORLD_TO_ACTION="${WAM_EXPECTED_WORLD_TO_ACTION:-}"

if [[ -n "${WAM_EXPECTED_PHASE}" && \
      "${WAM_EXPECTED_PHASE}" != "predictor_warmup" && \
      "${WAM_EXPECTED_PHASE}" != "gate_ft" ]]; then
    echo "[ERROR] WAM_EXPECTED_PHASE must be predictor_warmup or gate_ft" >&2
    exit 1
fi

verify_args=(
    --checkpoint "${CKPT}"
    --replan-steps "${REPLAN_STEPS}"
)
if [[ -n "${WAM_EXPECTED_PHASE}" ]]; then
    verify_args+=(--expected-wam-phase "${WAM_EXPECTED_PHASE}")
fi
if [[ -n "${WAM_EXPECTED_WORLD_TO_ACTION}" ]]; then
    if [[ "${WAM_EXPECTED_WORLD_TO_ACTION}" != "enabled" && \
          "${WAM_EXPECTED_WORLD_TO_ACTION}" != "disabled" ]]; then
        echo "[ERROR] WAM_EXPECTED_WORLD_TO_ACTION must be enabled or disabled" >&2
        exit 1
    fi
    verify_args+=(--expected-world-to-action "${WAM_EXPECTED_WORLD_TO_ACTION}")
fi
"${VERIFY_PYTHON}" "${SCRIPT_DIR}/verify_fastwam_checkpoint_contract.py" "${verify_args[@]}"

export ROBOTWIN_POLICY_NAME="model2robotwin_fastwam_interface"
export DEPLOY_POLICY_TEMPLATE_PATH="${SCRIPT_DIR}/deploy_policy_fastwam.yml"
export REPLAN_STEPS
export WAM_EXPECTED_PHASE
export WAM_EXPECTED_WORLD_TO_ACTION
export RUN_NAME="${RUN_NAME:-robotwin_fastwam_replan${REPLAN_STEPS}}"

# This launcher is for QwenGR00T/FastWAM checkpoints.  Do not let Co-Flow
# controls exported by an earlier command leak into the generated client YAML.
unset COFLOW_INFERENCE_MODE COFLOW_INFERENCE_HORIZON COFLOW_INFERENCE_SEED

exec bash "${SCRIPT_DIR}/eval_robotwin_8clients_replan.sh" "$@"
