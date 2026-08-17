#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The dense-current-DINO variant keeps the same H32 action and t+32 future
# contracts as the control; evaluation replans after 24 actions (Fast-WAM default).
export REPLAN_STEPS="${REPLAN_STEPS:-${ROBOTWIN_REPLAN_STEPS:-24}}"
export ROBOTWIN_REPLAN_STEPS="${REPLAN_STEPS}"
export RUN_NAME="${RUN_NAME:-robotwin_rynn_fastwam_h32_current_dino_fullres_replan${REPLAN_STEPS}}"

exec bash "${SCRIPT_DIR}/eval_robotwin_8clients_rynn_fastwam_h50.sh" "$@"
