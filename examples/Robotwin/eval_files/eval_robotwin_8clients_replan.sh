#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Current Rynn RoboTwin predicts H50 and executes 20 actions before replanning.
export REPLAN_STEPS="${REPLAN_STEPS:-20}"
export RUN_NAME="${RUN_NAME:-robotwin_replan${REPLAN_STEPS}}"

exec bash "${SCRIPT_DIR}/eval_robotwin_8clients_fast.sh" "$@"
