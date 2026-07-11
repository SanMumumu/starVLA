#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# FastWAM predicts 32 actions and executes 24 before replanning. StarVLA keeps
# its checkpoint-native chunk (normally 50) and only changes the execution horizon.
export REPLAN_STEPS="${REPLAN_STEPS:-24}"
export RUN_NAME="${RUN_NAME:-robotwin_replan${REPLAN_STEPS}}"

exec bash "${SCRIPT_DIR}/eval_robotwin_8clients_fast.sh" "$@"
