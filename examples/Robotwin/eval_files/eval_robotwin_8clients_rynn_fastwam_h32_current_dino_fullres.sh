#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The dense-current-DINO variant keeps FastWAM's H32/t+32 temporal contract;
# evaluation executes 24 actions before requesting a fresh chunk.
export REPLAN_STEPS="${REPLAN_STEPS:-${ROBOTWIN_REPLAN_STEPS:-24}}"
export ROBOTWIN_REPLAN_STEPS="${REPLAN_STEPS}"
export RUN_NAME="${RUN_NAME:-robotwin_rynn_fastwam_h32_current_dino_fullres_replan${REPLAN_STEPS}}"

exec bash "${SCRIPT_DIR}/eval_robotwin_8clients_rynn_fastwam_h32.sh" "$@"
