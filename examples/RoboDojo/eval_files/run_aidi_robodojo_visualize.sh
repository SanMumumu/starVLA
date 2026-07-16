#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBODOJO_EVAL_MODE=visualize
export ROBODOJO_TRIALS="${ROBODOJO_TRIALS:-5}"
export ROBODOJO_RUN_NAME="${ROBODOJO_RUN_NAME:-visualize_${ROBODOJO_TASK:-stack_bowls}_trials${ROBODOJO_TRIALS}}"

exec bash "${SCRIPT_DIR}/run_aidi_robodojo.sh"
