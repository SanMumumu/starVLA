#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROBODOJO_EVAL_MODE=fast
export ROBODOJO_TRIALS="${ROBODOJO_TRIALS:-native}"
export ROBODOJO_RUN_NAME="${ROBODOJO_RUN_NAME:-fast_${ROBODOJO_TASK:-stack_bowls}_${ROBODOJO_TRIALS}}"

exec bash "${SCRIPT_DIR}/run_aidi_robodojo.sh"
