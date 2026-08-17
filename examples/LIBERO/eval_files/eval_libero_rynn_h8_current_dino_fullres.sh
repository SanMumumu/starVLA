#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The checkpoint embeds current_dino_pool=1; serving reconstructs that contract
# from its saved config. LIBERO continues to execute the complete H8 chunk.
export LIBERO_EVAL_VARIANT="${LIBERO_EVAL_VARIANT:-current_dino_fullres_h8}"

exec bash "${SCRIPT_DIR}/eval_libero.sh" "$@"
