#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_YAML="examples/LIBERO/train_files/rynn_base_h8_current_dino_fullres_50k.yaml"

cd "${REPO_ROOT}"
export CONFIG_YAML

echo "[LIBERO fullres] current DINO=16x32/512 tokens; future DINO=8x16/128 tokens"
exec bash examples/LIBERO/train_files/run_libero_train.sh "$@"
