#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/rynn_base_h32_current_dino_fullres_50k.yaml"
PYTHON_BIN="${PYTHON_BIN:-python3}"

[[ -f "${CONFIG_PATH}" ]] || {
    echo "[RoboTwin fullres][ERROR] missing config: ${CONFIG_PATH}" >&2
    exit 1
}
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "[RoboTwin fullres][ERROR] cannot find ${PYTHON_BIN}" >&2
    exit 1
}

"${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
dino = cfg["framework"]["dino"]
mot = cfg["framework"]["world_action_mot"]
planner = cfg["framework"]["planner"]
action = cfg["framework"]["action_model"]
data = cfg["datasets"]["vla_data"]
trainer = cfg["trainer"]

assert dino["image_size"] == [384, 320]
assert dino["patch_size"] == 16
assert dino["dino_pool"] == 2
assert dino["current_dino_pool"] == 1
assert mot["world_grid_height"] == 12
assert mot["world_grid_width"] == 10
assert mot["max_world_tokens"] == 120
assert planner["num_action_queries"] == 32
assert action["action_horizon"] == 32
assert data["data_mix"] == "robotwin_fastwam"
assert data["action_horizon"] == 32
assert data["world_model"]["future_stride"] == 32
assert data["per_device_batch_size"] == 12
assert trainer["expected_global_batch_size"] == 768
assert trainer["gradient_accumulation_steps"] == 1
assert trainer["max_train_steps"] == 50000
print(
    "[RoboTwin fullres] recipe PASS: current=24x20/480 tokens, "
    "future=12x10/120 tokens, H32/t+32, replan=24, "
    "global_batch=768, steps=50000"
)
PY

if [[ "${VERIFY_ONLY:-0}" == "1" ]]; then
    echo "[RoboTwin fullres] VERIFY_ONLY=1; skipping distributed launcher"
    exit 0
fi

if [[ -f "${REPO_ROOT}/run_aidi_rbtw.sh" ]]; then
    TRAIN_LAUNCHER="${REPO_ROOT}/run_aidi_rbtw.sh"
elif [[ -f "${REPO_ROOT}/../run_aidi_rbtw.sh" ]]; then
    TRAIN_LAUNCHER="${REPO_ROOT}/../run_aidi_rbtw.sh"
else
    echo "[RoboTwin fullres][ERROR] cannot locate run_aidi_rbtw.sh" >&2
    exit 1
fi

export EXPECTED_NUM_MACHINES="${EXPECTED_NUM_MACHINES:-8}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
exec bash "${TRAIN_LAUNCHER}" "${CONFIG_PATH}"
