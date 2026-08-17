#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:?usage: launch.sh base_h25|base_h25_current_dino_fullres|base_history_h25_mem|base_text_h25_mem|base_text_h25_mem_bf16|base_text_h25_mem_bf16_current_dino_fullres}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

case "${MODE}" in
  base_h25) CONFIG_PATH="${SCRIPT_DIR}/rynn_base_h25_50k.yaml" ;;
  base_h25_current_dino_fullres) CONFIG_PATH="${SCRIPT_DIR}/rynn_base_h25_current_dino_fullres_50k.yaml" ;;
  base_history_h25_mem) CONFIG_PATH="${SCRIPT_DIR}/rynn_base_history_h25_mem_50k.yaml" ;;
  base_text_h25_mem) CONFIG_PATH="${SCRIPT_DIR}/rynn_base_text_h25_mem_50k.yaml" ;;
  base_text_h25_mem_bf16) CONFIG_PATH="${SCRIPT_DIR}/rynn_base_text_h25_mem_bf16_50k.yaml" ;;
  base_text_h25_mem_bf16_current_dino_fullres) CONFIG_PATH="${SCRIPT_DIR}/rynn_base_text_h25_mem_bf16_current_dino_fullres_50k.yaml" ;;
  *) echo "[released-rynn50k][ERROR] unsupported mode: ${MODE}" >&2; exit 2 ;;
esac

VERIFY_ARGS=(
  --config "${CONFIG_PATH}"
)
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python3 "${SCRIPT_DIR}/verify_recipe.py" "${VERIFY_ARGS[@]}"

if [[ "${STARVLA_RESUME:-}" == "1" || "${STARVLA_RESUME:-}" == "true" ]]; then
  # train_starvla.py's normalize_dotlist_args only keeps args that start with
  # "--"; bare OmegaConf "key=value" strings are silently dropped.
  STARVLA_TRAIN_EXTRA_ARGS="${STARVLA_TRAIN_EXTRA_ARGS:-} --trainer.is_resume=true"
  STARVLA_TRAIN_EXTRA_ARGS="${STARVLA_TRAIN_EXTRA_ARGS# }"
  export STARVLA_TRAIN_EXTRA_ARGS
  echo "[released-rynn50k] resume enabled: STARVLA_TRAIN_EXTRA_ARGS=${STARVLA_TRAIN_EXTRA_ARGS}"
fi

if [[ -f "${REPO_ROOT}/../run_aidi_rbtw.sh" ]]; then
  TRAIN_LAUNCHER="${REPO_ROOT}/../run_aidi_rbtw.sh"
elif [[ -f "${REPO_ROOT}/run_aidi_rbtw.sh" ]]; then
  TRAIN_LAUNCHER="${REPO_ROOT}/run_aidi_rbtw.sh"
else
  echo "[released-rynn50k][ERROR] cannot locate run_aidi_rbtw.sh" >&2
  exit 1
fi

export EXPECTED_NUM_MACHINES=8
export GPUS_PER_NODE=8
exec bash "${TRAIN_LAUNCHER}" "${CONFIG_PATH}"
