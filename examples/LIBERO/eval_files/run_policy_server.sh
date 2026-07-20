#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"

if [[ -z "${STARVLA_PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    STARVLA_PYTHON="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    STARVLA_PYTHON="$(command -v python)"
  else
    echo "[ERROR] neither python3 nor python found"
    exit 1
  fi
fi

CKPT="${CKPT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_qwenwam/gr00t/0618_libero4in1_qwen3gr00t/final_model/pytorch_model.pt}"

# GPU_ID=1 PORT=6696 CKPT=xxx bash examples/LIBERO/eval_files/run_policy_server.sh
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"

STRIP_DINO_KEYS="${STRIP_DINO_KEYS:-1}"

cd "${STARVLA_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

echo "[POLICY SERVER] STARVLA_DIR=${STARVLA_DIR}"
echo "[POLICY SERVER] PYTHON=${STARVLA_PYTHON}"
echo "[POLICY SERVER] CKPT=${CKPT}"
echo "[POLICY SERVER] GPU_ID=${GPU_ID}"
echo "[POLICY SERVER] PORT=${PORT}"
echo "[POLICY SERVER] USE_BF16=${USE_BF16}"
echo "[POLICY SERVER] STRIP_DINO_KEYS=${STRIP_DINO_KEYS}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT not found: ${CKPT}"
  exit 1
fi

if [[ "${STRIP_DINO_KEYS}" == "1" ]]; then
  CKPT_FOR_SERVER="$(
    CKPT="${CKPT}" "${STARVLA_PYTHON}" - <<'PYFILTER'
import os
import sys
import hashlib
from pathlib import Path

import torch

ckpt = Path(os.environ["CKPT"])
stat = ckpt.stat()
tag_src = f"{ckpt}:{stat.st_size}:{int(stat.st_mtime)}"
tag = hashlib.md5(tag_src.encode()).hexdigest()[:12]

out_path = ckpt.parent / f"{ckpt.stem}_strip_dino_{tag}.pt"

if out_path.exists() and out_path.stat().st_size > 0:
    print(str(out_path))
    sys.exit(0)

print(f"[CKPT FILTER] loading {ckpt}", file=sys.stderr)
obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)

if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
    sd = obj["state_dict"]
    nested = True
else:
    sd = obj
    nested = False

if not isinstance(sd, dict):
    print(f"[CKPT FILTER] not state_dict-like, keep original: {ckpt}", file=sys.stderr)
    print(str(ckpt))
    sys.exit(0)

drop_keys = [
    k for k in list(sd.keys())
    if str(k).startswith("dino.") or str(k).startswith("module.dino.")
]

if not drop_keys:
    print(f"[CKPT FILTER] no dino.* keys, keep original: {ckpt}", file=sys.stderr)
    print(str(ckpt))
    sys.exit(0)

print(f"[CKPT FILTER] dropping {len(drop_keys)} dino keys", file=sys.stderr)

filtered = {k: v for k, v in sd.items() if k not in drop_keys}

if nested:
    obj["state_dict"] = filtered
    torch.save(obj, str(out_path))
else:
    torch.save(filtered, str(out_path))

print(f"[CKPT FILTER] saved filtered ckpt: {out_path}", file=sys.stderr)
print(str(out_path))
PYFILTER
  )"
else
  CKPT_FOR_SERVER="${CKPT}"
fi

echo "[POLICY SERVER] CKPT_FOR_SERVER=${CKPT_FOR_SERVER}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT_FOR_SERVER}"
  --port "${PORT}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

echo "[POLICY SERVER] Launching..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
