#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${STARVLA_DIR}"

CKPT="${CKPT:?Please set CKPT}"
BASE_PORT="${BASE_PORT:-6698}"
NUM_SERVERS="${NUM_SERVERS:-8}"
USE_BF16="${USE_BF16:-1}"
STRIP_DINO_KEYS="${STRIP_DINO_KEYS:-0}"
LOG_DIR="${LOG_DIR:-${STARVLA_DIR}/logs/libero_policy_servers_8}"

mkdir -p "${LOG_DIR}"

if [[ -z "${STARVLA_PYTHON:-}" ]]; then
  STARVLA_PYTHON="$(command -v python3 || command -v python)"
fi

echo "[8x SERVER] STARVLA_DIR=${STARVLA_DIR}"
echo "[8x SERVER] CKPT=${CKPT}"
echo "[8x SERVER] BASE_PORT=${BASE_PORT}"
echo "[8x SERVER] NUM_SERVERS=${NUM_SERVERS}"
echo "[8x SERVER] LOG_DIR=${LOG_DIR}"
echo "[8x SERVER] Host IP candidates:"
hostname -I || true

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

echo "[8x SERVER] CKPT_FOR_SERVER=${CKPT_FOR_SERVER}"

trap 'echo "[8x SERVER] killing all servers"; kill 0' INT TERM EXIT

for IDX in $(seq 0 $((NUM_SERVERS - 1))); do
  GPU_ID="${IDX}"
  PORT="$((BASE_PORT + IDX))"
  LOG_FILE="${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log"

  echo "[8x SERVER] launch gpu=${GPU_ID}, port=${PORT}, log=${LOG_FILE}"

  (
    GPU_ID="${GPU_ID}" \
    PORT="${PORT}" \
    CKPT="${CKPT_FOR_SERVER}" \
    STRIP_DINO_KEYS=0 \
    USE_BF16="${USE_BF16}" \
    bash examples/LIBERO/eval_files/run_policy_server.sh
  ) > "${LOG_FILE}" 2>&1 &

  sleep 2
done

echo "[8x SERVER] all launched. Keep this job alive."
wait
