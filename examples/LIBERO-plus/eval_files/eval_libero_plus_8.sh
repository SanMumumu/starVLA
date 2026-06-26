#!/usr/bin/env bash
set -euo pipefail

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

STARVLA_DIR="${STARVLA_DIR:-/running_package/starvla_dev/starVLA}"
LIBERO_HOME="${LIBERO_HOME:-/opt/LIBERO-plus}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/opt/LIBERO-plus/.libero_plus_config}"

HOST="${HOST:?HOST is required}"
BASE_PORT="${BASE_PORT:-6698}"
NUM_CLIENTS="${NUM_CLIENTS:-8}"
CKPT="${CKPT:?CKPT is required}"
SEED="${SEED:-7}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
EXPECTED_TOTAL="${EXPECTED_TOTAL:-10030}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$(dirname "${CKPT}")")/libero_plus_eval_results_$(basename "$(dirname "${CKPT}")")_$(basename "${CKPT}" .pt)_${NUM_CLIENTS}shard}"

cd "$STARVLA_DIR"
export LIBERO_HOME LIBERO_CONFIG_PATH
export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/shards"

echo "========== LIBERO-Plus preflight =========="
python - <<'PY'
import os
import pathlib
import libero
import libero.libero as libero_core
from libero.libero import get_libero_path

print("libero module       :", libero_core.__file__)
print("LIBERO_HOME         :", os.environ.get("LIBERO_HOME"))
print("LIBERO_CONFIG_PATH  :", os.environ.get("LIBERO_CONFIG_PATH"))
for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
    path = pathlib.Path(get_libero_path(key))
    print(f"{key:20s}: {path}")
    assert path.exists(), (key, path)
classification = pathlib.Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json"
assert classification.is_file(), classification
assert "LIBERO-plus" in str(pathlib.Path(libero_core.__file__).resolve()), libero_core.__file__
print("task classification :", classification)
print("LIBERO-Plus preflight OK")
PY

for ((i=0; i<NUM_CLIENTS; i++)); do
    port=$((BASE_PORT + i))
    python - "$HOST" "$port" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
with socket.create_connection((host, port), timeout=10):
    print(f"server reachable: {host}:{port}")
PY
done

pids=()
for ((i=0; i<NUM_CLIENTS; i++)); do
    port=$((BASE_PORT + i))
    log="$OUTPUT_DIR/logs/shard_${i}.log"
    extra=()
    [[ "$SAVE_VIDEO" == "1" ]] && extra+=(--save-video)

    echo "launch client shard=$i gpu=$i server=$HOST:$port"
    (
        export CUDA_VISIBLE_DEVICES="$i"
        export MUJOCO_EGL_DEVICE_ID=0
        echo "[CLIENT EGL ENV] shard=$i CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID-UNSET}"
        python examples/LIBERO-plus/eval_files/eval_libero_plus_sharded.py \
            --host "$HOST" \
            --port "$port" \
            --ckpt "$CKPT" \
            --output-dir "$OUTPUT_DIR" \
            --shard-id "$i" \
            --num-shards "$NUM_CLIENTS" \
            --seed "$SEED" \
            --expected-total "$EXPECTED_TOTAL" \
            "${extra[@]}"
    ) >"$log" 2>&1 &
    pids+=("$!")
done

failed=0
for ((i=0; i<NUM_CLIENTS; i++)); do
    if wait "${pids[$i]}"; then
        echo "client shard $i finished"
    else
        echo "ERROR: client shard $i failed; inspect $OUTPUT_DIR/logs/shard_${i}.log"
        failed=1
    fi
done
[[ "$failed" == "0" ]] || exit 1

python examples/LIBERO-plus/eval_files/aggregate_libero_plus_8.py \
    --output-dir "$OUTPUT_DIR" \
    --num-shards "$NUM_CLIENTS" \
    --expected-total "$EXPECTED_TOTAL"
