#!/usr/bin/env bash
# AIDI SERVER：生成统计，起 policy server，发布 servers.json，等待 client 完成。
# 只改这里的 CKPT；client 脚本里必须填同一个 CKPT。
set -euo pipefail

cd "$(dirname "$0")/../../.."
test -f pyproject.toml || { echo "ERROR: not in StarVLA repo root: $PWD"; exit 1; }
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export NO_ALBUMENTATIONS_UPDATE=1

# 中文注释：server 镜像里 NumPy 2.x 会和 pandas/pyarrow 的 1.x ABI 冲突，先固定版本。
python -m pip install --user --force-reinstall --no-deps numpy==1.26.4 pandas==2.2.3 pyarrow==14.0.1

# ===== USER CONFIG =====
CKPT="${CKPT:-}"
NUM_SERVERS="${NUM_SERVERS:-8}"
BASE_PORT="${BASE_PORT:-6500}"
# =======================
if [[ -z "${CKPT}" ]]; then echo "ERROR: set CKPT."; exit 1; fi

RUN_DIR="$(cd "$(dirname "${CKPT}")/.." && pwd)"
EVAL_ID="${JOINTFLOW_EVAL_ID:-$(basename "${RUN_DIR}")__$(basename "${CKPT}" .pt)}"
RDV_DIR="${RUN_DIR}/eval/${EVAL_ID}"

mkdir -p "${RDV_DIR}"
rm -f "${RDV_DIR}/servers.json" "${RDV_DIR}/CLIENTS_DONE"  # 中文注释：避免读到上次残留。

echo "[server] prepare eval assets -> ${RUN_DIR}"
python -m starVLA.jointflow.eval.prepare_eval_assets --ckpt_path "${CKPT}"

pids=()
for ((i = 0; i < NUM_SERVERS; i++)); do
  port="$((BASE_PORT + i))"
  # 中文注释：本 recipe 是 fp32 原生(backbone fp32_forward + 两个 flow head 内部强制 fp32)。
  # 加 --use_bf16 会把权重转 bf16,与 forward 里强制 fp32 的输入相乘 → "mat1 and mat2 must
  # have the same dtype, Float vs BFloat16"。eval 用 fp32(与训练数值口径一致),不要 --use_bf16。
  CUDA_VISIBLE_DEVICES="${i}" python -m starVLA.jointflow.eval.server_jointflow \
    --ckpt_path "${CKPT}" --port "${port}" --idle_timeout "${SERVER_IDLE_TIMEOUT:-1800}" \
    >"${RDV_DIR}/server_${i}.log" 2>&1 &
  pids+=("$!")
  echo "[server] GPU ${i} -> port ${port}, log=${RDV_DIR}/server_${i}.log"
done
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT

python -m starVLA.jointflow.eval.rendezvous publish \
  --rdv_dir "${RDV_DIR}" --base_port "${BASE_PORT}" --num_servers "${NUM_SERVERS}"

echo "[server] published. waiting CLIENTS_DONE in ${RDV_DIR}"
while [[ ! -f "${RDV_DIR}/CLIENTS_DONE" ]]; do sleep 30; done
echo "[server] done."
