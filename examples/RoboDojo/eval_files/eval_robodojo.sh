#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 10 ]]; then
  echo "Usage: bash eval_robodojo.sh <bench> <task> <ckpt_name> <env_cfg> <action_type> <seed> <policy_gpu> <env_gpu> <policy_env> <eval_env>" >&2
  exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboDojo}"
XPL_ROOT="${ROBODOJO_ROOT}/XPolicyLab"
UTILS_DIR="${XPL_ROOT}/utils"
OVERLAY_ROOT="${SCRIPT_DIR}/xpolicy_overlay"
DEPLOY_YAML="${SCRIPT_DIR}/deploy.yml"
CHECKPOINT_PATH="${STARVLA_CKPT_PATH:?export STARVLA_CKPT_PATH=/absolute/path/to/checkpoint.pt}"
VERIFY_CHECKPOINT="${SCRIPT_DIR}/verify_robodojo_checkpoint_contract.py"

if [[ "${bench_name}" != "RoboDojo" ]]; then
  echo "[RoboDojo][ERROR] this adapter only supports bench_name=RoboDojo, got ${bench_name}" >&2
  exit 2
fi
if [[ "${action_type}" != "joint" ]]; then
  echo "[RoboDojo][ERROR] this checkpoint ABI requires action_type=joint, got ${action_type}" >&2
  exit 2
fi

for required in \
  "${CHECKPOINT_PATH}" \
  "${STARVLA_ROOT}/deployment/model_server/checkpoint_contract.py" \
  "${VERIFY_CHECKPOINT}" \
  "${XPL_ROOT}/setup_policy_server.py" \
  "${UTILS_DIR}/get_free_port.sh" \
  "${UTILS_DIR}/wait_for_policy_server.sh" \
  "${ROBODOJO_ROOT}/src/eval_client/main.py" \
  "${ROBODOJO_ROOT}/env_cfg/${env_cfg_type}.yml" \
  "${ROBODOJO_ROOT}/task/RoboDojo/config/${task_name}.yml" \
  "${ROBODOJO_ROOT}/task/RoboDojo/tasks/${task_name}.py" \
  "${OVERLAY_ROOT}/XPolicyLab/policy/starVLA/model.py" \
  "${OVERLAY_ROOT}/XPolicyLab/policy/starVLA/deploy.py" \
  "${DEPLOY_YAML}"; do
  [[ -f "${required}" ]] || { echo "[RoboDojo][ERROR] missing ${required}" >&2; exit 1; }
done

# RoboDojo resolves assets relative to its checkout.  Validate the official
# directory names before importing Isaac Sim so an incomplete mount fails in
# seconds instead of after a long simulator startup.
for required_dir in \
  "${ROBODOJO_ROOT}/Assets/Robots" \
  "${ROBODOJO_ROOT}/Assets/Object" \
  "${ROBODOJO_ROOT}/Assets/Material" \
  "${ROBODOJO_ROOT}/Assets/Eval_Layout"; do
  [[ -d "${required_dir}" ]] || { echo "[RoboDojo][ERROR] missing asset directory ${required_dir}" >&2; exit 1; }
done

starvla_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
policy_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
while [[ "${policy_port}" == "${starvla_port}" ]]; do
  policy_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
done

STARVLA_SERVER_PID=""
XPOLICY_SERVER_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [[ -z "${XPOLICY_SERVER_PID}" ]] || kill "${XPOLICY_SERVER_PID}" 2>/dev/null || true
  [[ -z "${STARVLA_SERVER_PID}" ]] || kill "${STARVLA_SERVER_PID}" 2>/dev/null || true
  [[ -z "${XPOLICY_SERVER_PID}" ]] || wait "${XPOLICY_SERVER_PID}" 2>/dev/null || true
  [[ -z "${STARVLA_SERVER_PID}" ]] || wait "${STARVLA_SERVER_PID}" 2>/dev/null || true
  exit "${rc}"
}
trap cleanup EXIT INT TERM

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"
export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
# The overlay must stay before RoboDojo's own XPolicyLab package in both the
# policy-server and simulator processes.  This makes model.py *and* deploy.py
# come from the uploaded StarVLA tree while all transport/simulator modules are
# resolved from the external RoboDojo checkout.
export PYTHONPATH="${OVERLAY_ROOT}:${STARVLA_ROOT}:${ROBODOJO_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"

echo "[RoboDojo] active StarVLA=${STARVLA_ROOT}"
echo "[RoboDojo] simulator=${ROBODOJO_ROOT} checkpoint=${CHECKPOINT_PATH}"
echo "[RoboDojo] task=${task_name} StarVLA-port=${starvla_port} XPolicy-port=${policy_port}"

python "${VERIFY_CHECKPOINT}" --checkpoint "${CHECKPOINT_PATH}"

(
  cd "${STARVLA_ROOT}"
  export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
  exec python deployment/model_server/server_policy.py \
    --ckpt_path "${CHECKPOINT_PATH}" \
    --port "${starvla_port}" \
    --idle_timeout 3600 \
    --use_bf16
) &
STARVLA_SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" \
  127.0.0.1 "${starvla_port}" "${STARVLA_SERVER_PID}" "StarVLA server" 1800

CUDA_VISIBLE_DEVICES="${policy_gpu_id}" python "${XPL_ROOT}/setup_policy_server.py" \
  --config_path "${DEPLOY_YAML}" \
  --overrides \
    port="${policy_port}" \
    host=127.0.0.1 \
    bench_name="${bench_name}" \
    task_name="${task_name}" \
    ckpt_name="${ckpt_name}" \
    env_cfg_type="${env_cfg_type}" \
    action_type="${action_type}" \
    seed="${seed}" \
    starvla_server_host=127.0.0.1 \
    starvla_server_port="${starvla_port}" \
    include_state=true \
    expected_action_chunk_size=16 \
    expected_action_dim=14 \
    expected_state_dim=14 &
XPOLICY_SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" \
  127.0.0.1 "${policy_port}" "${XPOLICY_SERVER_PID}" "RoboDojo XPolicy server" 600

additional_info="ckpt_name=${ckpt_name},action_type=${action_type},starvla_contract=state_zscore_fastwam_h16"

# Do not call RoboDojo's generic eval_policy.sh here: that script prepends its
# own XPolicyLab directory and can silently import a stale/missing StarVLA
# deploy.py.  Launch the same official simulator entry directly with our
# overlay first, retaining RoboDojo's bounded PhysX restart behavior.
conda activate "${eval_env_conda_env}"
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)-$$}"
max_retries="${ROBODOJO_MAX_BASH_RETRIES:-10}"
attempt=0
kit_args="--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
while :; do
  set +e
  (
    cd "${ROBODOJO_ROOT}"
    export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
    exec python -u src/eval_client/main.py \
      --task_name "${task_name}" \
      --env_cfg_type "${env_cfg_type}" \
      --num_envs 1 \
      --enable_cameras \
      --kit_args "${kit_args}" \
      --device_id "${env_gpu_id}" \
      --policy_name starVLA \
      --host 127.0.0.1 \
      --port "${policy_port}" \
      --protocol ws \
      --policy_server_url "ws://127.0.0.1:${policy_port}" \
      --additional_info "${additional_info}" \
      --seed "${seed}" \
      --headless
  )
  rc=$?
  set -e
  case "${rc}" in
    0)
      break
      ;;
    99|134|139)
      attempt=$((attempt + 1))
      if (( attempt >= max_retries )); then
        echo "[RoboDojo][ERROR] simulator failed rc=${rc} after ${attempt} attempts" >&2
        exit "${rc}"
      fi
      echo "[RoboDojo] simulator restart ${attempt}/${max_retries} after rc=${rc}" >&2
      sleep 5
      ;;
    *)
      exit "${rc}"
      ;;
  esac
done

echo "[RoboDojo] evaluation finished"
