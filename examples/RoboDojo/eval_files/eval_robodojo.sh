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
# Kept as positional ABI compatibility with upstream XPolicyLab. The split
# client resolves one RoboDojo/Isaac Python executable directly and does not
# require a shell-level conda command.
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

EVAL_MODE="${ROBODOJO_EVAL_MODE:-fast}"
EVAL_MODE="${EVAL_MODE,,}"
case "${EVAL_MODE}" in
  fast)
    export ROBODOJO_DISABLE_EVAL_VIDEO=1
    requested_trials="${ROBODOJO_TRIALS:-${EVAL_NUM:-native}}"
    ;;
  visualize)
    export ROBODOJO_DISABLE_EVAL_VIDEO=0
    requested_trials="${ROBODOJO_TRIALS:-${EVAL_NUM:-5}}"
    ;;
  *)
    echo "[RoboDojo][ERROR] ROBODOJO_EVAL_MODE must be fast or visualize, got ${EVAL_MODE}" >&2
    exit 2
    ;;
esac
if [[ "${requested_trials}" != "native" ]] && { ! [[ "${requested_trials}" =~ ^[0-9]+$ ]] || (( 10#${requested_trials} <= 0 )); }; then
  echo "[RoboDojo][ERROR] ROBODOJO_TRIALS must be native or a positive integer, got ${requested_trials}" >&2
  exit 2
fi
if [[ "${requested_trials}" == "native" ]]; then
  unset EVAL_NUM
else
  export EVAL_NUM="${requested_trials}"
fi

# Keep evaluation artifacts next to the trained experiment, matching the
# RoboTwin launcher layout.  RoboDojo itself writes to a relative
# ``eval_result`` directory, so run the simulator from a per-run native
# workdir instead of mutating/symlinking the shared RoboDojo checkout.
if [[ "${CHECKPOINT_PATH}" == *"/checkpoints/"* ]]; then
  MODEL_ROOT="$(dirname "$(dirname "${CHECKPOINT_PATH}")")"
else
  MODEL_ROOT="$(dirname "${CHECKPOINT_PATH}")"
fi
MODEL_ROOT="$(cd "${MODEL_ROOT}" && pwd -P)"
CKPT_STEM="$(basename "${CHECKPOINT_PATH}")"
CKPT_STEM="${CKPT_STEM%.*}"
export ROBODOJO_RUN_ID="${ROBODOJO_RUN_ID:-$(date +%Y%m%d_%H%M%S)-$$}"
trial_tag="$([[ "${requested_trials}" == "native" ]] && printf native || printf 'trials%s' "${requested_trials}")"
RUN_NAME="${ROBODOJO_RUN_NAME:-${EVAL_MODE}_${task_name}_${trial_tag}}"
requested_output_root="${ROBODOJO_OUTPUT_ROOT:-${MODEL_ROOT}/robodojo_eval_results/${RUN_NAME}_${CKPT_STEM}_${ROBODOJO_RUN_ID}}"
case "${requested_output_root}" in
  "${MODEL_ROOT}"/*) ;;
  *)
    echo "[RoboDojo][ERROR] output must stay under checkpoint experiment directory ${MODEL_ROOT}, got ${requested_output_root}" >&2
    exit 2
    ;;
esac
mkdir -p "${requested_output_root}"
OUTPUT_ROOT="$(cd "${requested_output_root}" && pwd -P)"
case "${OUTPUT_ROOT}" in
  "${MODEL_ROOT}"/*) ;;
  *)
    echo "[RoboDojo][ERROR] output must stay under checkpoint experiment directory ${MODEL_ROOT}, got ${OUTPUT_ROOT}" >&2
    exit 2
    ;;
esac
LOG_DIR="${OUTPUT_ROOT}/logs"
NATIVE_WORKDIR="${OUTPUT_ROOT}/native"
NATIVE_RESULT_ROOT="${NATIVE_WORKDIR}/eval_result"
CLIENT_LOG="${LOG_DIR}/client.log"
mkdir -p "${LOG_DIR}" "${NATIVE_WORKDIR}"

# Preserve live AIDI output while keeping a persistent log. Suppress only the
# repeated square-pixel aperture adjustment; all other Isaac warnings/errors
# remain visible. Set ROBODOJO_HIDE_CAMERA_APERTURE_WARNING=0 to debug it.
filter_robodojo_output() {
  local line
  while IFS= read -r line || [[ -n "${line}" ]]; do
    if [[ "${ROBODOJO_HIDE_CAMERA_APERTURE_WARNING:-1}" == "1" \
      && "${line}" == *"[Warning] [isaacsim.sensors.camera.camera]"* \
      && "${line}" == *"are inconsistent with the pixel resolution aspect ratio"* \
      && "${line}" == *"Setting 'verticalAperture'"* ]]; then
      continue
    fi
    printf '%s\n' "${line}"
  done
}
exec > >(filter_robodojo_output | tee -a "${CLIENT_LOG}") 2>&1

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
  "${XPL_ROOT}/setup_policy_server.py" \
  "${UTILS_DIR}/get_free_port.sh" \
  "${UTILS_DIR}/wait_for_policy_server.sh" \
  "${ROBODOJO_ROOT}/src/eval_client/main.py" \
  "${ROBODOJO_ROOT}/env_cfg/${env_cfg_type}.yml" \
  "${ROBODOJO_ROOT}/task/RoboDojo/config/${task_name}.yml" \
  "${ROBODOJO_ROOT}/task/RoboDojo/tasks/${task_name}.py" \
  "${SCRIPT_DIR}/launch_xpolicy_server.py" \
  "${SCRIPT_DIR}/launch_robodojo_client.py" \
  "${SCRIPT_DIR}/summarize_robodojo_table1.py" \
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

# This launcher is client-only. Requiring an explicit remote address prevents
# an accidental second model copy from being loaded in the Isaac client job.
starvla_host="${STARVLA_SERVER_HOST:?export STARVLA_SERVER_HOST=<H20 server IP>}"
starvla_port="${STARVLA_SERVER_PORT:-7777}"
if [[ -n "${ROBODOJO_XPOLICY_PORT:-}" ]]; then
  [[ "${ROBODOJO_XPOLICY_PORT}" =~ ^[0-9]+$ ]] \
    && (( ROBODOJO_XPOLICY_PORT >= 1 && ROBODOJO_XPOLICY_PORT <= 65535 )) || {
      echo "[RoboDojo][ERROR] ROBODOJO_XPOLICY_PORT must be an integer in [1, 65535]" >&2
      exit 2
    }
  policy_port="${ROBODOJO_XPOLICY_PORT}"
else
  policy_port="$(bash "${UTILS_DIR}/get_free_port.sh")"
fi

XPOLICY_SERVER_PID=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [[ -z "${XPOLICY_SERVER_PID}" ]] || kill "${XPOLICY_SERVER_PID}" 2>/dev/null || true
  [[ -z "${XPOLICY_SERVER_PID}" ]] || wait "${XPOLICY_SERVER_PID}" 2>/dev/null || true
  exit "${rc}"
}
trap cleanup EXIT INT TERM

export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export NO_ALBUMENTATIONS_UPDATE=1
# The overlay must stay before RoboDojo's own XPolicyLab package in both the
# policy-server and simulator processes.  This makes model.py *and* deploy.py
# come from the uploaded StarVLA tree while all transport/simulator modules are
# resolved from the external RoboDojo checkout.
export PYTHONPATH="${OVERLAY_ROOT}:${STARVLA_ROOT}:${ROBODOJO_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"

resolve_robodojo_python() {
  local candidate
  local -a candidates=()
  [[ -z "${ROBODOJO_PYTHON:-}" ]] || candidates+=("${ROBODOJO_PYTHON}")
  [[ -z "${CONDA_PREFIX:-}" ]] || candidates+=("${CONDA_PREFIX}/bin/python")
  candidates+=(
    "/opt/robodojo-env/bin/python"
    "/root/miniconda3/envs/${eval_env_conda_env}/bin/python"
    "/opt/conda/envs/${eval_env_conda_env}/bin/python"
    "/usr/local/miniconda3/envs/${eval_env_conda_env}/bin/python"
    "${HOME}/miniconda3/envs/${eval_env_conda_env}/bin/python"
    "/root/miniconda3/envs/RoboDojo/bin/python"
    "/opt/conda/envs/RoboDojo/bin/python"
  )
  command -v python >/dev/null 2>&1 && candidates+=("$(command -v python)")
  command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")

  for candidate in "${candidates[@]}"; do
    [[ -x "${candidate}" ]] || continue
    if "${candidate}" - <<'PY' >/dev/null 2>&1
import importlib.util

required = ("isaacsim", "torch", "torchvision", "cv2", "websockets", "yaml", "msgpack", "msgpack_numpy")
missing = [name for name in required if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
    then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! ROBODOJO_PYTHON="$(resolve_robodojo_python)"; then
  echo "[RoboDojo][ERROR] cannot find a Python with Isaac Sim dependencies." >&2
  echo "[RoboDojo][ERROR] set ROBODOJO_PYTHON=/absolute/path/to/RoboDojo/bin/python." >&2
  echo "[RoboDojo][ERROR] current PATH=${PATH}" >&2
  exit 1
fi
export ROBODOJO_PYTHON

echo "[RoboDojo] active StarVLA=${STARVLA_ROOT}"
echo "[RoboDojo] simulator=${ROBODOJO_ROOT} checkpoint=${CHECKPOINT_PATH}"
echo "[RoboDojo] python=${ROBODOJO_PYTHON} (policy_env_arg=${policy_conda_env}, eval_env_arg=${eval_env_conda_env})"
echo "[RoboDojo] task=${task_name} StarVLA=${starvla_host}:${starvla_port} XPolicy-port=${policy_port}"
echo "[RoboDojo] using external StarVLA server ${starvla_host}:${starvla_port}"
echo "[RoboDojo] mode=${EVAL_MODE} trials=${requested_trials} videos=$([[ "${ROBODOJO_DISABLE_EVAL_VIDEO}" == "1" ]] && echo disabled || echo enabled)"
echo "[RoboDojo] output=${OUTPUT_ROOT}"
echo "[RoboDojo] log=${CLIENT_LOG}"
echo "[RoboDojo] native-results=${NATIVE_RESULT_ROOT}"

# Use the repository-owned launcher instead of executing XPolicyLab's script
# directly. It makes the overlay import deterministic and supplies the narrow
# server compatibility required by the client image's websockets 12 package.
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" "${ROBODOJO_PYTHON}" "${SCRIPT_DIR}/launch_xpolicy_server.py" \
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
    starvla_server_host="${starvla_host}" \
    starvla_server_port="${starvla_port}" \
    include_state=true \
    expected_action_chunk_size=16 \
    expected_action_dim=14 \
    expected_state_dim=14 &
XPOLICY_SERVER_PID=$!
bash "${UTILS_DIR}/wait_for_policy_server.sh" \
  127.0.0.1 "${policy_port}" "${XPOLICY_SERVER_PID}" "RoboDojo XPolicy server" 600

additional_info="ckpt_name=${ckpt_name},action_type=${action_type},starvla_contract=state_zscore_fastwam_h16,eval_mode=${EVAL_MODE}"

# Do not call RoboDojo's generic eval_policy.sh here: that script prepends its
# own XPolicyLab directory and can silently import a stale/missing StarVLA
# deploy.py.  Launch the same official simulator entry directly with our
# overlay first, retaining RoboDojo's bounded PhysX restart behavior.
max_retries="${ROBODOJO_MAX_BASH_RETRIES:-10}"
attempt=0
kit_args="--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
while :; do
  set +e
  (
    # eval_env.py intentionally writes to ./eval_result.  All source/config
    # paths are resolved from ROBODOJO_ROOT/PYTHONPATH, so changing only cwd
    # redirects native results without touching the shared simulator tree.
    cd "${NATIVE_WORKDIR}"
    export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
    exec "${ROBODOJO_PYTHON}" -u "${SCRIPT_DIR}/launch_robodojo_client.py" \
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
echo "[RoboDojo] output=${OUTPUT_ROOT}"
if [[ "${ROBODOJO_SKIP_TABLE1_SUMMARY:-0}" != "1" ]]; then
  "${ROBODOJO_PYTHON}" "${SCRIPT_DIR}/summarize_robodojo_table1.py" \
    --eval-root "${MODEL_ROOT}/robodojo_eval_results" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --ckpt-name "${ckpt_name}" \
    --seed "${seed}"
fi
