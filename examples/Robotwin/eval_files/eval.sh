#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: bash examples/Robotwin/eval_files/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <policy_ckpt_path> [policy_port] [policy_host]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOTWIN_PATH="${ROBOTWIN_PATH:-/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboTwin}"
if [[ ! -d "${ROBOTWIN_PATH}" ]]; then
    echo "ROBOTWIN_PATH does not exist: ${ROBOTWIN_PATH}" >&2
    exit 1
fi

robotwin_eval_script="${ROBOTWIN_PATH}/script/eval_policy.py"
if [[ ! -f "${robotwin_eval_script}" ]]; then
    echo "RoboTwin eval entry does not exist: ${robotwin_eval_script}" >&2
    exit 1
fi

patch_check_tool=""
if command -v rg >/dev/null 2>&1; then
    patch_check_tool="rg"
    patch_check_cmd=(rg -q "policy_ckpt_path" "${robotwin_eval_script}")
elif command -v grep >/dev/null 2>&1; then
    patch_check_tool="grep"
    patch_check_cmd=(grep -q "policy_ckpt_path" "${robotwin_eval_script}")
else
    echo "Neither rg nor grep is available, so the RoboTwin patch check cannot run." >&2
    exit 1
fi

if ! "${patch_check_cmd[@]}"; then
    echo "Your third-party RoboTwin checkout is missing the required policy_ckpt_path patch: ${robotwin_eval_script}" >&2
    echo "Patch check used: ${patch_check_tool}" >&2
    echo "Apply the documented patch in your own RoboTwin repo; see examples/Robotwin/README.md." >&2
    exit 1
fi

policy_name="${ROBOTWIN_POLICY_NAME:-model2robotwin_interface}"
task_name="$1"
task_config="$2"
ckpt_setting="${3:-starvla_demo}"
seed="${4:-0}"
gpu_id="${5:-0}"
policy_ckpt_path="$6"
policy_port="${7:-${ROBOTWIN_POLICY_PORT:-5694}}"
policy_host="${8:-${ROBOTWIN_POLICY_HOST:-127.0.0.1}}"
robotwin_python="${ROBOTWIN_PYTHON:-python}"
deploy_policy_template="${DEPLOY_POLICY_TEMPLATE_PATH:-${SCRIPT_DIR}/deploy_policy.yml}"
replan_steps="${ROBOTWIN_REPLAN_STEPS:-${REPLAN_STEPS:-}}"
wam_expected_phase="${WAM_EXPECTED_PHASE:-}"
wam_expected_world_to_action="${WAM_EXPECTED_WORLD_TO_ACTION:-}"
coflow_inference_mode="${COFLOW_INFERENCE_MODE:-}"
coflow_inference_horizon="${COFLOW_INFERENCE_HORIZON:-}"
coflow_inference_seed="${COFLOW_INFERENCE_SEED:-}"

if [[ -n "${replan_steps}" && ! "${replan_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPLAN_STEPS must be a positive integer, got: ${replan_steps}" >&2
    exit 1
fi
if [[ -n "${wam_expected_phase}" && \
      "${wam_expected_phase}" != "predictor_warmup" && \
      "${wam_expected_phase}" != "gate_ft" ]]; then
    echo "WAM_EXPECTED_PHASE must be predictor_warmup or gate_ft, got: ${wam_expected_phase}" >&2
    exit 1
fi
if [[ -n "${wam_expected_world_to_action}" && \
      "${wam_expected_world_to_action}" != "enabled" && \
      "${wam_expected_world_to_action}" != "disabled" ]]; then
    echo "WAM_EXPECTED_WORLD_TO_ACTION must be enabled or disabled, got: ${wam_expected_world_to_action}" >&2
    exit 1
fi
if [[ -n "${coflow_inference_mode}" && "${coflow_inference_mode}" != "policy" && "${coflow_inference_mode}" != "diagonal" ]]; then
    echo "COFLOW_INFERENCE_MODE must be policy or diagonal, got: ${coflow_inference_mode}" >&2
    exit 1
fi
if [[ -n "${coflow_inference_horizon}" && ! "${coflow_inference_horizon}" =~ ^[1-9][0-9]*$ ]]; then
    echo "COFLOW_INFERENCE_HORIZON must be a positive integer, got: ${coflow_inference_horizon}" >&2
    exit 1
fi
if [[ -n "${coflow_inference_seed}" && ! "${coflow_inference_seed}" =~ ^[0-9]+$ ]]; then
    echo "COFLOW_INFERENCE_SEED must be a non-negative integer, got: ${coflow_inference_seed}" >&2
    exit 1
fi
if [[ -n "${coflow_inference_mode}" && -z "${coflow_inference_horizon}" ]] || \
   [[ -z "${coflow_inference_mode}" && -n "${coflow_inference_horizon}" ]]; then
    echo "COFLOW_INFERENCE_MODE and COFLOW_INFERENCE_HORIZON must be set together" >&2
    exit 1
fi
if [[ -n "${coflow_inference_seed}" && -z "${coflow_inference_mode}" ]]; then
    echo "COFLOW_INFERENCE_SEED requires COFLOW_INFERENCE_MODE/HORIZON" >&2
    exit 1
fi

if [[ ! -f "${deploy_policy_template}" ]]; then
    echo "Deploy policy template does not exist: ${deploy_policy_template}" >&2
    exit 1
fi

runtime_deploy_policy="$(mktemp "${TMPDIR:-/tmp}/robotwin_deploy_policy.XXXXXX.yml")"
cleanup() {
    rm -f "${runtime_deploy_policy}"
}
trap cleanup EXIT

runtime_replan_steps="${replan_steps:-null}"
runtime_wam_expected_phase="${wam_expected_phase:-null}"
case "${wam_expected_world_to_action}" in
    enabled) runtime_wam_expected_world_to_action=true ;;
    disabled) runtime_wam_expected_world_to_action=false ;;
    *) runtime_wam_expected_world_to_action=null ;;
esac
runtime_coflow_mode="${coflow_inference_mode:-null}"
runtime_coflow_horizon="${coflow_inference_horizon:-null}"
runtime_coflow_seed="${coflow_inference_seed:-null}"
sed \
    -e "s/^host:.*/host: \"${policy_host}\"/" \
    -e "s/^port:.*/port: ${policy_port}/" \
    -e "s/^replan_steps:.*/replan_steps: ${runtime_replan_steps}/" \
    -e "s/^wam_expected_phase:.*/wam_expected_phase: ${runtime_wam_expected_phase}/" \
    -e "s/^wam_expected_world_to_action:.*/wam_expected_world_to_action: ${runtime_wam_expected_world_to_action}/" \
    -e "s/^coflow_inference_mode:.*/coflow_inference_mode: ${runtime_coflow_mode}/" \
    -e "s/^coflow_inference_horizon:.*/coflow_inference_horizon: ${runtime_coflow_horizon}/" \
    -e "s/^coflow_inference_seed:.*/coflow_inference_seed: ${runtime_coflow_seed}/" \
    "${deploy_policy_template}" > "${runtime_deploy_policy}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH="${SCRIPT_DIR}"
STARVLA_PATH="${REPO_ROOT}"

export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}"
export PYTHONPATH="${STARVLA_PATH}:${PYTHONPATH}"
export PYTHONPATH="${EVAL_FILES_PATH}:${PYTHONPATH}"

cd "${ROBOTWIN_PATH}"

echo "PYTHONPATH: ${PYTHONPATH}"
echo "task_name: ${task_name}"
echo "task_config: ${task_config}"
echo "ckpt_setting: ${ckpt_setting}"
echo "policy_port: ${policy_port}"
echo "replan_steps: ${replan_steps:-full model chunk}"
echo "wam_expected_phase: ${wam_expected_phase:-unchecked}"
echo "wam_expected_world_to_action: ${wam_expected_world_to_action:-unchecked}"
echo "coflow_inference: ${coflow_inference_mode:-checkpoint default}/${coflow_inference_horizon:-checkpoint chunk}"
echo "coflow_seed: ${coflow_inference_seed:-stochastic}"

PYTHONWARNINGS=ignore::UserWarning \
"${robotwin_python}" script/eval_policy.py --config "${runtime_deploy_policy}" \
    --policy_ckpt_path "${policy_ckpt_path}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}"
