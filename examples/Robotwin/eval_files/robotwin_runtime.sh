#!/usr/bin/env bash
# Shared runtime resolution for RoboTwin evaluation. This file is sourced.

ROBOTWIN_RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_STARVLA_ROOT="$(cd "${ROBOTWIN_RUNTIME_DIR}/../../.." && pwd)"

prepare_robotwin_runtime() {
    local robotwin_path="${1:?missing RoboTwin path}"
    local probe_gpu="${2:-0}"
    local explicit_python="${ROBOTWIN_PYTHON:-}"
    local extension_root="${ROBOTWIN_TORCH_EXTENSIONS_ROOT:-${TMPDIR:-/tmp}/starvla_robotwin_torch_extensions_${UID:-0}_$$}"
    local -a candidates=()

    export PYTHONNOUSERSITE=1
    export MAX_JOBS="${MAX_JOBS:-8}"

    if [[ -n "${explicit_python}" ]]; then
        if [[ "${explicit_python}" == */* ]]; then
            candidates+=("${explicit_python}")
        elif command -v "${explicit_python}" >/dev/null 2>&1; then
            candidates+=("$(command -v "${explicit_python}")")
        else
            echo "[ERROR] ROBOTWIN_PYTHON is not executable or on PATH: ${explicit_python}" >&2
            return 1
        fi
    else
        candidates+=(
            "/opt/conda/envs/robotwin/bin/python"
            "/opt/conda/envs/RoboTwin/bin/python"
            "/opt/robotwin-env/bin/python"
            "${HOME:-/root}/miniconda3/envs/robotwin/bin/python"
            "${HOME:-/root}/miniconda3/envs/RoboTwin/bin/python"
            "${HOME:-/root}/anaconda3/envs/robotwin/bin/python"
            "${HOME:-/root}/anaconda3/envs/RoboTwin/bin/python"
        )
        command -v python3 >/dev/null 2>&1 && candidates+=("$(command -v python3)")
        command -v python >/dev/null 2>&1 && candidates+=("$(command -v python)")
    fi

    local index=0 candidate extension_dir
    local -a attempted=()
    for candidate in "${candidates[@]}"; do
        [[ -x "${candidate}" ]] || continue
        # Invoke the environment's own Python path. Resolving a venv/conda
        # symlink to the base interpreter can discard its sys.prefix.
        if [[ " ${attempted[*]:-} " == *" ${candidate} "* ]]; then
            continue
        fi
        attempted+=("${candidate}")
        extension_dir="${extension_root}/candidate_${index}"
        mkdir -p "${extension_dir}"
        echo "[RoboTwin ABI] probing ${candidate} (serialized CuRobo warmup)"
        if CUDA_VISIBLE_DEVICES="${probe_gpu}" \
            TORCH_EXTENSIONS_DIR="${extension_dir}" \
            PYTHONPATH="${robotwin_path}:${PYTHONPATH:-}" \
            "${candidate}" "${ROBOTWIN_RUNTIME_DIR}/verify_robotwin_runtime.py" \
                --robotwin-path "${robotwin_path}"
        then
            if [[ "${ROBOTWIN_POLICY_NAME:-}" == "model2robotwin_fastwam_interface" ]]; then
                if ! CUDA_VISIBLE_DEVICES="${probe_gpu}" \
                    TORCH_EXTENSIONS_DIR="${extension_dir}" \
                    PYTHONPATH="${ROBOTWIN_RUNTIME_DIR}:${ROBOTWIN_STARVLA_ROOT}:${robotwin_path}:${PYTHONPATH:-}" \
                    "${candidate}" -c \
                        "import model2robotwin_fastwam_interface; print('ROBOTWIN_FASTWAM_ADAPTER_PASS')"
                then
                    echo "[RoboTwin ABI] FastWAM adapter rejected python=${candidate}" >&2
                    index=$((index + 1))
                    continue
                fi
            fi
            ROBOTWIN_PYTHON="${candidate}"
            TORCH_EXTENSIONS_DIR="${extension_dir}"
            export ROBOTWIN_PYTHON TORCH_EXTENSIONS_DIR
            echo "[RoboTwin ABI] selected python=${ROBOTWIN_PYTHON}"
            return 0
        fi
        echo "[RoboTwin ABI] rejected python=${candidate}" >&2
        index=$((index + 1))
    done

    echo "[ERROR] No RoboTwin Python passed the CuRobo/PyTorch ABI preflight." >&2
    echo "[ERROR] Attempted: ${attempted[*]:-none}" >&2
    echo "[ERROR] Use a RoboTwin image whose CuRobo extension was built against its active PyTorch." >&2
    return 1
}
