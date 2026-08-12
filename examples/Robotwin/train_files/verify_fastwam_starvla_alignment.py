#!/usr/bin/env python3
"""Static audit of the IID-only FastWAM checkpoint contract."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUNBOOK_DIR_NAME = "\u6267\u884c\u811a\u672c"
IID_PATH = HERE / "starvla_qwengroot_robotwin_fastwam_old.yaml"
DEPLOY_PATH = REPO_ROOT / "examples/Robotwin/eval_files/deploy_policy_fastwam.yml"


def _job_path(name: str) -> Path:
    """Resolve RBT beside starVLA in the real AIDI source/package layout."""

    candidates = []
    if os.environ.get("ROBOTWIN_JOB_DIR"):
        candidates.append(Path(os.environ["ROBOTWIN_JOB_DIR"]) / name)
    candidates.extend(
        [
            REPO_ROOT.parent / "RBT" / name,
            REPO_ROOT / "RBT" / name,
            REPO_ROOT / RUNBOOK_DIR_NAME / "RBT" / name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {name}; checked: {[str(path) for path in candidates]}")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assignment(node: ast.ClassDef, name: str) -> str | None:
    for child in node.body:
        if isinstance(child, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in child.targets
        ):
            return ast.unparse(child.value)
    return None


def main() -> None:
    iid = _load(IID_PATH)
    deploy = _load(DEPLOY_PATH)
    job = _load(_job_path("robotwin_qwengroot_fastwam_iid.yaml"))
    errors = []

    data = iid["datasets"]["vla_data"]
    expected_data = {
        "dataset_py": "lerobot_datasets",
        "data_mix": "robotwin_fastwam",
        "lerobot_version": "v2.0",
        "fastwam_expected_fps": 50,
        "fastwam_val_fraction": 0.01,
        "fastwam_split": "train",
        "fastwam_split_seed": 42,
        "fastwam_direct_frame_sampling": True,
        "include_state": True,
        "obs_image_size": [320, 384],
        "per_device_batch_size": 12,
        "balance_dataset_weights": False,
        "balance_trajectory_weights": False,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": False,
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            errors.append(f"datasets.vla_data.{key}: expected {expected!r}, got {data.get(key)!r}")
    action = iid["framework"]["action_model"]
    for key, expected in {
        "action_dim": 14,
        "state_dim": 14,
        "action_horizon": 32,
        "repeated_diffusion_steps": 2,
        "use_correlated_noise": False,
    }.items():
        if action.get(key) != expected:
            errors.append(f"framework.action_model.{key}: expected {expected!r}, got {action.get(key)!r}")
    if iid.get("trainer", {}).get("expected_global_batch_size") != 768:
        errors.append("trainer.expected_global_batch_size must be 768")

    for key, expected in {
        "policy_name": "model2robotwin_fastwam_interface",
        "normalization_mode": "fastwam_zscore",
        "replan_steps": 24,
    }.items():
        if deploy.get(key) != expected:
            errors.append(f"deploy_policy_fastwam.yml:{key}: expected {expected!r}, got {deploy.get(key)!r}")

    run_script = job.get("REQUIRED", {}).get("RUN_SCRIPTS")
    expected_run_script = (
        "EXPECTED_NUM_MACHINES=8 ${WORKING_PATH}/run_aidi_rbtw.sh "
        "examples/Robotwin/train_files/starvla_qwengroot_robotwin_fastwam_old.yaml"
    )
    if run_script != expected_run_script:
        errors.append(f"AIDI job launches the wrong config: {run_script!r}")
    remark = str(job.get("OPTIONAL", {}).get("REMARK", "")).lower()
    for token in ("composite", "state14", "h32", "bs12", "8x8", "iid"):
        if token not in remark:
            errors.append(f"AIDI job remark is missing {token!r}: {remark!r}")
    required_job = job.get("REQUIRED", {})
    if required_job.get("WORKER_MIN_NUM") != 8 or required_job.get("WORKER_MAX_NUM") != 8:
        errors.append("AIDI baseline launcher must request exactly 8 workers")
    if required_job.get("GPU_PER_WORKER") != 8:
        errors.append("AIDI baseline launcher must request 8 GPUs per worker")

    registry_path = HERE / "data_registry/data_config.py"
    tree = ast.parse(registry_path.read_text(encoding="utf-8"), filename=str(registry_path))
    config_class = _class(tree, "FastWAMRobotWinDataConfig")
    raw_state = "['state.left_joints', 'state.left_gripper', 'state.right_joints', 'state.right_gripper']"
    raw_action = "['action.left_joints', 'action.left_gripper', 'action.right_joints', 'action.right_gripper']"
    if _assignment(config_class, "state_keys") != raw_state:
        errors.append("FastWAM state_keys no longer preserve release order")
    if _assignment(config_class, "action_keys") != raw_action:
        errors.append("FastWAM action_keys no longer preserve release order")
    if _assignment(config_class, "action_indices") != "list(range(32))":
        errors.append("FastWAM action_indices must be list(range(32))")

    dataset_source = (REPO_ROOT / "starVLA/dataloader/fastwam_robotwin_dataset.py").read_text(encoding="utf-8")
    for symbol in (
        "build_robotwin_composite",
        "FastWAMEpochSampler",
        "def _get_trajectories",
        "def get_step_data",
        "def _pack_sample",
        '"action_is_pad"',
    ):
        if symbol not in dataset_source:
            errors.append(f"FastWAM dataset adapter is missing {symbol}")
    interface_source = (REPO_ROOT / "examples/Robotwin/eval_files/model2robotwin_fastwam_interface.py").read_text(
        encoding="utf-8"
    )
    for symbol in ("build_robotwin_composite", "_prepare_state_for_server", "_prepare_action_for_env"):
        if symbol not in interface_source:
            errors.append(f"FastWAM eval adapter is missing {symbol}")

    if errors:
        raise SystemExit("FastWAM alignment failed:\n- " + "\n- ".join(errors))
    print("FastWAM alignment PASS: IID-only composite/state checkpoint ABI; replan=24/32.")


if __name__ == "__main__":
    main()
