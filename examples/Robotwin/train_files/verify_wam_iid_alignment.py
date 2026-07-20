#!/usr/bin/env python3
"""Static audit for the strict IID 80k predictor -> 20k gate recipe."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUNBOOK_DIR_NAME = "\u6267\u884c\u811a\u672c"
PAIRS = (
    ("robotwin_wam_warmup_rand.yaml", "robotwin_wam_gate_rand2clean.yaml"),
    ("robotwin_wam_warmup_clean.yaml", "robotwin_wam_gate_clean2clean.yaml"),
)
BASELINE_NAME = "starvla_qwengroot_robotwin_fastwam.yaml"
OUTPUT_ROOT = (
    "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/"
    "starvla_wam_robotwin_baselinepreserve"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _job_dir() -> Path:
    candidates = []
    if os.environ.get("ROBOTWIN_JOB_DIR"):
        candidates.append(Path(os.environ["ROBOTWIN_JOB_DIR"]))
    candidates.extend(
        (
            REPO_ROOT.parent / "RBT",
            REPO_ROOT / "RBT",
            REPO_ROOT / RUNBOOK_DIR_NAME / "RBT",
        )
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find RBT job directory; checked {candidates}")


def _active_tasks(config: dict) -> list[str]:
    weights = config["framework"]["tasks"]["weights"]
    return [str(name) for name, weight in weights.items() if float(weight) > 0.0]


def _check_common(name: str, config: dict, errors: list[str]) -> None:
    action = config["framework"]["action_model"]
    guidance = config["framework"]["wam"]["guidance"]
    data = config["datasets"]["vla_data"]
    trainer = config["trainer"]
    if action.get("use_correlated_noise") is not False:
        errors.append(f"{name}: use_correlated_noise must be false")
    stale = sorted(key for key in action if str(key).startswith("correlation_"))
    if stale:
        errors.append(f"{name}: correlation keys remain: {stale}")
    if data.get("include_state") is not True:
        errors.append(f"{name}: include_state must be true")
    if data.get("per_device_batch_size") != 12:
        errors.append(f"{name}: per_device_batch_size must be 12 for 8x8 global batch 768")
    if data.get("num_workers") != 4:
        errors.append(f"{name}: train num_workers must be 4 (32 loader processes/node)")
    for key, expected in {
        "bridge_source": "predicted",
        "detach_world": True,
        "detached_prediction_eval_mode": True,
        "world_condition_on_state": True,
        "include_context_in_world_memory": True,
    }.items():
        if guidance.get(key) != expected:
            errors.append(f"{name}: guidance.{key} expected {expected!r}")
    validation = trainer.get("world_validation", {})
    if validation.get("enabled") is not True or validation.get("interval") != 5000:
        errors.append(f"{name}: world validation must run every 5000 steps")
    if validation.get("num_workers") != 2:
        errors.append(f"{name}: world-validation num_workers must be 2")
    if trainer.get("gradient_accumulation_steps") != 1:
        errors.append(f"{name}: gradient_accumulation_steps must be 1")
    if trainer.get("expected_global_batch_size") != 768:
        errors.append(f"{name}: expected_global_batch_size must be 768")


def main() -> None:
    errors: list[str] = []
    job_dir = _job_dir()
    baseline = _load(HERE / BASELINE_NAME)
    for warmup_name, gate_name in PAIRS:
        warmup = _load(HERE / warmup_name)
        gate = _load(HERE / gate_name)
        _check_common(warmup_name, warmup, errors)
        _check_common(gate_name, gate, errors)
        if warmup.get("run_root_dir") != OUTPUT_ROOT:
            errors.append(f"{warmup_name}: run_root_dir must be {OUTPUT_ROOT}")
        if gate.get("run_root_dir") != OUTPUT_ROOT:
            errors.append(f"{gate_name}: run_root_dir must be {OUTPUT_ROOT}")

        warmup_guidance = warmup["framework"]["wam"]["guidance"]
        gate_guidance = gate["framework"]["wam"]["guidance"]
        if warmup["trainer"].get("wam_two_stage_phase") != "predictor_warmup":
            errors.append(f"{warmup_name}: wrong two-stage phase")
        if warmup["trainer"].get("wam_two_stage_recipe") != "baseline_preserving_v3":
            errors.append(f"{warmup_name}: recipe must be baseline_preserving_v3")
        if warmup["trainer"].get("max_train_steps") != 80000:
            errors.append(f"{warmup_name}: predictor warmup must be 80k")
        if _active_tasks(warmup) != ["joint_detached"]:
            errors.append(f"{warmup_name}: only joint_detached may be active")
        if warmup_guidance.get("action_world_bypass") is not True:
            errors.append(f"{warmup_name}: action must bypass world injection")
        if warmup_guidance.get("detach_action_backbone") is not False:
            errors.append(f"{warmup_name}: action loss must update shared Qwen")
        if warmup_guidance.get("detach_world_backbone") is not True:
            errors.append(f"{warmup_name}: world loss must detach shared Qwen")
        if warmup_guidance.get("baseline_action_context") is not True:
            errors.append(f"{warmup_name}: action must use the native IID-baseline Qwen context")
        if warmup["trainer"].get("num_warmup_steps") != 2000:
            errors.append(f"{warmup_name}: LR warmup must match baseline at 2000 steps")
        if warmup["trainer"].get("action_eval_enabled") is not True:
            errors.append(f"{warmup_name}: native baseline action eval must remain enabled")
        if warmup["trainer"].get("seed_before_model_init") is not True:
            errors.append(f"{warmup_name}: model construction must use the declared seed")
        if baseline["trainer"].get("seed_before_model_init") is not True:
            errors.append(f"{BASELINE_NAME}: model construction must use the declared seed")
        if warmup["trainer"].get("eval_interval") != baseline["trainer"].get("eval_interval"):
            errors.append(f"{warmup_name}: action eval cadence must match IID baseline")
        if warmup["framework"].get("qwenvl") != baseline["framework"].get("qwenvl"):
            errors.append(f"{warmup_name}: framework.qwenvl must exactly match IID baseline")
        if warmup["framework"].get("action_model") != baseline["framework"].get("action_model"):
            errors.append(f"{warmup_name}: framework.action_model must exactly match IID baseline")
        if bool(warmup["framework"]["wam"].get("action_gradient_checkpointing", True)):
            errors.append(
                f"{warmup_name}: WAM action_gradient_checkpointing must be false "
                "to preserve native baseline Action DiT execution"
            )
        baseline_data_keys = (
            "data_mix",
            "lerobot_version",
            "fastwam_expected_fps",
            "fastwam_val_fraction",
            "fastwam_split",
            "fastwam_split_seed",
            "fastwam_direct_frame_sampling",
            "include_state",
            "action_type",
            "action_mode",
            "sequential_step_sampling",
            "balance_dataset_weights",
            "balance_trajectory_weights",
            "per_device_batch_size",
            "load_all_data_for_training",
            "obs_image_size",
            "video_backend",
            "num_workers",
            "prefetch_factor",
            "pin_memory",
            "persistent_workers",
        )
        if warmup_name == "robotwin_wam_warmup_rand.yaml":
            # The clean ablation intentionally uses a physically filtered
            # dataset and its own statistics; the full/rand experiment must
            # point to the exact same release and z-score file as baseline.
            baseline_data_keys = (
                "data_root_dir",
                "fastwam_dataset_stats_path",
                *baseline_data_keys,
            )
        baseline_data = baseline["datasets"]["vla_data"]
        warmup_data = warmup["datasets"]["vla_data"]
        for key in baseline_data_keys:
            if warmup_data.get(key) != baseline_data.get(key):
                errors.append(
                    f"{warmup_name}: datasets.vla_data.{key}={warmup_data.get(key)!r} "
                    f"must match baseline {baseline_data.get(key)!r}"
                )
        for key in ("base", "qwen_vl_interface", "action_model"):
            actual_lr = warmup["trainer"]["learning_rate"].get(key)
            baseline_lr = baseline["trainer"]["learning_rate"].get(key)
            if actual_lr != baseline_lr:
                errors.append(
                    f"{warmup_name}: learning_rate.{key}={actual_lr} must match "
                    f"baseline {baseline_lr}"
                )
        for key in ("lr_scheduler_type", "scheduler_specific_kwargs"):
            if warmup["trainer"].get(key) != baseline["trainer"].get(key):
                errors.append(f"{warmup_name}: trainer.{key} must match IID baseline")
        for key in (
            "expected_global_batch_size",
            "seed_before_model_init",
            "max_train_steps",
            "num_warmup_steps",
            "action_eval_enabled",
            "eval_interval",
            "freeze_modules",
            "loss_scale",
            "max_grad_norm",
            "weight_decay",
            "logging_frequency",
            "gradient_clipping",
            "gradient_accumulation_steps",
            "optimizer",
        ):
            if warmup["trainer"].get(key) != baseline["trainer"].get(key):
                errors.append(f"{warmup_name}: trainer.{key} must exactly match IID baseline")

        if gate["trainer"].get("wam_two_stage_phase") != "gate_ft":
            errors.append(f"{gate_name}: wrong two-stage phase")
        if gate["trainer"].get("wam_two_stage_recipe") != "baseline_preserving_v3":
            errors.append(f"{gate_name}: recipe must be baseline_preserving_v3")
        if gate["trainer"].get("max_train_steps") != 20000:
            errors.append(f"{gate_name}: gate fine-tune must be 20k")
        if _active_tasks(gate) != ["policy"]:
            errors.append(f"{gate_name}: only policy may be active")
        if gate_guidance.get("action_world_bypass") is not False:
            errors.append(f"{gate_name}: predicted-world injection must be enabled")
        if gate_guidance.get("detach_world_backbone") is not True:
            errors.append(f"{gate_name}: must preserve the policy-first parent ABI")
        if gate_guidance.get("baseline_action_context") is not True:
            errors.append(f"{gate_name}: gate FT must preserve native baseline action context")
        expected_checkpoint = (
            f"{OUTPUT_ROOT}/{warmup['run_id']}/final_model/pytorch_model.pt"
        )
        if gate["trainer"].get("pretrained_checkpoint") != expected_checkpoint:
            errors.append(
                f"{gate_name}: pretrained checkpoint expected {expected_checkpoint}"
            )
        frozen = {
            item.strip()
            for item in str(gate["trainer"].get("freeze_modules", "")).split(",")
            if item.strip()
        }
        required = {"qwen_vl_interface", "wam_visual_head", "wam_state_ctx", "wam_act_ctx"}
        if not required.issubset(frozen):
            errors.append(f"{gate_name}: missing frozen modules {sorted(required - frozen)}")

        for name in (warmup_name, gate_name):
            job = _load(job_dir / name)
            command = job.get("REQUIRED", {}).get("RUN_SCRIPTS")
            expected = (
                "EXPECTED_NUM_MACHINES=8 ${WORKING_PATH}/run_aidi_rbtw.sh "
                f"examples/Robotwin/train_files/{name}"
            )
            if command != expected:
                errors.append(f"{name}: launcher points to {command!r}, expected {expected!r}")
            required_job = job.get("REQUIRED", {})
            if required_job.get("WORKER_MIN_NUM") != 8 or required_job.get("WORKER_MAX_NUM") != 8:
                errors.append(f"{name}: launcher must request exactly 8 workers")
            if required_job.get("GPU_PER_WORKER") != 8:
                errors.append(f"{name}: launcher must request 8 GPUs per worker")

    # Standalone control: exactly the full/rand predictor-warmup recipe, with
    # only its run identity and the physical world->action path removed.
    no_w2a_name = "robotwin_wam_dual_branch_no_world2action.yaml"
    no_w2a = _load(HERE / no_w2a_name)
    expected_no_w2a = _load(HERE / "robotwin_wam_warmup_rand.yaml")
    expected_no_w2a["run_id"] = (
        "robotwin_wam_m5_fastwam_dualbranch_no_world2action_rand_"
        "state_h32_dinojitx_iid_80k"
    )
    expected_guidance = expected_no_w2a["framework"]["wam"]["guidance"]
    expected_guidance["world_to_action_enabled"] = False
    expected_guidance["log_gate_openness"] = False
    if no_w2a != expected_no_w2a:
        errors.append(
            f"{no_w2a_name}: must differ from robotwin_wam_warmup_rand.yaml only "
            "by run_id, world_to_action_enabled=false, and log_gate_openness=false"
        )
    _check_common(no_w2a_name, no_w2a, errors)
    no_w2a_guidance = no_w2a["framework"]["wam"]["guidance"]
    if no_w2a_guidance.get("world_to_action_enabled") is not False:
        errors.append(f"{no_w2a_name}: world_to_action_enabled must be false")
    if no_w2a_guidance.get("action_world_bypass") is not True:
        errors.append(f"{no_w2a_name}: action_world_bypass must be true")
    if _active_tasks(no_w2a) != ["joint_detached"]:
        errors.append(f"{no_w2a_name}: only joint_detached may be active")
    no_w2a_job = _load(job_dir / no_w2a_name)
    expected_command = (
        "EXPECTED_NUM_MACHINES=8 ${WORKING_PATH}/run_aidi_rbtw.sh "
        f"examples/Robotwin/train_files/{no_w2a_name}"
    )
    if no_w2a_job.get("REQUIRED", {}).get("RUN_SCRIPTS") != expected_command:
        errors.append(f"{no_w2a_name}: launcher must point to the standalone ablation YAML")
    no_w2a_required = no_w2a_job.get("REQUIRED", {})
    if (
        no_w2a_required.get("WORKER_MIN_NUM") != 8
        or no_w2a_required.get("WORKER_MAX_NUM") != 8
        or no_w2a_required.get("GPU_PER_WORKER") != 8
    ):
        errors.append(f"{no_w2a_name}: launcher must request exactly 8x8 GPUs")

    if errors:
        raise SystemExit("WAM IID audit failed:\n- " + "\n- ".join(errors))
    print(
        "WAM IID audit PASS: 80k baseline-preserving warmup (native action path, "
        "action->Qwen, detached world->Qwen) + 20k frozen-predictor gate FT; "
        "standalone dual-branch/no-world-to-action baseline aligned."
    )


if __name__ == "__main__":
    main()
