"""Deployment-only regression tests for the FastWAM checkpoint verifier."""

from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFIER_PATH = Path(__file__).with_name("verify_fastwam_checkpoint_contract.py")


def test_verifier_does_not_import_training_dataloader(tmp_path, monkeypatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pytorch3d" or name.startswith("pytorch3d.") or name.startswith(
            "starVLA.dataloader"
        ):
            raise AssertionError(f"verifier imported a training-only dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("_fastwam_deployment_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints" / "steps_1_pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = (
        REPO_ROOT
        / "examples/Robotwin/train_files/robotwin_wam_dual_branch_no_world2action.yaml"
    ).read_text(encoding="utf-8")
    (run_dir / "config.yaml").write_text(config, encoding="utf-8")
    (run_dir / "config.full.yaml").write_text(config, encoding="utf-8")

    zeros = [0.0] * 14
    modality = {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": [1.0] * 14,
        "q01": zeros,
        "q99": zeros,
        "mask": [True] * 14,
    }
    stats = {"new_embodiment": {"state": modality, "action": modality}}
    (run_dir / "dataset_statistics.json").write_text(json.dumps(stats), encoding="utf-8")

    summary = verifier.verify(
        checkpoint,
        replan_steps=24,
        expected_wam_phase="predictor_warmup",
        expected_world_to_action=False,
    )
    assert summary["normalization"] == "fastwam_zscore"
    assert summary["unnorm_key"] == "new_embodiment"
    assert summary["expects_state"] is True
    assert summary["wam_world_to_action_enabled"] is False


def test_verifier_distinguishes_gate_ft_from_warmup(tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("_fastwam_gate_verifier", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    run_dir = tmp_path / "gate_run"
    checkpoint = run_dir / "checkpoints" / "steps_20000_pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = (
        REPO_ROOT / "examples/Robotwin/train_files/robotwin_wam_gate_rand2clean.yaml"
    ).read_text(encoding="utf-8")
    (run_dir / "config.yaml").write_text(config, encoding="utf-8")
    (run_dir / "config.full.yaml").write_text(config, encoding="utf-8")

    zeros = [0.0] * 14
    modality = {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": [1.0] * 14,
        "q01": zeros,
        "q99": zeros,
        "mask": [True] * 14,
    }
    (run_dir / "dataset_statistics.json").write_text(
        json.dumps({"new_embodiment": {"state": modality, "action": modality}}),
        encoding="utf-8",
    )

    summary = verifier.verify(
        checkpoint,
        replan_steps=24,
        expected_wam_phase="gate_ft",
    )
    assert summary["wam_two_stage_phase"] == "gate_ft"
    assert summary["wam_guidance_enabled"] is True
    assert summary["wam_action_world_bypass"] is False
    assert summary["wam_baseline_action_context"] is True
    assert summary["wam_bridge_source"] == "predicted"
    assert summary["wam_world_eval_mode"] == "correct"

    with pytest.raises(ValueError, match="wam_two_stage_phase"):
        verifier.verify(
            checkpoint,
            replan_steps=24,
            expected_wam_phase="predictor_warmup",
        )


@pytest.mark.parametrize(
    ("config_name", "phase", "step"),
    [
        ("robotwin_wam_sharedqwen_warmup_rand.yaml", "predictor_warmup", 80000),
        ("robotwin_wam_sharedqwen_gate_ft_rand.yaml", "gate_ft", 20000),
    ],
)
def test_verifier_accepts_shared_qwen_two_stage_contract(
    tmp_path,
    config_name: str,
    phase: str,
    step: int,
) -> None:
    spec = importlib.util.spec_from_file_location(
        f"_fastwam_shared_qwen_{phase}", VERIFIER_PATH
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    run_dir = tmp_path / phase
    checkpoint = run_dir / "checkpoints" / f"steps_{step}_pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config = (
        REPO_ROOT / "examples/Robotwin/train_files" / config_name
    ).read_text(encoding="utf-8")
    (run_dir / "config.yaml").write_text(config, encoding="utf-8")
    (run_dir / "config.full.yaml").write_text(config, encoding="utf-8")

    zeros = [0.0] * 14
    modality = {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": [1.0] * 14,
        "q01": zeros,
        "q99": zeros,
        "mask": [True] * 14,
    }
    (run_dir / "dataset_statistics.json").write_text(
        json.dumps({"new_embodiment": {"state": modality, "action": modality}}),
        encoding="utf-8",
    )

    summary = verifier.verify(
        checkpoint,
        replan_steps=24,
        expected_wam_phase=phase,
    )
    assert summary["wam_two_stage_recipe"] == "shared_qwen_queries_v5"
    assert summary["wam_future_query_through_qwen"] is True
    assert summary["qwen_attn_implementation"] == "flash_attention_2"
    assert summary["wam_query_attention_pattern"] == "causal_act_then_future"
    assert summary["wam_single_qwen_forward"] is True
    assert summary["wam_queries_are_final_suffix"] is True
    assert summary["wam_action_query_count"] == 32
    assert summary["wam_future_query_count"] == 64
    assert summary["wam_separate_world_backbone_pass"] is False
    assert summary["wam_detach_action_query_in_world_pass"] is False
