"""Regressions for the single-bridge Action--World Co-Flow implementation."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import tempfile
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from starVLA.model.framework.VLM4A.action_world_coflow.attention_mask import (
    build_block_causal_attention_mask,
    render_bool_attention_mask,
)
from starVLA.model.framework.VLM4A.action_world_coflow.bridge import QantaraWorldBridge
from starVLA.model.framework.VLM4A.action_world_coflow.joint_model import (
    ActionWorldCoFlowModel,
    ModalityExpertBlock,
)
from starVLA.model.framework.VLM4A.action_world_coflow.noise_plane import NoisePlaneSampler
from starVLA.model.framework.VLM4A.action_world_coflow.vision_latent import (
    FrozenQwenMultiLayerVisionLatent,
    select_vision_layers,
)
from starVLA.model.framework.VLM4A.QwenActionWorldCoFlow import (
    QwenActionWorldCoFlow,
    QwenActionWorldCoFlowDefaultConfig,
)
from starVLA.model.framework.VLM4A.QwenGR00T import QwenGR00TDefaultConfig
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.train_starvla import VLATrainer


REPO_ROOT = Path(__file__).resolve().parents[5]
RUNBOOK_DIR_NAME = "\u6267\u884c\u811a\u672c"
REMOVED_COFLOW_FIELDS = {
    "segment_boundaries",
    "future_strides",
    "z32_loss_weight",
    "intermediate_state_source",
    "predicted_z16_detach",
    "predicted_action_prefix_detach",
    "z16_teacher_ratio_start",
    "z16_teacher_ratio_end",
    "z16_teacher_decay_steps",
}


def test_single_bridge_attention_visibility_and_padding() -> None:
    block_ids = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    valid = torch.tensor([[False, True, True, True, True]])
    mask = build_block_causal_attention_mask(block_ids, key_valid_mask=valid)
    assert mask.shape == (1, 1, 5, 5)
    assert not bool(mask[0, 0, :2, 2:].any())
    assert bool(mask[0, 0, 2:, 1:].all())
    assert not bool(mask[0, 0, :, 0].any())
    assert render_bool_attention_mask(build_block_causal_attention_mask(block_ids)) == (
        "##...\n##...\n#####\n#####\n#####"
    )
    with pytest.raises(ValueError, match="single-bridge"):
        build_block_causal_attention_mask(torch.tensor([0, 1, 2]))


def test_modality_experts_and_sparse_mot_receive_gradients() -> None:
    torch.manual_seed(29)
    dense = ModalityExpertBlock(
        hidden_size=12, num_heads=3, mlp_ratio=2.0, dropout=0.0
    ).eval()
    sparse = ModalityExpertBlock(
        hidden_size=12,
        num_heads=3,
        mlp_ratio=2.0,
        dropout=0.0,
        expert_mlp_ratios=(2.0, 2.0, 2.0),
        sparse_expert_routing=True,
    ).eval()
    sparse.load_state_dict(dense.state_dict(), strict=True)
    token_types = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    attention_mask = torch.ones(2, 1, 6, 6, dtype=torch.bool)
    dense_input = torch.randn(2, 6, 12, requires_grad=True)
    sparse_input = dense_input.detach().clone().requires_grad_(True)
    torch.testing.assert_close(
        sparse(sparse_input, token_types, attention_mask),
        dense(dense_input, token_types, attention_mask),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    sparse(sparse_input, token_types, attention_mask).square().mean().backward()
    assert all(
        parameter.grad is not None
        for expert in sparse.ffns
        for parameter in expert.parameters()
    )


def test_qantara_bridge_endpoints_and_reprojection() -> None:
    start = torch.zeros(2, 3, 4)
    target = torch.ones_like(start) * 2
    noise = torch.ones_like(start)
    bridge = QantaraWorldBridge(noise_scale=1.0, prediction_type="qantara_x_delta")
    torch.testing.assert_close(bridge.interpolate(start, target, torch.zeros(2), noise), start)
    torch.testing.assert_close(bridge.interpolate(start, target, torch.ones(2), noise), target)
    torch.testing.assert_close(
        bridge.reproject(start, target, 0.25, add_marginal_noise=False),
        torch.full_like(start, 0.5),
    )


@pytest.mark.parametrize("mode", ["policy", "forward", "inverse", "joint", "diagonal"])
def test_single_bridge_noise_plane_loci(mode: str) -> None:
    ratios = {
        f"{name}_ratio": float(name == mode)
        for name in ("policy", "forward", "inverse", "joint", "diagonal")
    }
    sampler = NoisePlaneSampler(ratios)
    torch.manual_seed(17)
    batch = sampler.sample(256, "cpu")
    assert batch.tau_action.shape == batch.tau_world.shape == (256,)
    assert bool(((batch.tau_action >= 0) & (batch.tau_action <= 1)).all())
    assert bool(((batch.tau_world >= 0) & (batch.tau_world <= 1)).all())
    if mode == "policy":
        assert bool((batch.tau_world == 0).all())
    elif mode == "forward":
        assert bool((batch.tau_action == 1).all())
    elif mode == "inverse":
        assert bool((batch.tau_world == 1).all())
    elif mode == "joint":
        assert not torch.equal(batch.tau_action, batch.tau_world)
    else:
        torch.testing.assert_close(batch.tau_action, batch.tau_world)


def test_noise_plane_bfloat16_contract() -> None:
    sampler = NoisePlaneSampler(
        {
            "policy_ratio": 0.0,
            "forward_ratio": 0.0,
            "inverse_ratio": 0.0,
            "joint_ratio": 0.0,
            "diagonal_ratio": 1.0,
        }
    )

    def promoted_world_sample(_self, batch_size, device, _dtype):
        return torch.rand(batch_size, device=device, dtype=torch.float32)

    sampler._sample_world_variable = types.MethodType(promoted_world_sample, sampler)
    batch = sampler.sample(32, "cpu", dtype=torch.bfloat16)
    assert batch.tau_action.dtype == batch.tau_world.dtype == torch.bfloat16
    torch.testing.assert_close(batch.tau_action, batch.tau_world)


def test_fixed_multilayer_qwen_target_is_deterministic() -> None:
    assert select_vision_layers(27, "evenly_spaced", 4, None) == (0, 9, 17, 26)
    extractor = FrozenQwenMultiLayerVisionLatent(
        vision_depth=4,
        strategy="evenly_spaced",
        num_layers=2,
        fusion_type="fixed_mean",
        normalization="layernorm",
        token_grid=(2, 2),
    )
    assert sum(parameter.numel() for parameter in extractor.parameters()) == 0
    captured = {
        0: torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6),
        3: torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6) + 7,
    }
    first = extractor.pool_captured(captured, torch.tensor([[1, 4, 4]]), spatial_merge_size=2)
    second = extractor.pool_captured(captured, torch.tensor([[1, 4, 4]]), spatial_merge_size=2)
    assert first.shape == (1, 4, 6)
    torch.testing.assert_close(first, second)


def test_future_encoder_makes_exactly_one_t16_call() -> None:
    framework = QwenActionWorldCoFlow.__new__(QwenActionWorldCoFlow)
    nn.Module.__init__(framework)

    class FakeQwen:
        def __init__(self):
            self.batch_sizes = []

        def build_qwenvl_inputs(self, images, instructions):
            self.batch_sizes.append(len(images))
            assert len(images) == len(instructions)
            return {
                "pixel_values": torch.arange(len(images), dtype=torch.float32).reshape(-1, 1),
                "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long),
            }

    class FakeExtractor(nn.Module):
        def encode(self, _visual, pixel_values, _grid_thw):
            return pixel_values[:, None, :]

    framework.qwen_vl_interface = FakeQwen()
    framework.world_target_extractor = FakeExtractor()
    visual = nn.Linear(1, 1, bias=False)
    framework._visual = types.MethodType(lambda _self: visual, framework)
    framework._resize_batch_images = types.MethodType(lambda _self, images: images, framework)
    framework._autocast_context = types.MethodType(lambda _self: nullcontext(), framework)
    examples = [{"image_16": [object()], "future_valid_16": 1} for _ in range(3)]
    target = framework._encode_future_target(examples)
    assert target.shape == (3, 1, 1)
    assert not target.requires_grad
    assert framework.qwen_vl_interface.batch_sizes == [3]

    invalid = copy.deepcopy(examples)
    invalid[0]["image_32"] = [object()]
    with pytest.raises(ValueError, match="removed multi-bridge"):
        framework._encode_future_target(invalid)


def _tiny_model(
    *,
    action_horizon: int = 16,
    coflow_overrides: dict | None = None,
) -> ActionWorldCoFlowModel:
    action = {
        "action_horizon": action_horizon,
        "action_dim": 3,
        "state_dim": 2,
        "prediction_type": "jit_x",
        "jit_t_eps": 0.05,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "use_correlated_noise": False,
    }
    coflow = {
        "world_token_grid": [2, 2],
        "num_world_tokens": 4,
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "enable_gradient_checkpointing": False,
        "world_bridge_type": "qantara_brownian_bridge",
        "bridge_noise_scale": 1.0,
        "world_prediction_type": "qantara_x_delta",
        "action_timestep_sampling": "starvla_gr00t",
        "world_timestep_sampling": "qantara_monotone",
        "action_loss_weight": 1.0,
        "world_loss_weight": 0.01,
        "noise_plane_sampling": {
            "policy_ratio": 1.0,
            "forward_ratio": 0.0,
            "inverse_ratio": 0.0,
            "joint_ratio": 0.0,
            "diagonal_ratio": 0.0,
        },
        "action_inference_steps": 2,
        "world_inference_steps": 2,
        "default_inference_mode": "policy",
        "log_attention_statistics": True,
    }
    if coflow_overrides:
        coflow.update(copy.deepcopy(coflow_overrides))
    return ActionWorldCoFlowModel(
        context_dim=8,
        world_dim=6,
        action_config=action,
        coflow_config=coflow,
    )


def _batch(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "context": torch.randn(batch, 5, 8),
        "context_valid": torch.ones(batch, 5, dtype=torch.bool),
        "z0": torch.randn(batch, 4, 6),
        "z16_target": torch.randn(batch, 4, 6),
        "actions": torch.randn(batch, 16, 3),
        "state": torch.randn(batch, 1, 2),
        "action_is_pad": torch.zeros(batch, 16, dtype=torch.bool),
        "future_valid_16": torch.ones(batch),
    }


def test_model_rejects_h32_and_has_no_multibridge_api() -> None:
    with pytest.raises(ValueError, match="single-bridge"):
        _tiny_model(action_horizon=32)
    model = _tiny_model()
    assert not hasattr(model, "num_blocks")
    assert not hasattr(model, "segment_boundaries")
    source = inspect.getsource(ActionWorldCoFlowModel)
    for removed in ("z32", "a2", "prefix_action", "teacher_ratio", "_sample_block"):
        assert removed not in source
    forward_parameters = inspect.signature(model.forward_train).parameters
    assert "z32_target" not in forward_parameters
    assert "future_valid_32" not in forward_parameters


def test_single_bridge_training_and_backward() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    output = model.forward_train(**_batch(), global_step=10)
    assert output["action_loss"].ndim == 0 and torch.isfinite(output["action_loss"])
    assert output["coflow_mode_policy_ratio"] == 1
    assert output["coflow_z16_loss_raw"].ndim == 0
    assert not any("32" in key or "prefix" in key or "teacher" in key for key in output)
    output["action_loss"].backward()
    assert model.transformer.layers[0].attention.qkv.weight.grad is not None


def test_policy_and_diagonal_inference_use_one_bridge_call() -> None:
    model = _tiny_model().eval()
    inputs = _batch()
    common = {key: inputs[key] for key in ("context", "context_valid", "z0", "state")}
    original = model._assemble_sequence
    calls = []

    def wrapped(_self, **kwargs):
        calls.append((kwargs["tau_action"].clone(), kwargs["tau_world"].clone()))
        return original(**kwargs)

    model._assemble_sequence = types.MethodType(wrapped, model)
    with torch.no_grad():
        actions, future = model.sample_actions(**common, inference_mode="policy", output_horizon=16)
    assert actions.shape == (2, 16, 3)
    assert future.shape == (2, 4, 6)
    assert len(calls) == model.action_inference_steps
    assert all(bool((tau_world == 0).all()) for _, tau_world in calls)

    calls.clear()
    with torch.no_grad():
        model.sample_actions(**common, inference_mode="diagonal", output_horizon=16)
    assert len(calls) == model.action_inference_steps
    assert all(torch.equal(tau_action, tau_world) for tau_action, tau_world in calls)
    with pytest.raises(ValueError, match="must be 16"):
        model.sample_actions(**common, output_horizon=32)


def test_inference_seed_is_local_and_reproducible() -> None:
    model = _tiny_model().eval()
    inputs = _batch()
    common = {key: inputs[key] for key in ("context", "context_valid", "z0", "state")}
    torch.manual_seed(101)
    global_state = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = model.sample_actions(**common, inference_seed=123)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    with torch.no_grad():
        second = model.sample_actions(**common, inference_seed=123)
    for left, right in zip(first, second, strict=True):
        torch.testing.assert_close(left, right)


def test_bfloat16_inference_needs_no_ambient_autocast() -> None:
    model = _tiny_model().to(torch.bfloat16).eval()
    inputs = _batch()
    common = {
        key: value.to(torch.bfloat16) if value.is_floating_point() else value
        for key, value in inputs.items()
        if key in ("context", "context_valid", "z0", "state")
    }
    with torch.no_grad():
        actions, future = model.sample_actions(**common)
    assert actions.dtype == future.dtype == torch.bfloat16
    assert bool(torch.isfinite(actions).all()) and bool(torch.isfinite(future).all())


def test_invalid_future_target_and_padded_actions_are_loss_inert() -> None:
    model = _tiny_model()
    inputs = _batch()
    inputs["future_valid_16"] = torch.zeros(2)
    torch.manual_seed(5)
    first = model.forward_train(**inputs, global_step=0)
    changed = copy.deepcopy(inputs)
    changed["z16_target"] = torch.full_like(changed["z16_target"], -1.0e6)
    torch.manual_seed(5)
    second = model.forward_train(**changed, global_step=0)
    torch.testing.assert_close(first["action_loss"], second["action_loss"])
    assert first["coflow_z16_loss_raw"] == 0

    errors = torch.tensor([[[1.0], [9.0], [100.0]]])
    valid = torch.tensor([[True, False, False]])
    assert model._masked_action_loss(errors, valid) == 1.0


def test_world_loss_schedule_switches_at_step_50000() -> None:
    model = _tiny_model(
        coflow_overrides={
            "world_loss_weight": 0.1,
            "world_loss_schedule": {
                "enabled": True,
                "transition_step": 50000,
                "before_weight": 0.1,
                "after_weight": 0.025,
            },
        }
    )
    assert model.world_loss_weight_at_step(49999) == pytest.approx(0.1)
    assert model.world_loss_weight_at_step(50000) == pytest.approx(0.025)


def test_trainer_dashboard_has_only_single_bridge_losses() -> None:
    total = torch.tensor(2.0)
    output = {
        "coflow_action_loss_raw": torch.tensor(1.0),
        "coflow_z16_loss_raw": torch.tensor(0.5),
        "coflow_world_loss_raw": torch.tensor(0.5),
        "coflow_action_loss_weighted": torch.tensor(1.0),
        "coflow_world_loss_weighted": torch.tensor(1.0),
    }
    metrics = VLATrainer._build_native_loss_metrics(output, total)
    assert metrics["train/task"] == "action_world_coflow"
    assert metrics["train/loss_total"] == 2.0
    assert "train/z32_loss_raw" not in metrics
    assert "train/actual_gt_z16_prefix_ratio" not in metrics
    assert "train/z32_loss_raw" not in VLATrainer._COFLOW_DASHBOARD_KEYS


def test_only_better_yaml_and_job_define_the_single_bridge_experiment() -> None:
    train_dir = REPO_ROOT / "examples/Robotwin/train_files"
    runbook = REPO_ROOT / RUNBOOK_DIR_NAME / "RBT"
    assert not (train_dir / "robotwin_action_world_coflow.yaml").exists()
    assert not (runbook / "robotwin_action_world_coflow.yaml").exists()
    assert not (runbook / "run_coflow.md").exists()

    config = yaml.safe_load(
        (train_dir / "robotwin_action_world_coflow_better.yaml").read_text(encoding="utf-8")
    )
    framework = config["framework"]
    coflow = framework["action_world_coflow"]
    data = config["datasets"]["vla_data"]
    assert framework["name"] == "QwenActionWorldCoFlow"
    assert framework["action_model"]["action_horizon"] == 16
    assert not (REMOVED_COFLOW_FIELDS & set(coflow))
    assert data["data_mix"] == "robotwin_fastwam_h16"
    assert data["fastwam_action_world_coflow_targets"] is True
    assert "fastwam_coflow_future_strides" not in data
    assert data["include_state"] is True
    assert data["per_device_batch_size"] == 12
    assert config["trainer"]["expected_global_batch_size"] == 768

    job = yaml.safe_load(
        (runbook / "robotwin_action_world_coflow_better.yaml").read_text(encoding="utf-8")
    )
    assert job["REQUIRED"]["WORKER_MIN_NUM"] == 8
    assert job["REQUIRED"]["WORKER_MAX_NUM"] == 8
    assert job["REQUIRED"]["GPU_PER_WORKER"] == 8
    assert "robotwin_action_world_coflow_better.yaml" in job["REQUIRED"]["RUN_SCRIPTS"]
    run_notes = (runbook / "run.sh").read_text(encoding="utf-8")
    assert "aidi-inf-cli job submit -f robotwin_action_world_coflow_better.yaml" in run_notes
    assert "aidi-inf-cli job submit -f robotwin_action_world_coflow.yaml" not in run_notes
    assert "qwenlatent_h32" not in run_notes


def test_checkpoint_audit_accepts_only_single_bridge_h16() -> None:
    verifier_path = (
        REPO_ROOT
        / "examples/Robotwin/eval_files/verify_action_world_coflow_checkpoint_contract.py"
    )
    spec = importlib.util.spec_from_file_location("coflow_checkpoint_verifier", verifier_path)
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verify = verifier.verify

    config_path = (
        REPO_ROOT
        / "examples/Robotwin/train_files/robotwin_action_world_coflow_better.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    zeros = [0.0] * 14
    stats = {
        "new_embodiment": {
            modality: {
                "min": zeros,
                "max": zeros,
                "mean": zeros,
                "std": [1.0] * 14,
                "q01": zeros,
                "q99": zeros,
                **({"mask": [True] * 14} if modality == "action" else {}),
            }
            for modality in ("state", "action")
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        checkpoint = run_dir / "checkpoints/steps_1_pytorch_model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        (run_dir / "config.full.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
        (run_dir / "dataset_statistics.json").write_text(
            json.dumps(stats), encoding="utf-8"
        )
        summary = verify(checkpoint, 16, inference_mode="policy", inference_horizon=16)
        assert summary["checkpoint_chunk"] == summary["bridge_horizon"] == 16

        invalid = copy.deepcopy(config)
        invalid["framework"]["action_model"]["action_horizon"] = 32
        (run_dir / "config.full.yaml").write_text(
            yaml.safe_dump(invalid), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="action_horizon=16"):
            verify(checkpoint, 16, inference_mode="policy", inference_horizon=16)


def test_registration_and_defaults_do_not_contain_multibridge_fields() -> None:
    defaults = QwenActionWorldCoFlowDefaultConfig()
    legacy_defaults = QwenGR00TDefaultConfig()
    assert defaults.name == "QwenActionWorldCoFlow"
    assert defaults.enable_action_world_coflow is False
    assert defaults.action_model["action_horizon"] == 16
    assert not (REMOVED_COFLOW_FIELDS & set(defaults.action_world_coflow))
    assert not hasattr(legacy_defaults, "enable_action_world_coflow")
    assert FRAMEWORK_REGISTRY["QwenActionWorldCoFlow"] is QwenActionWorldCoFlow
