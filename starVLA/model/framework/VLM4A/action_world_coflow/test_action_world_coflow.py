"""Numerical and contract regressions for the isolated Co-Flow implementation."""

from __future__ import annotations

import inspect
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf
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
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups


REPO_ROOT = Path(__file__).resolve().parents[5]


def test_block_causal_mask_exact_visibility_and_padding() -> None:
    block_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    valid = torch.tensor([[False, True, True, True, True, True, True]])
    mask = build_block_causal_attention_mask(block_ids, key_valid_mask=valid)
    assert mask.shape == (1, 1, 7, 7)
    # Context cannot read either action/world block.
    assert not bool(mask[0, 0, 0, 2:].any())
    # Same block is bidirectional; block 1 cannot see block 2.
    assert bool(mask[0, 0, 2, 4]) and bool(mask[0, 0, 4, 2])
    assert not bool(mask[0, 0, 2:5, 5:].any())
    # Block 2 reads context + both blocks, except the padded context key.
    assert bool(mask[0, 0, 5:, 1:].all())
    assert not bool(mask[0, 0, :, 0].any())

    unpadded = build_block_causal_attention_mask(block_ids)
    assert render_bool_attention_mask(unpadded) == (
        "##.....\n"
        "##.....\n"
        "#####..\n"
        "#####..\n"
        "#####..\n"
        "#######\n"
        "#######"
    )


def test_all_modality_specific_ffns_execute_and_receive_gradients() -> None:
    block = ModalityExpertBlock(hidden_size=12, num_heads=3, mlp_ratio=2.0, dropout=0.0)
    calls = [0, 0, 0]
    handles = []
    for index, ffn in enumerate(block.ffns):
        def mark_called(_module, _inputs, _output, *, index=index):
            calls[index] += 1

        handles.append(ffn.register_forward_hook(mark_called))
    try:
        x = torch.randn(2, 3, 12, requires_grad=True)
        token_types = torch.tensor([0, 1, 2], dtype=torch.long)
        attention_mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
        block(x, token_types, attention_mask).square().mean().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert calls == [1, 1, 1]
    assert all(parameter.grad is not None for ffn in block.ffns for parameter in ffn.parameters())


def test_qantara_bridge_endpoints_variance_and_reprojection() -> None:
    start = torch.zeros(2, 3, 4)
    target = torch.ones_like(start) * 2
    noise = torch.ones_like(start)
    bridge = QantaraWorldBridge(noise_scale=1.0, prediction_type="qantara_x_delta")
    torch.testing.assert_close(bridge.interpolate(start, target, torch.zeros(2), noise), start)
    torch.testing.assert_close(bridge.interpolate(start, target, torch.ones(2), noise), target)
    middle = bridge.interpolate(start, target, torch.full((2,), 0.5), noise)
    torch.testing.assert_close(middle, torch.full_like(middle, 1.5))
    raw_delta = target - start
    clean = bridge.clean_from_prediction(raw_delta, start)
    torch.testing.assert_close(clean, target)
    torch.testing.assert_close(
        bridge.reproject(start, clean, 0.25, add_marginal_noise=False),
        torch.full_like(start, 0.5),
    )


def test_noise_plane_ratio_validation_and_policy_edge() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        NoisePlaneSampler({"policy_ratio": 0.9})
    sampler = NoisePlaneSampler(
        {
            "policy_ratio": 1.0,
            "forward_ratio": 0.0,
            "inverse_ratio": 0.0,
            "joint_ratio": 0.0,
            "diagonal_ratio": 0.0,
        }
    )
    batch = sampler.sample(32, "cpu")
    assert bool((batch.tau_world == 0).all())
    assert bool(((batch.tau_action >= 0) & (batch.tau_action < 1)).all())
    # StarVLA keeps one action FM time over the whole 32-step chunk.
    torch.testing.assert_close(batch.tau_action[:, 0], batch.tau_action[:, 1])


@pytest.mark.parametrize("mode", ["policy", "forward", "inverse", "joint", "diagonal"])
def test_every_noise_plane_locus_has_the_declared_semantics(mode: str) -> None:
    ratios = {f"{name}_ratio": float(name == mode) for name in ("policy", "forward", "inverse", "joint", "diagonal")}
    sampler = NoisePlaneSampler(ratios, world_timestep_sampling="qantara_monotone")
    torch.manual_seed(17)
    batch = sampler.sample(256, "cpu")
    assert bool(((batch.tau_action >= 0) & (batch.tau_action <= 1)).all())
    assert bool(((batch.tau_world >= 0) & (batch.tau_world <= 1)).all())
    if mode == "policy":
        assert bool((batch.tau_world == 0).all())
    elif mode == "forward":
        assert bool((batch.tau_action == 1).all())
        assert bool((batch.tau_world[:, 1] <= batch.tau_world[:, 0]).all())
    elif mode == "inverse":
        assert bool((batch.tau_world == 1).all())
    elif mode == "joint":
        # User-facing joint means the full 2-D interior; the two axes are
        # independently sampled (Qantara calls this locus ``square``).
        assert not bool(torch.equal(batch.tau_action, batch.tau_world))
        assert bool((batch.tau_world[:, 1] <= batch.tau_world[:, 0]).all())
    else:
        torch.testing.assert_close(batch.tau_action, batch.tau_world)


def test_fixed_multilayer_qwen_target_has_no_trainable_pooling_parameters() -> None:
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
    # Raw Qwen order for one t=1,h=4,w=4 image; shape and deterministic fixed
    # pooling are what define the target ABI.
    captured = {
        0: torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6),
        3: torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6) + 7,
    }
    first = extractor.pool_captured(captured, torch.tensor([[1, 4, 4]]), spatial_merge_size=2)
    second = extractor.pool_captured(captured, torch.tensor([[1, 4, 4]]), spatial_merge_size=2)
    assert first.shape == (1, 4, 6)
    torch.testing.assert_close(first, second)


def test_qwen_spatial_merge_token_order_is_restored_to_physical_rows_and_columns() -> None:
    physical = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    # Qwen flattens [block_row, block_col, intra_row, intra_col].
    qwen_order = physical.view(2, 2, 2, 2).permute(0, 2, 1, 3).reshape(16, 1)
    restored = FrozenQwenMultiLayerVisionLatent._restore_spatial_grid(
        qwen_order, t=1, h=4, w=4, merge=2
    )
    torch.testing.assert_close(restored[0, :, :, 0], physical)


def test_dynamic_qwen_layer_selection_supports_explicit_and_all_layers() -> None:
    assert select_vision_layers(6, "explicit", 4, [0, 2, -1]) == (0, 2, 5)
    assert select_vision_layers(4, "all_layers", 1, None) == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="outside depth"):
        select_vision_layers(4, "explicit", 1, [4])


def test_fixed_and_trainable_layer_fusion_interfaces_are_explicit() -> None:
    fixed = FrozenQwenMultiLayerVisionLatent(
        vision_depth=4,
        strategy="explicit",
        num_layers=2,
        explicit_layers=[0, 3],
        fusion_type="fixed_weighted_mean",
        fixed_layer_weights=[1, 3],
        token_grid=(2, 2),
    )
    torch.testing.assert_close(fixed.layer_weights, torch.tensor([0.25, 0.75]))
    assert sum(parameter.numel() for parameter in fixed.parameters()) == 0
    trainable = FrozenQwenMultiLayerVisionLatent(
        vision_depth=4,
        strategy="explicit",
        num_layers=2,
        explicit_layers=[0, 3],
        fusion_type="trainable_weighted_mean",
        token_grid=(2, 2),
    )
    assert trainable.layer_weight_logits.requires_grad
    assert sum(parameter.numel() for parameter in trainable.parameters()) == 2


def test_future_horizons_use_separate_b_sized_qwen_vision_calls() -> None:
    # Real Qwen3-VL packed vision features are batch-slot dependent.  Combining
    # t+16/t+32 into one 2B call breaks the fixed-space equality with current
    # Z0, so this ABI test locks both horizons to separate B-sized calls.
    framework = QwenActionWorldCoFlow.__new__(QwenActionWorldCoFlow)
    nn.Module.__init__(framework)

    class FakeQwen:
        def __init__(self):
            self.batch_sizes = []

        def build_qwenvl_inputs(self, images, instructions):
            batch = len(images)
            self.batch_sizes.append(batch)
            assert len(instructions) == batch
            return {
                "pixel_values": torch.arange(batch, dtype=torch.float32).reshape(batch, 1),
                "image_grid_thw": torch.ones(batch, 3, dtype=torch.long),
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
    examples = [
        {
            "image_16": [object()],
            "image_32": [object()],
            "future_valid_16": 1,
            "future_valid_32": 1,
            "coflow_future_strides": [16, 32],
        }
        for _ in range(3)
    ]
    z16, z32 = framework._encode_future_targets(examples)
    assert framework.qwen_vl_interface.batch_sizes == [3, 3]
    assert z16.shape == z32.shape == (3, 1, 1)
    assert not z16.requires_grad and not z32.requires_grad


def _tiny_model() -> ActionWorldCoFlowModel:
    action = {
        "action_horizon": 32,
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
        "segment_boundaries": [16, 32],
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
        "z16_loss_weight": 0.5,
        "z32_loss_weight": 0.5,
        "intermediate_state_source": "scheduled",
        "predicted_z16_detach": True,
        "predicted_action_prefix_detach": True,
        "z16_teacher_ratio_start": 1.0,
        "z16_teacher_ratio_end": 0.0,
        "z16_teacher_decay_steps": 10,
        "noise_plane_sampling": {
            "policy_ratio": 1.0,
            "forward_ratio": 0.0,
            "inverse_ratio": 0.0,
            "joint_ratio": 0.0,
            "diagonal_ratio": 0.0,
        },
        "action_inference_steps": 1,
        "world_inference_steps": 1,
        "log_attention_statistics": True,
    }
    return ActionWorldCoFlowModel(
        context_dim=8,
        world_dim=6,
        action_config=action,
        coflow_config=coflow,
    )


def test_joint_model_policy_training_and_predicted_only_inference() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    batch = 2
    common = {
        "context": torch.randn(batch, 5, 8),
        "context_valid": torch.ones(batch, 5, dtype=torch.bool),
        "z0": torch.randn(batch, 4, 6),
        "state": torch.randn(batch, 1, 2),
    }
    output = model.forward_train(
        **common,
        z16_target=torch.randn(batch, 4, 6),
        z32_target=torch.randn(batch, 4, 6),
        actions=torch.randn(batch, 32, 3),
        action_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        future_valid_16=torch.ones(batch),
        future_valid_32=torch.ones(batch),
        global_step=10,
    )
    assert output["action_loss"].ndim == 0 and torch.isfinite(output["action_loss"])
    assert output["coflow_mode_policy_ratio"] == 1
    assert output["coflow_teacher_ratio_configured"] == 0
    output["action_loss"].backward()
    assert model.transformer.layers[0].attention.qkv.weight.grad is not None

    model.eval()
    # sample_actions has no future-target argument by design: Block 2 can only
    # consume the Z16 returned by Block 1.
    with torch.no_grad():
        actions, z16, z32 = model.sample_actions(**common)
    assert actions.shape == (batch, 32, 3)
    assert z16.shape == z32.shape == (batch, 4, 6)


def test_coflow_gradient_checkpointing_preserves_modality_ffn_backward() -> None:
    torch.manual_seed(11)
    model = _tiny_model().train()
    model.transformer.gradient_checkpointing = True
    batch = 2
    output = model.forward_train(
        context=torch.randn(batch, 5, 8),
        context_valid=torch.ones(batch, 5, dtype=torch.bool),
        z0=torch.randn(batch, 4, 6),
        z16_target=torch.randn(batch, 4, 6),
        z32_target=torch.randn(batch, 4, 6),
        actions=torch.randn(batch, 32, 3),
        state=torch.randn(batch, 1, 2),
        action_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        future_valid_16=torch.ones(batch),
        future_valid_32=torch.ones(batch),
        global_step=10,
    )
    output["action_loss"].backward()
    assert model.transformer.layers[0].ffns[1][0].weight.grad is not None
    assert model.transformer.layers[0].attention.qkv.weight.grad is not None


def test_invalid_future_targets_are_loss_inert_and_cannot_leak_into_actions() -> None:
    model = _tiny_model()
    # Inverse mode would expose clean future targets at tau_z=1.  Invalid
    # episode-tail targets must override that edge to tau_z=0 and use the
    # predicted intermediate state, making their actual pixel latents inert.
    model.noise_plane.ratios = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float64)
    batch = 2
    kwargs = {
        "context": torch.randn(batch, 5, 8),
        "context_valid": torch.ones(batch, 5, dtype=torch.bool),
        "z0": torch.randn(batch, 4, 6),
        "actions": torch.randn(batch, 32, 3),
        "state": torch.randn(batch, 1, 2),
        "action_is_pad": torch.zeros(batch, 32, dtype=torch.bool),
        "future_valid_16": torch.zeros(batch),
        "future_valid_32": torch.zeros(batch),
        "global_step": 0,
    }
    first_z16 = torch.randn(batch, 4, 6)
    first_z32 = torch.randn(batch, 4, 6)
    second_z16 = torch.full((batch, 4, 6), 1.0e4)
    second_z32 = torch.full((batch, 4, 6), -1.0e4)
    torch.manual_seed(123)
    first = model.forward_train(
        **kwargs,
        z16_target=first_z16,
        z32_target=first_z32,
    )
    torch.manual_seed(123)
    second = model.forward_train(
        **kwargs,
        z16_target=second_z16,
        z32_target=second_z32,
    )
    torch.testing.assert_close(first["action_loss"], second["action_loss"], rtol=0, atol=0)
    assert first["coflow_world_loss_raw"] == 0
    assert first["coflow_teacher_ratio_realized"] == 0
    assert first["coflow_predicted_z16_ratio"] == 0

    invalid_order = dict(kwargs)
    invalid_order["future_valid_32"] = torch.ones(batch)
    with pytest.raises(ValueError, match="future_valid_32=true requires future_valid_16=true"):
        model.forward_train(
            **invalid_order,
            z16_target=torch.randn(batch, 4, 6),
            z32_target=torch.randn(batch, 4, 6),
        )


def test_padded_action_tokens_are_hidden_keys_and_cannot_change_valid_loss() -> None:
    model = _tiny_model()
    batch = 2
    action_is_pad = torch.zeros(batch, 32, dtype=torch.bool)
    action_is_pad[:, 8:] = True
    actions = torch.randn(batch, 32, 3)
    changed_actions = actions.clone()
    changed_actions[:, 8:] = 1.0e4
    kwargs = {
        "context": torch.randn(batch, 5, 8),
        "context_valid": torch.ones(batch, 5, dtype=torch.bool),
        "z0": torch.randn(batch, 4, 6),
        "z16_target": torch.randn(batch, 4, 6),
        "z32_target": torch.randn(batch, 4, 6),
        "state": torch.randn(batch, 1, 2),
        "action_is_pad": action_is_pad,
        "future_valid_16": torch.zeros(batch),
        "future_valid_32": torch.zeros(batch),
        "global_step": 10,
    }
    torch.manual_seed(91)
    first = model.forward_train(**kwargs, actions=actions)
    torch.manual_seed(91)
    second = model.forward_train(**kwargs, actions=changed_actions)
    torch.testing.assert_close(first["action_loss"], second["action_loss"], rtol=0, atol=0)

    assembled = model._assemble_sequence(
        context=kwargs["context"],
        context_valid=kwargs["context_valid"],
        state=kwargs["state"],
        z0=kwargs["z0"],
        a1=actions[:, :16],
        z16=kwargs["z16_target"],
        tau_a1=torch.full((batch,), 0.5),
        tau_z16=torch.full((batch,), 0.5),
        a1_valid=~action_is_pad[:, :16],
    )
    padded_action_keys = torch.arange(assembled.slices["a1"].start + 8, assembled.slices["a1"].stop)
    assert not bool(assembled.attention_mask[..., padded_action_keys].any())


def test_action_loss_excludes_forward_edge_rows_from_its_denominator() -> None:
    error = torch.tensor([[[4.0], [9.0]], [[100.0], [100.0]]])
    valid = torch.tensor([[True, True], [False, False]])
    mixed = ActionWorldCoFlowModel._masked_action_loss(error, valid)
    only_supervised = ActionWorldCoFlowModel._masked_action_loss(error[:1], valid[:1])
    torch.testing.assert_close(mixed, only_supervised)
    assert float(mixed) == pytest.approx(6.5)


def test_forward_and_inverse_edges_preserve_clean_prefix_semantics() -> None:
    model = _tiny_model()
    batch = 2
    kwargs = {
        "context": torch.randn(batch, 5, 8),
        "context_valid": torch.ones(batch, 5, dtype=torch.bool),
        "z0": torch.randn(batch, 4, 6),
        "z16_target": torch.randn(batch, 4, 6),
        "z32_target": torch.randn(batch, 4, 6),
        "actions": torch.randn(batch, 32, 3),
        "state": torch.randn(batch, 1, 2),
        "action_is_pad": torch.zeros(batch, 32, dtype=torch.bool),
        "future_valid_16": torch.ones(batch),
        "future_valid_32": torch.ones(batch),
        "global_step": 10,
    }

    model.noise_plane.ratios = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float64)
    forward = model.forward_train(**kwargs)
    assert forward["coflow_forward_clean_action_prefix_ratio"] == 1

    model.noise_plane.ratios = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float64)
    inverse = model.forward_train(**kwargs)
    assert inverse["coflow_inverse_clean_z16_prefix_ratio"] == 1
    assert inverse["coflow_actual_gt_z16_prefix_ratio"] == 1
    # Explicit inverse conditioning isn't counted as scheduled teacher forcing.
    assert inverse["coflow_teacher_ratio_realized"] == 0


def test_default_scheduled_policy_edge_is_predicted_only_from_step_zero() -> None:
    model = _tiny_model()
    batch = 3
    output = model.forward_train(
        context=torch.randn(batch, 5, 8),
        context_valid=torch.ones(batch, 5, dtype=torch.bool),
        z0=torch.randn(batch, 4, 6),
        z16_target=torch.randn(batch, 4, 6),
        z32_target=torch.randn(batch, 4, 6),
        actions=torch.randn(batch, 32, 3),
        state=torch.randn(batch, 1, 2),
        action_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
        future_valid_16=torch.ones(batch),
        future_valid_32=torch.ones(batch),
        # Configured teacher ratio is exactly one here, but policy is never
        # eligible for clean Z16 exposure.
        global_step=0,
    )
    assert output["coflow_teacher_ratio_configured"] == 1
    assert output["coflow_teacher_ratio_realized"] == 0
    assert output["coflow_actual_gt_z16_prefix_ratio"] == 0
    assert output["coflow_policy_predicted_z16_prefix_ratio"] == 1


def test_all_intermediate_z16_sources_and_stop_gradient_contracts() -> None:
    model = _tiny_model()
    target = torch.randn(2, 4, 6)
    prediction = torch.randn(2, 4, 6, requires_grad=True)
    valid = torch.tensor([True, False])

    model.intermediate_state_source = "ground_truth"
    source, mask, ratio = model._choose_intermediate_source(target, prediction, valid, global_step=5)
    torch.testing.assert_close(source[0], target[0])
    torch.testing.assert_close(source[1], prediction.detach()[1])
    assert mask.tolist() == [True, False] and ratio == 1

    model.intermediate_state_source = "predicted_detach"
    source, mask, ratio = model._choose_intermediate_source(target, prediction, valid, global_step=5)
    assert not source.requires_grad and not bool(mask.any()) and ratio == 0

    model.intermediate_state_source = "predicted_e2e"
    source, mask, ratio = model._choose_intermediate_source(target, prediction, valid, global_step=5)
    assert source is prediction and source.requires_grad and not bool(mask.any()) and ratio == 0

    model.intermediate_state_source = "scheduled"
    model.predicted_z16_detach = True
    assert model.teacher_ratio(0) == 1
    assert model.teacher_ratio(5) == pytest.approx(0.5)
    assert model.teacher_ratio(10) == 0
    early, early_mask, early_ratio = model._choose_intermediate_source(target, prediction, valid, global_step=0)
    torch.testing.assert_close(early[0], target[0])
    torch.testing.assert_close(early[1], prediction.detach()[1])
    assert early_mask.tolist() == [True, False] and early_ratio == 1
    late, late_mask, late_ratio = model._choose_intermediate_source(target, prediction, valid, global_step=10)
    torch.testing.assert_close(late, prediction.detach())
    assert not late.requires_grad and not bool(late_mask.any()) and late_ratio == 0


def test_inference_block2_consumes_the_predicted_block1_prefix_only() -> None:
    model = _tiny_model().eval()
    batch = 2
    common = {
        "context": torch.randn(batch, 5, 8),
        "context_valid": torch.ones(batch, 5, dtype=torch.bool),
        "z0": torch.randn(batch, 4, 6),
        "state": torch.randn(batch, 1, 2),
    }
    calls = []
    predicted_a1 = torch.full((batch, 16, 3), 3.0)
    predicted_z16 = torch.full((batch, 4, 6), 7.0)

    def fake_sample_block(self, **kwargs):
        calls.append(kwargs)
        if kwargs["prefix_action"] is None:
            torch.testing.assert_close(kwargs["world_start"], common["z0"])
            return predicted_a1, predicted_z16
        torch.testing.assert_close(kwargs["prefix_action"], predicted_a1)
        torch.testing.assert_close(kwargs["prefix_world"], predicted_z16)
        torch.testing.assert_close(kwargs["world_start"], predicted_z16)
        return torch.full((batch, 16, 3), 5.0), torch.full((batch, 4, 6), 9.0)

    model._sample_block = types.MethodType(fake_sample_block, model)
    actions, z16, z32 = model.sample_actions(**common)
    assert len(calls) == 2
    torch.testing.assert_close(actions[:, :16], predicted_a1)
    torch.testing.assert_close(z16, predicted_z16)
    assert bool((actions[:, 16:] == 5).all()) and bool((z32 == 9).all())
    assert "z16_target" not in inspect.signature(model.sample_actions).parameters
    assert "z32_target" not in inspect.signature(model.sample_actions).parameters


def test_joint_model_has_no_scalar_world_gate() -> None:
    model = _tiny_model()
    assert not any("gate" in name.lower() for name, _ in model.named_parameters())
    assert model.action_input_projection is not model.world_input_projection
    assert model.action_output_head is not model.world_output_head


def test_frozen_parameters_are_excluded_from_optimizer_groups() -> None:
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.teacher = nn.Linear(3, 3)
            self.student = nn.Linear(3, 3)
            for parameter in self.teacher.parameters():
                parameter.requires_grad_(False)

    model = Tiny()
    cfg = OmegaConf.create(
        {
            "trainer": {
                "learning_rate": {"base": 1.0e-4},
                "freeze_modules": "",
            }
        }
    )
    optimized = {id(parameter) for group in build_param_lr_groups(model, cfg) for parameter in group["params"]}
    assert all(id(parameter) not in optimized for parameter in model.teacher.parameters())
    assert all(id(parameter) in optimized for parameter in model.student.parameters())


def test_trainer_logs_policy_and_world_without_changing_legacy_metric_semantics() -> None:
    legacy = VLATrainer._build_native_loss_metrics(
        {"action_loss": torch.tensor(3.0)}, torch.tensor(3.0)
    )
    assert legacy == {
        "train/task": "action",
        "train/loss_total": 3.0,
        "train/loss_action": 3.0,
        "train/loss_action_raw": 3.0,
        "train/loss_action_weighted": 3.0,
    }
    coflow = VLATrainer._build_native_loss_metrics(
        {
            "action_loss": torch.tensor(2.1),
            "coflow_action_loss_raw": torch.tensor(2.0),
            "coflow_action_loss_weighted": torch.tensor(2.0),
            "coflow_world_loss_raw": torch.tensor(1.0),
            "coflow_world_loss_weighted": torch.tensor(0.1),
        },
        torch.tensor(2.1),
    )
    assert coflow["train/task"] == "action_world_coflow"
    assert coflow["train/loss_total"] == pytest.approx(2.1)
    assert coflow["train/loss_action"] == pytest.approx(2.0)
    assert coflow["train/loss_world"] == pytest.approx(1.0)
    assert coflow["train/loss_world_weighted"] == pytest.approx(0.1)


def test_new_yaml_is_opt_in_and_preserves_fastwam_zscore_contract() -> None:
    path = REPO_ROOT / "examples/Robotwin/train_files/robotwin_action_world_coflow.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["framework"]["name"] == "QwenActionWorldCoFlow"
    assert cfg["framework"]["enable_action_world_coflow"] is True
    assert cfg["framework"]["action_model"]["action_horizon"] == 32
    coflow = cfg["framework"]["action_world_coflow"]
    assert coflow["segment_boundaries"] == [16, 32]
    assert coflow["future_strides"] == [16, 32]
    assert coflow["freeze_qwen_vision_encoder"] is True
    assert coflow["world_feature_layer_strategy"] == "evenly_spaced"
    assert coflow["world_feature_num_layers"] == 4
    assert coflow["world_feature_layers"] is None
    assert coflow["world_layer_fusion_type"] == "fixed_mean"
    assert coflow["world_spatial_pool_type"] == "adaptive_avg_pool2d"
    assert coflow["world_token_grid"] == [4, 8]
    assert coflow["num_world_tokens"] == 32
    assert coflow["world_bridge_type"] == "qantara_brownian_bridge"
    assert coflow["world_prediction_type"] == "qantara_x_delta"
    assert coflow["intermediate_state_source"] == "scheduled"
    assert coflow["predicted_z16_detach"] is True
    ratios = coflow["noise_plane_sampling"]
    assert all(float(ratios[f"{name}_ratio"]) > 0 for name in ("policy", "forward", "inverse", "joint", "diagonal"))
    assert sum(float(value) for value in ratios.values()) == pytest.approx(1.0)
    assert coflow["action_inference_steps"] > 0 and coflow["world_inference_steps"] > 0
    assert coflow["enable_gradient_checkpointing"] is True
    assert coflow["log_attention_statistics"] is True
    assert cfg["datasets"]["vla_data"]["fastwam_coflow_future_strides"] == [16, 32]
    assert cfg["datasets"]["vla_data"]["data_mix"] == "robotwin_fastwam"
    assert cfg["datasets"]["vla_data"]["include_state"] is True
    assert cfg["datasets"]["vla_data"]["per_device_batch_size"] == 16
    assert cfg["trainer"]["gradient_accumulation_steps"] == 1
    assert cfg["framework"]["action_model"]["use_correlated_noise"] is False


def test_coflow_registration_and_defaults_are_isolated_from_legacy_qwengroot() -> None:
    assert FRAMEWORK_REGISTRY["QwenActionWorldCoFlow"] is QwenActionWorldCoFlow
    assert QwenActionWorldCoFlowDefaultConfig().enable_action_world_coflow is False
    assert not hasattr(QwenGR00TDefaultConfig(), "enable_action_world_coflow")
    legacy_path = REPO_ROOT / "examples/Robotwin/train_files/starvla_qwengroot_robotwin_fastwam_corrnoise.yaml"
    legacy = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))
    assert legacy["framework"]["name"] == "QwenGR00T"
    assert "enable_action_world_coflow" not in legacy["framework"]
    assert "action_world_coflow" not in legacy["framework"]
