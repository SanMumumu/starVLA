"""Small unit checks for WAM/FastWAM loss-contract plumbing."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T, QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.jointflow.dino_v3 import DINOv3Backbone, dino_num_patches, dino_patch_grid
from starVLA.model.framework.VLM4A.jointflow.visual_dino_flow_head import VisualFlowMatchingHead
from starVLA.model.framework.VLM4A.wam_guidance import linear_gradient_ramp
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import (
    BasicTransformerBlock,
    DiT,
)


class _ActionRecorder:
    def __init__(self):
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return args[1].sum() * 0.0


class _ActionHarness:
    _repeat_wam_batch = staticmethod(Qwen_GR00T._repeat_wam_batch)

    def __init__(self):
        self.config = SimpleNamespace(framework=SimpleNamespace(action_model={"repeated_diffusion_steps": 2}))
        self.action_model = _ActionRecorder()


def test_wam_action_loss_repeats_state_and_padding_like_baseline() -> None:
    harness = _ActionHarness()
    mem = torch.randn(3, 5, 8)
    actions = torch.randn(3, 32, 14)
    state = torch.randn(3, 1, 14)
    mask = torch.ones(3, 5, dtype=torch.bool)
    pad = torch.zeros(3, 32, dtype=torch.bool)
    pad[0, -2:] = True
    world = torch.randn(3, 7, 8)
    world_mask = torch.ones(3, 7, dtype=torch.bool)
    world_global = torch.randn(3, 8)

    Qwen_GR00T._wam_action_loss(
        harness,
        mem,
        actions,
        state,
        mask,
        pad,
        world_embs=world,
        world_attention_mask=world_mask,
        world_global=world_global,
        guidance_mode="dual_xattn",
    )
    call_mem, call_actions, call_state = harness.action_model.args
    assert call_mem.shape == (6, 5, 8)
    assert call_actions.shape == (6, 32, 14)
    assert call_state.shape == (6, 1, 14)
    assert harness.action_model.kwargs["action_is_pad"].shape == (6, 32)
    assert harness.action_model.kwargs["action_is_pad"][:3].equal(pad)
    assert harness.action_model.kwargs["action_is_pad"][3:].equal(pad)
    assert harness.action_model.kwargs["world_embs"].shape == (6, 7, 8)
    assert harness.action_model.kwargs["world_global"].shape == (6, 8)


class _VisualRecorder:
    def __init__(self):
        self.weights = None

    def __call__(self, _cond, target, weights=None):
        self.weights = weights
        return target.sum() * 0.0


class _VisualHarness:
    def __init__(self):
        self.wam_visual_head = _VisualRecorder()

    def _stack_jointflow_field(self, _examples, key, required=False):
        assert key == "future_valid" and not required
        return torch.tensor([1.0, 0.0])


def test_wam_visual_loss_masks_padded_future_without_scale_drift() -> None:
    harness = _VisualHarness()
    target = torch.randn(2, 480, 1024)
    Qwen_GR00T._wam_visual_loss(harness, torch.randn(2, 64, 8), target, [{}, {}])
    weights = harness.wam_visual_head.weights
    assert weights.shape == (2, 480)
    assert torch.all(weights[0] == 2.0)
    assert torch.all(weights[1] == 0.0)


class _ChangeVisualHarness:
    def __init__(self) -> None:
        self.wam_visual_head = _VisualRecorder()
        self.wam_fdm_delta = False
        self.config = SimpleNamespace(
            framework=SimpleNamespace(
                visual_model={
                    "patch_weighting": "change_balanced",
                    "change_weight_strength": 1.0,
                }
            )
        )
        self.z0 = torch.zeros(1, 4, 2)

    def _stack_jointflow_field(self, _examples, key, required=False):
        assert key == "future_valid" and not required
        return torch.ones(1)

    def _wam_dino_target(self, _examples, key, _online_keys):
        assert key == "dino_0"
        return self.z0


def test_wam_visual_loss_applies_balanced_change_patch_weights() -> None:
    harness = _ChangeVisualHarness()
    target = torch.tensor(
        [[[0.01, 0.01], [3.0, 3.0], [0.02, 0.02], [0.01, 0.01]]]
    )
    Qwen_GR00T._wam_visual_loss(harness, torch.randn(1, 2, 3), target, [{}])
    weights = harness.wam_visual_head.weights
    assert weights.shape == (1, 4)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    assert float(weights[0, 1]) > float(weights[0, 0])


def test_visual_jit_x_has_direct_clean_objective_and_finite_gradients() -> None:
    cfg = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {"vl_hidden_dim": 8},
                "visual_model": {
                    "d_dino": 4,
                    "hidden_size": 8,
                    "cross_attention_dim": 8,
                    "num_attention_heads": 2,
                    "attention_head_dim": 4,
                    "num_layers": 2,
                    "dropout": 0.0,
                    "add_pos_embed": True,
                    "max_target_tokens": 6,
                    "num_timestep_buckets": 32,
                    "num_inference_timesteps": 3,
                    "prediction_type": "jit_x",
                    "flow_time_sampling": "gr00t",
                    "jit_t_eps": 0.05,
                    "clean_target_loss_weight": 1.0,
                    "cosine_loss_weight": 0.1,
                },
            }
        }
    )
    torch.manual_seed(11)
    head = VisualFlowMatchingHead(cfg).train()
    cond = torch.randn(2, 3, 8, requires_grad=True)
    target = torch.randn(2, 6, 4)
    loss, details = head(cond, target, return_details=True)
    assert torch.isfinite(loss)
    assert set(details) == {"flow_loss_raw", "clean_loss_raw", "cosine_loss_raw"}
    assert all(torch.isfinite(value) for value in details.values())
    loss.backward()
    assert cond.grad is not None and bool(torch.isfinite(cond.grad).all())
    assert head.x_decode.weight.grad is not None
    assert float(head.x_decode.weight.grad.norm()) > 0.0

    head.eval()
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    pred_a = head.predict_latent(cond.detach(), n=6, generator=generator_a)
    pred_b = head.predict_latent(cond.detach(), n=6, generator=generator_b)
    torch.testing.assert_close(pred_a, pred_b)
    assert pred_a.shape == target.shape and bool(torch.isfinite(pred_a).all())


class _RectangularDinoBody(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device_anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = (images.shape[-2] // 16) * (images.shape[-1] // 16)
        return {"x_norm_patchtokens": images.new_zeros(images.shape[0], tokens, 8)}


class _RectangularDinoBackbone(DINOv3Backbone):
    def _load_body(self, trust_repo: bool) -> torch.nn.Module:
        del trust_repo
        return _RectangularDinoBody()


def test_dino_preserves_fastwam_composite_as_24_by_20_grid() -> None:
    assert dino_patch_grid([384, 320], 16) == (24, 20)
    assert dino_num_patches([384, 320], 16) == 480
    backbone = _RectangularDinoBackbone(image_size=[384, 320], patch_size=16, embed_dim=8)
    image = Image.fromarray(np.zeros((384, 320, 3), dtype=np.uint8), mode="RGB")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tensor = backbone.preprocess_batch([image])
    assert not any("NumPy array is not writable" in str(warning.message) for warning in caught)
    assert tensor.shape == (1, 3, 384, 320)
    assert backbone(tensor).shape == (1, 480, 8)


def test_frozen_dino_teacher_is_not_checkpointed() -> None:
    owner = torch.nn.Module()
    teacher = torch.nn.Linear(4, 4)
    Qwen_GR00T._set_dino_teacher(owner, teacher)
    assert not any(name.startswith("_dino_teacher") for name in owner.state_dict())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert not teacher.training


def test_action_dit_checkpointing_preserves_forward_and_gradients() -> None:
    torch.manual_seed(7)
    reference = DiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=2,
        dropout=0.0,
        final_dropout=False,
        interleave_self_attention=True,
        norm_type="ada_norm",
        cross_attention_dim=8,
        world_cross_attention=True,
        world_cross_attention_dim=8,
    ).train()
    gates = [block.world_gate for block in reference.transformer_blocks if block.world_gate is not None]
    assert gates and all(torch.count_nonzero(gate).item() == 0 for gate in gates)
    checkpointed = copy.deepcopy(reference).train()
    checkpointed.gradient_checkpointing = True

    hidden = torch.randn(2, 5, 8)
    memory = torch.randn(2, 6, 8)
    world = torch.randn(2, 4, 8)
    timestep = torch.tensor([2, 5])

    def run(model):
        x = hidden.clone().requires_grad_(True)
        mem = memory.clone().requires_grad_(True)
        wmem = world.clone().requires_grad_(True)
        output = model(
            x,
            mem,
            timestep=timestep,
            world_hidden_states=wmem,
            world_mode="dual",
        )
        output.square().mean().backward()
        return output.detach(), x.grad, mem.grad, wmem.grad

    expected = run(reference)
    actual = run(checkpointed)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-6)


class _PredictedWorldHead(torch.nn.Module):
    def predict_latent(self, cond: torch.Tensor, n: int) -> torch.Tensor:
        assert n == cond.shape[1]
        return 2.0 * cond


class _WorldGradientHarness:
    _jointflow_module_dtype = staticmethod(Qwen_GR00T._jointflow_module_dtype)

    def __init__(self) -> None:
        self.wam_guidance = {
            "signal": "z_pred",
            "mode": "dual_xattn",
            "bridge_source": "predicted",
            "detach_world": False,
        }
        self._wam_signal_is_latent = True
        self._wam_signal_is_oracle = False
        self._wam_world_n = 3
        self.wam_visual_head = _PredictedWorldHead()
        self.world_qformer = None
        self.world_adapter = torch.nn.Identity()


def test_joint_e2e_predicted_future_forward_is_unchanged_and_backward_is_ramped() -> None:
    assert [linear_gradient_ramp(step, 10, 30) for step in (0, 10, 20, 30, 100)] == [0.0, 0.0, 0.5, 1.0, 1.0]
    harness = _WorldGradientHarness()
    future = torch.randn(1, 3, 4, requires_grad=True)
    world = Qwen_GR00T._build_world_signal(
        harness,
        future,
        examples=[{}],
        action_world_grad_scale=0.25,
    )
    torch.testing.assert_close(world, 2.0 * future)
    world.sum().backward()
    # Predictor derivative (2) x ramp (0.25) = 0.5.  The forward value is not
    # interpolated or detached, so there is no oracle-to-predicted switch.
    torch.testing.assert_close(future.grad, torch.full_like(future, 0.5))


class _TrainablePredictedWorldHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.observed_training_modes = []

    def predict_latent(self, cond: torch.Tensor, n: int) -> torch.Tensor:
        self.observed_training_modes.append(bool(self.training))
        assert n <= cond.shape[1]
        conditioned = cond[:, :n] + cond.mean(dim=1, keepdim=True)
        return self.proj(conditioned)


class _DetachedWorldGradientHarness:
    _jointflow_module_dtype = staticmethod(Qwen_GR00T._jointflow_module_dtype)

    def __init__(self) -> None:
        self.wam_guidance = {
            "signal": "z_pred",
            "mode": "dual_xattn",
            "bridge_source": "predicted",
            "detach_world": True,
            "detached_prediction_eval_mode": True,
        }
        self._wam_signal_is_latent = True
        self._wam_signal_is_oracle = False
        self._wam_world_n = 3
        self.wam_visual_head = _TrainablePredictedWorldHead()
        self.wam_state_ctx = torch.nn.Linear(4, 4, bias=False)
        self.world_qformer = None
        self.world_adapter = torch.nn.Linear(4, 4, bias=False)

    def _wam_visual_condition(self, h_future, examples, task):
        assert examples and task == "policy"
        state_seed = torch.ones(h_future.shape[0], 1, 4, device=h_future.device)
        state_token = self.wam_state_ctx(state_seed)
        return torch.cat([h_future, state_token], dim=1)


def test_joint_detached_action_gradient_cannot_reach_world_predictor() -> None:
    torch.manual_seed(19)
    harness = _DetachedWorldGradientHarness()
    future = torch.randn(2, 3, 4, requires_grad=True)
    world_tokens = Qwen_GR00T._build_world_signal(harness, future, examples=[{}, {}])
    assert harness.wam_visual_head.observed_training_modes == [False]
    assert harness.wam_visual_head.training
    world_tokens.square().mean().backward()

    # Policy-side adapter learns to consume the detached prediction, while
    # neither predictor nor future-query representation receives action loss.
    assert harness.world_adapter.weight.grad is not None
    assert float(harness.world_adapter.weight.grad.norm()) > 0.0
    assert harness.wam_visual_head.proj.weight.grad is None
    assert harness.wam_state_ctx.weight.grad is None
    assert future.grad is None

    # The independent reconstruction objective on the same joint batch still
    # updates the world predictor and its future-query conditioning.
    future_world = torch.randn(2, 3, 4, requires_grad=True)
    world_condition = harness._wam_visual_condition(
        future_world, examples=[{}, {}], task="policy"
    )
    prediction = harness.wam_visual_head.predict_latent(world_condition, n=3)
    prediction.square().mean().backward()
    assert harness.wam_visual_head.proj.weight.grad is not None
    assert float(harness.wam_visual_head.proj.weight.grad.norm()) > 0.0
    assert harness.wam_state_ctx.weight.grad is not None
    assert float(harness.wam_state_ctx.weight.grad.norm()) > 0.0
    assert future_world.grad is not None and float(future_world.grad.norm()) > 0.0


class _JointE2EForwardHarness:
    _wam_action_world_grad_scale = Qwen_GR00T._wam_action_world_grad_scale
    _wam_world_training_condition = Qwen_GR00T._wam_world_training_condition
    _wam_uses_native_action_context = Qwen_GR00T._wam_uses_native_action_context
    _wam_auxiliary_rng_context = staticmethod(Qwen_GR00T._wam_auxiliary_rng_context)

    def __init__(self) -> None:
        self.wam_guidance = {
            "mode": "dual_xattn",
            "bridge_source": "predicted",
            "detach_world": False,
            "action_world_gradient_ramp": {
                "enabled": True,
                "start_step": 10,
                "end_step": 30,
                "start_scale": 0.0,
                "end_scale": 1.0,
            },
        }
        self.wam_dino_loss_weight = 0.01
        self.action_horizon = 2
        self.action_dim = 2
        self.wam_visual_head = torch.nn.Linear(4, 4, bias=False)
        self.action_model = torch.nn.Linear(4, 4, bias=False)
        self.backbone_calls = 0
        self.world_signal_scales = []
        self.action_world_tokens = None
        self.expected_action_world_bypass = False
        self.expected_action_backbone_detach = False
        self.expected_task = "joint_e2e"
        self.last_h_act = None
        self.last_h_future = None
        self.last_hidden = None
        self.native_hidden = None
        self.native_context_calls = 0

    def _native_qwen_action_context(self, examples, resize_to_training):
        assert len(examples) == 1 and not resize_to_training
        self.native_context_calls += 1
        self.native_hidden = torch.full((1, 4, 4), 1.5, requires_grad=True)
        return self.native_hidden, torch.ones(1, 4, dtype=torch.bool)

    @staticmethod
    def _native_action_loss_from_context(examples, last_hidden, backbone_attention_mask):
        assert len(examples) == 1
        assert torch.equal(backbone_attention_mask, torch.ones(1, 4, dtype=torch.bool))
        return last_hidden.square().mean()

    def _wam_guided_backbone(self, examples, task):
        assert len(examples) == 1 and task == self.expected_task
        self.backbone_calls += 1
        h_act = torch.ones(1, 2, 4, requires_grad=True)
        h_future = torch.full((1, 3, 4), 2.0, requires_grad=True)
        hidden = torch.ones(1, 5, 4, requires_grad=True)
        self.last_h_act = h_act
        self.last_h_future = h_future
        self.last_hidden = hidden
        attention = torch.ones(1, 5, dtype=torch.bool)
        placeholders = torch.zeros(1, 5, dtype=torch.bool)
        return h_act, h_future, hidden, attention, placeholders

    def _wam_world_target(self, examples):
        assert len(examples) == 1
        return torch.zeros(1, 3, 4)

    @staticmethod
    def _wam_visual_condition(h_future, examples, task):
        assert examples and task in {"joint_e2e", "joint_detached"}
        return h_future

    @staticmethod
    def _jointflow_module_dtype(module, fallback):
        return next(module.parameters()).dtype

    def _wam_visual_loss(self, cond, target, examples, return_details=False):
        assert cond.shape == target.shape == (1, 3, 4)
        loss = (cond - target).square().mean()
        if return_details:
            return loss, {
                "flow_loss_raw": loss.detach(),
                "clean_loss_raw": loss.detach() * 0.5,
                "cosine_loss_raw": loss.detach() * 0.25,
            }
        return loss

    def _build_world_signal(self, h_future, examples, action_world_grad_scale=1.0):
        self.world_signal_scales.append(float(action_world_grad_scale))
        return 3.0 * h_future

    def _assemble_guided_inputs(self, mode, h_act, h_future, hidden, attn, ph_mask, world_tokens):
        assert mode == "dual_xattn"
        if self.expected_action_backbone_detach:
            assert not h_act.requires_grad
            assert not h_future.requires_grad
            assert not hidden.requires_grad
        self.action_world_tokens = world_tokens
        return h_act, torch.ones(h_act.shape[:2], dtype=torch.bool), world_tokens, None, None

    def _stack_jointflow_field(self, examples, key, required):
        assert key == "action" and required
        return torch.ones(1, 2, 2)

    @staticmethod
    def _wam_action_state_and_mask(examples):
        return None, torch.zeros(1, 2, dtype=torch.bool)

    def _wam_action_loss(self, mem, actions, state, mem_mask, action_is_pad, **kwargs):
        assert state is None and kwargs["guidance_mode"] == "dual_xattn"
        if self.expected_action_world_bypass:
            assert kwargs["world_embs"] is None
            return mem.square().mean()
        assert kwargs["world_embs"] is not None
        return mem.square().mean() + kwargs["world_embs"].square().mean()

    def _wam_guided_unused_anchor(self, task, loss):
        assert task == self.expected_task
        return loss.new_zeros(())

    def _wam_action_world_bypass_anchor(self, task, loss):
        assert task == self.expected_task
        return loss.new_zeros(())

    @staticmethod
    def _wam_world_gate_metrics():
        return {
            "world_gate_openness": torch.tensor(0.25),
            "world_gate_signed_mean": torch.tensor(-0.125),
            "world_gate_max_openness": torch.tensor(0.5),
        }

    def _wam_world_to_action_metrics(self, ref):
        del ref
        return self._wam_world_gate_metrics()


def test_joint_e2e_one_backbone_forward_returns_action_and_world_losses() -> None:
    harness = _JointE2EForwardHarness()
    output = Qwen_GR00T._wam_guided_forward(harness, examples=[{}], task="joint_e2e", global_step=20)
    assert harness.backbone_calls == 1
    assert harness.world_signal_scales == [0.5]
    assert torch.equal(harness.action_world_tokens, torch.full((1, 3, 4), 6.0))
    assert set(key for key in output if key.endswith("_loss")) == {"action_loss", "world_loss"}
    torch.testing.assert_close(output["world_loss_raw"], torch.tensor(4.0))
    torch.testing.assert_close(output["world_loss"], torch.tensor(0.04))
    torch.testing.assert_close(output["action_world_grad_scale"], torch.tensor(0.5))
    torch.testing.assert_close(output["world_gate_openness"], torch.tensor(0.25))
    torch.testing.assert_close(output["world_gate_signed_mean"], torch.tensor(-0.125))
    torch.testing.assert_close(output["world_gate_max_openness"], torch.tensor(0.5))


def test_predictor_warmup_joint_batch_bypasses_world_only_for_action() -> None:
    """Legacy detached-action checkpoints retain their saved gradient ABI."""

    harness = _JointE2EForwardHarness()
    harness.wam_guidance["action_world_bypass"] = True
    harness.wam_guidance["detach_action_backbone"] = True
    harness.expected_action_world_bypass = True
    harness.expected_action_backbone_detach = True
    output = Qwen_GR00T._wam_guided_forward(
        harness,
        examples=[{}],
        task="joint_e2e",
        global_step=20,
    )
    assert harness.backbone_calls == 1
    assert harness.world_signal_scales == []
    assert harness.action_world_tokens is None
    assert set(key for key in output if key.endswith("_loss")) == {"action_loss", "world_loss"}
    torch.testing.assert_close(output["world_loss_raw"], torch.tensor(4.0))
    torch.testing.assert_close(output["action_world_bypassed"], torch.tensor(1.0))
    torch.testing.assert_close(output["action_backbone_detached"], torch.tensor(1.0))


def test_policy_first_warmup_routes_qwen_gradient_only_from_action() -> None:
    harness = _JointE2EForwardHarness()
    harness.wam_guidance.update(
        {
            "action_world_bypass": True,
            "detach_action_backbone": False,
            "detach_world_backbone": True,
            "detach_world": True,
        }
    )
    harness.expected_action_world_bypass = True
    harness.expected_task = "joint_detached"
    output = Qwen_GR00T._wam_guided_forward(
        harness,
        examples=[{}],
        task="joint_detached",
        global_step=20,
    )
    (output["action_loss"] + output["world_loss"]).backward()

    # Action memory is live, so the primary objective updates Qwen's action
    # representation. The auxiliary world condition was detached before its
    # loss and cannot update the future-query backbone tensor.
    assert harness.last_h_act.grad is not None
    assert float(harness.last_h_act.grad.abs().sum()) > 0.0
    assert harness.last_h_future.grad is None
    torch.testing.assert_close(output["action_backbone_detached"], torch.tensor(0.0))
    torch.testing.assert_close(output["world_backbone_detached"], torch.tensor(1.0))


def test_policy_first_world_loss_still_updates_visual_head() -> None:
    class Harness:
        wam_guidance = {"detach_world_backbone": True}

        @staticmethod
        def _wam_visual_condition(h_future, examples, task):
            assert examples and task == "joint_detached"
            return h_future

    harness = Harness()
    head = torch.nn.Linear(4, 4, bias=False)
    h_future = torch.randn(2, 3, 4, requires_grad=True)
    condition = Qwen_GR00T._wam_world_training_condition(
        harness,
        h_future,
        [{}],
        task="joint_detached",
    )
    head(condition).square().mean().backward()

    assert h_future.grad is None
    assert head.weight.grad is not None
    assert float(head.weight.grad.abs().sum()) > 0.0


def test_baseline_preserving_warmup_uses_only_native_context_for_action() -> None:
    harness = _JointE2EForwardHarness()
    harness.wam_guidance.update(
        {
            "action_world_bypass": True,
            "baseline_action_context": True,
            "detach_action_backbone": False,
            "detach_world_backbone": True,
            "detach_world": True,
        }
    )
    harness.expected_action_world_bypass = True
    harness.expected_task = "joint_detached"
    output = Qwen_GR00T._wam_guided_forward(
        harness,
        examples=[{}],
        task="joint_detached",
        global_step=20,
    )
    (output["action_loss"] + output["world_loss"]).backward()

    assert harness.native_context_calls == 1
    assert harness.native_hidden.grad is not None
    assert float(harness.native_hidden.grad.abs().sum()) > 0.0
    # The dual-query pass is auxiliary-only under v3.
    assert harness.last_h_act.grad is None
    assert harness.last_h_future.grad is None
    torch.testing.assert_close(output["baseline_action_context"], torch.tensor(1.0))


def test_baseline_preserving_auxiliary_rng_does_not_shift_policy_rng() -> None:
    reference = torch.zeros(())
    torch.manual_seed(1234)
    expected_first = torch.rand(())
    expected_second = torch.rand(())

    torch.manual_seed(1234)
    actual_first = torch.rand(())
    with Qwen_GR00T._wam_auxiliary_rng_context(reference):
        _ = torch.rand(128)
    actual_second = torch.rand(())

    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(actual_second, expected_second)


def test_zero_gated_world_modules_preserve_baseline_core_initialization() -> None:
    kwargs = {
        "dim": 16,
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "dropout": 0.1,
        "cross_attention_dim": 16,
        "norm_type": "ada_norm",
        "final_dropout": True,
    }
    torch.manual_seed(2028)
    baseline = BasicTransformerBlock(**kwargs, world_cross_attention=False)
    torch.manual_seed(2028)
    guided = BasicTransformerBlock(
        **kwargs,
        world_cross_attention=True,
        world_cross_attention_dim=16,
        world_gate_init=0.0,
    )

    guided_state = guided.state_dict()
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(guided_state[name], value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(guided.world_gate, torch.zeros_like(guided.world_gate))

    dit_kwargs = {
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "output_dim": 16,
        "num_layers": 4,
        "dropout": 0.1,
        "final_dropout": True,
        "interleave_self_attention": True,
        "norm_type": "ada_norm",
        "cross_attention_dim": 16,
    }
    torch.manual_seed(2029)
    baseline_dit = DiT(**dit_kwargs, world_cross_attention=False)
    torch.manual_seed(2029)
    guided_dit = DiT(
        **dit_kwargs,
        world_cross_attention=True,
        world_cross_attention_dim=16,
        world_gate_init=0.0,
    )
    guided_dit_state = guided_dit.state_dict()
    for name, value in baseline_dit.state_dict().items():
        torch.testing.assert_close(guided_dit_state[name], value, rtol=0.0, atol=0.0)


def test_native_baseline_forward_and_v3_warmup_share_action_helpers() -> None:
    class NativeForwardHarness:
        wam_enabled = False
        jointflow_enabled = False

        def __init__(self):
            self.context_calls = 0
            self.loss_calls = 0

        def _native_qwen_action_context(self, examples, resize_to_training):
            assert examples == [{"sample": 1}]
            assert not resize_to_training
            self.context_calls += 1
            return torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)

        def _native_action_loss_from_context(self, examples, hidden, mask):
            assert examples == [{"sample": 1}]
            assert hidden.shape == (1, 3, 4)
            assert mask.shape == (1, 3)
            self.loss_calls += 1
            return torch.tensor(0.75)

    harness = NativeForwardHarness()
    output = Qwen_GR00T.forward(harness, examples=[{"sample": 1}])
    torch.testing.assert_close(output["action_loss"], torch.tensor(0.75))
    assert harness.context_calls == 1
    assert harness.loss_calls == 1


def test_baseline_preserving_warmup_inference_skips_future_branch() -> None:
    class WarmupInferenceHarness:
        wam_guidance = {
            "mode": "dual_xattn",
            "baseline_action_context": True,
            "action_world_bypass": True,
        }
        _wam_uses_native_action_context = Qwen_GR00T._wam_uses_native_action_context

        def __init__(self):
            self.future_calls = 0

        @staticmethod
        def _native_qwen_action_context(examples, resize_to_training):
            assert len(examples) == 1 and resize_to_training
            return torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)

        @staticmethod
        def _native_predict_action_from_context(examples, hidden, mask):
            assert len(examples) == 1
            assert hidden.shape == (1, 3, 4) and mask.shape == (1, 3)
            return {"normalized_actions": np.zeros((1, 32, 14), dtype=np.float32)}

        def _wam_guided_backbone(self, examples, task):
            self.future_calls += 1
            raise AssertionError("warmup inference must not execute the future branch")

    harness = WarmupInferenceHarness()
    output = Qwen_GR00T._wam_guided_predict_action(harness, examples=[{}])
    assert output["normalized_actions"].shape == (1, 32, 14)
    assert harness.future_calls == 0


def test_world_gate_metrics_report_effective_tanh_openness() -> None:
    dit = DiT(
        num_attention_heads=2,
        attention_head_dim=4,
        output_dim=8,
        num_layers=4,
        dropout=0.0,
        final_dropout=False,
        interleave_self_attention=True,
        norm_type="ada_norm",
        cross_attention_dim=8,
        world_cross_attention=True,
        world_cross_attention_dim=8,
    )
    gates = [block.world_gate for block in dit.transformer_blocks if block.world_gate is not None]
    assert len(gates) == 2
    with torch.no_grad():
        gates[0].copy_(torch.atanh(torch.tensor([0.25])))
        gates[1].copy_(torch.atanh(torch.tensor([-0.75])))
    harness = SimpleNamespace(action_model=SimpleNamespace(model=dit))
    metrics = Qwen_GR00T._wam_world_gate_metrics(harness)
    torch.testing.assert_close(metrics["world_gate_openness"], torch.tensor(0.5))
    torch.testing.assert_close(metrics["world_gate_signed_mean"], torch.tensor(-0.25))
    torch.testing.assert_close(metrics["world_gate_max_openness"], torch.tensor(0.75))


def test_joint_e2e_trainer_metrics_expose_world_gate_to_wandb() -> None:
    from starVLA.training.train_starvla import VLATrainer

    output = {
        "action_loss": torch.tensor(0.002),
        "world_loss": torch.tensor(0.0003),
        "world_loss_raw": torch.tensor(0.03),
        "action_world_grad_scale": torch.tensor(0.4),
        "world_gate_openness": torch.tensor(0.2),
        "world_gate_signed_mean": torch.tensor(-0.05),
        "world_gate_max_openness": torch.tensor(0.7),
    }
    metrics = VLATrainer._build_loss_metrics(
        output,
        task="joint_e2e",
        total_loss=output["action_loss"] + output["world_loss"],
    )
    assert metrics["train/world_gate_openness"] == float(output["world_gate_openness"])
    assert metrics["train/world_gate_signed_mean"] == float(output["world_gate_signed_mean"])
    assert metrics["train/world_gate_max_openness"] == float(output["world_gate_max_openness"])


def test_two_stage_gate_metrics_expose_world_gate_to_wandb() -> None:
    from starVLA.training.train_starvla import VLATrainer

    for task, loss_key in (("policy", "action_loss"), ("passive", "passive_loss")):
        output = {
            loss_key: torch.tensor(0.002),
            "world_gate_openness": torch.tensor(0.2),
            "world_gate_signed_mean": torch.tensor(-0.05),
            "world_gate_max_openness": torch.tensor(0.7),
        }
        if task == "passive":
            output["passive_loss_raw"] = torch.tensor(0.02)
        metrics = VLATrainer._build_loss_metrics(
            output,
            task=task,
            total_loss=output[loss_key],
        )
        assert metrics["train/world_gate_openness"] == float(output["world_gate_openness"])
        assert metrics["train/world_gate_signed_mean"] == float(output["world_gate_signed_mean"])
        assert metrics["train/world_gate_max_openness"] == float(output["world_gate_max_openness"])


def test_dual_query_layout_is_causal_act_to_future_and_suffix_is_not_context() -> None:
    # [padding/shared-prefix | ACT ACT | FUTURE FUTURE | assistant suffix]
    act = torch.tensor(
        [
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    future = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    Qwen_GR00T._wam_validate_dual_query_layout(act, future, expected_act=2, expected_future=2)

    excluded = Qwen_GR00T._wam_query_suffix_exclusion_mask(act, future)
    expected = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(excluded, expected)

    reversed_future = future.clone()
    reversed_future[0] = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    try:
        Qwen_GR00T._wam_validate_dual_query_layout(act, reversed_future, expected_act=2, expected_future=2)
    except ValueError as exc:
        assert "ACT->FUTURE" in str(exc)
    else:
        raise AssertionError("A FUTURE-before-ACT prompt was not rejected")


def test_new_e2e_context_mask_closes_post_query_gate_bypass() -> None:
    harness = SimpleNamespace(
        wam_guidance={"mode": "dual_xattn", "include_context_in_world_memory": True}
    )
    h_act = torch.randn(1, 2, 4)
    h_future = torch.randn(1, 2, 4)
    hidden = torch.randn(1, 8, 4)
    attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    act = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    future = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 0]], dtype=torch.bool)
    excluded = Qwen_GR00T._wam_query_suffix_exclusion_mask(act, future)

    _, memory_mask = Qwen_GR00T._build_guided_action_memory(
        harness,
        h_act,
        h_future,
        hidden,
        attention,
        excluded,
        world_tokens=None,
    )
    # Explicit ACT queries remain; raw Qwen memory retains only the valid shared
    # prefix. ACT/FUTURE placeholders and the assistant suffix are all masked.
    expected = torch.tensor([[1, 1, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(memory_mask, expected)


def test_action_and_world_memory_context_can_be_decoupled() -> None:
    harness = SimpleNamespace(
        wam_guidance={
            "mode": "dual_xattn",
            "include_context_in_action_memory": True,
            "include_context_in_world_memory": False,
        }
    )
    h_act = torch.randn(1, 2, 4)
    h_future = torch.randn(1, 2, 4)
    hidden = torch.randn(1, 8, 4)
    world_tokens = torch.randn(1, 3, 4)
    attention = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    act = torch.tensor([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.bool)
    future = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 0]], dtype=torch.bool)
    excluded = Qwen_GR00T._wam_query_suffix_exclusion_mask(act, future)

    action_memory, action_mask = Qwen_GR00T._build_guided_action_memory(
        harness,
        h_act,
        h_future,
        hidden,
        attention,
        excluded,
        world_tokens,
    )
    world_memory, world_mask = Qwen_GR00T._build_world_memory(
        harness,
        world_tokens,
        hidden,
        attention,
        excluded,
    )

    # The normal action cross-attention still sees h_act + Qwen context.
    assert action_memory.shape == (1, 10, 4)
    torch.testing.assert_close(action_memory[:, :2], h_act)
    torch.testing.assert_close(action_memory[:, 2:], hidden)
    expected_action_mask = torch.tensor(
        [[1, 1, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool
    )
    assert torch.equal(action_mask, expected_action_mask)

    # The gated world cross-attention receives predicted-future tokens only.
    assert world_memory.shape == (1, 3, 4)
    torch.testing.assert_close(world_memory, world_tokens)
    assert torch.equal(world_mask, torch.ones(1, 3, dtype=torch.bool))


def test_yaml_task_weights_replace_framework_default_task_set() -> None:
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "QwenGR00T",
                "tasks": {"weights": {"joint_e2e": 1.0}},
            }
        }
    )
    merged = merge_framework_config(QwenGR00TDefaultConfig, cfg)
    assert OmegaConf.to_container(merged.framework.tasks.weights) == {"joint_e2e": 1.0}


def test_e2e_yaml_injects_zero_initialized_world_gates() -> None:
    cfg = OmegaConf.create(
        {
            "framework": {
                "wam": {
                    "guidance": {
                        "enabled": True,
                        "mode": "dual_xattn",
                        "gate_init": 0.0,
                    }
                },
                "action_model": {
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 8,
                    }
                },
            }
        }
    )
    harness = SimpleNamespace(config=cfg)
    Qwen_GR00T._inject_guidance_dit_flags(harness)
    dit_cfg = cfg.framework.action_model.diffusion_model_cfg
    assert bool(dit_cfg.world_cross_attention)
    assert float(dit_cfg.world_gate_init) == 0.0


def test_dual_branch_no_world2action_has_no_action_injection_modules() -> None:
    config_path = (
        REPO_ROOT
        / "examples/Robotwin/train_files/robotwin_wam_dual_branch_no_world2action.yaml"
    )
    cfg = OmegaConf.load(config_path)
    merged = merge_framework_config(QwenGR00TDefaultConfig, cfg)
    harness = SimpleNamespace(config=merged)
    Qwen_GR00T._validate_joint_e2e_contract(harness)
    Qwen_GR00T._validate_wam_two_stage_contract(harness)
    Qwen_GR00T._inject_guidance_dit_flags(harness)

    guidance = merged.framework.wam.guidance
    dit_cfg = merged.framework.action_model.diffusion_model_cfg
    assert not bool(guidance.world_to_action_enabled)
    assert bool(guidance.action_world_bypass)
    assert not bool(dit_cfg.get("world_cross_attention", False))
    assert not bool(dit_cfg.get("world_adaln", False))

    metric_harness = SimpleNamespace(
        wam_guidance={"world_to_action_enabled": False}
    )
    metrics = Qwen_GR00T._wam_world_to_action_metrics(
        metric_harness,
        torch.ones(()),
    )
    assert all(float(value) == 0.0 for value in metrics.values())


def test_no_world2action_runtime_contract_requires_physical_absence() -> None:
    from starVLA.training.train_starvla import VLATrainer

    class TwoBranchModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qwen_vl_interface = torch.nn.Linear(2, 2)
            self.action_model = torch.nn.Linear(2, 2)
            self.wam_visual_head = torch.nn.Linear(2, 2)
            self.wam_state_ctx = torch.nn.Linear(2, 2)

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = OmegaConf.load(
        REPO_ROOT
        / "examples/Robotwin/train_files/robotwin_wam_dual_branch_no_world2action.yaml"
    )
    trainer.model = TwoBranchModel()
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    VLATrainer._validate_wam_two_stage_runtime_contract(trainer)

    trainer.model.world_adapter = torch.nn.Linear(2, 2)
    try:
        VLATrainer._validate_wam_two_stage_runtime_contract(trainer)
    except RuntimeError as exc:
        assert "forbidden gate/adapter/attention" in str(exc)
    else:
        raise AssertionError("No-W2A runtime accepted a world adapter")


def test_robodojo_e2e_yaml_passes_early_joint_contract() -> None:
    config_paths = (
        REPO_ROOT / "examples/RoboDojo/train_files/starvla_qwengroot_robodojo_wam_e2e.yaml",
    )
    for config_path in config_paths:
        cfg = OmegaConf.load(config_path)
        merged = merge_framework_config(QwenGR00TDefaultConfig, cfg)
        harness = SimpleNamespace(config=merged)
        Qwen_GR00T._validate_joint_e2e_contract(harness)
        active = [key for key, value in merged.framework.tasks.weights.items() if float(value) > 0]
        assert active == ["joint_e2e"]
        assert str(cfg.framework.qwenvl.attn_implementation) == "flash_attention_2"
        assert bool(cfg.framework.wam.guidance.exclude_post_query_context)
        assert not bool(cfg.framework.action_model.use_correlated_noise)
        assert "corrnoise" not in str(cfg.run_id)
        assert not any(str(key).startswith("correlation_") for key in cfg.framework.action_model.keys())
        assert int(cfg.datasets.vla_data.per_device_batch_size) == 16
        assert int(cfg.trainer.gradient_accumulation_steps) == 1


def test_joint_detached_iid_yaml_has_strict_world_learning_contract() -> None:
    config_path = (
        REPO_ROOT
        / "examples/Robotwin/train_files/robotwin_wam_joint_detached_iid.yaml"
    )
    cfg = OmegaConf.load(config_path)
    merged = merge_framework_config(QwenGR00TDefaultConfig, cfg)
    Qwen_GR00T._validate_joint_e2e_contract(SimpleNamespace(config=merged))

    active = [key for key, value in cfg.framework.tasks.weights.items() if float(value) > 0]
    assert active == ["joint_detached"]
    assert bool(cfg.framework.wam.guidance.detach_world)
    assert bool(cfg.framework.wam.guidance.detached_prediction_eval_mode)
    assert str(cfg.framework.wam.guidance.bridge_source) == "predicted"
    assert bool(cfg.framework.wam.guidance.world_condition_on_state)
    assert not bool(cfg.framework.wam.guidance.action_world_gradient_ramp.enabled)
    assert not bool(cfg.framework.action_model.use_correlated_noise)
    assert not any(str(key).startswith("correlation_") for key in cfg.framework.action_model.keys())
    assert bool(cfg.datasets.vla_data.include_state)
    assert str(cfg.datasets.vla_data.fastwam_split) == "train"
    assert float(cfg.datasets.vla_data.fastwam_val_fraction) > 0.0
    assert str(cfg.framework.visual_model.prediction_type) == "jit_x"
    assert str(cfg.framework.visual_model.patch_weighting) == "change_balanced"
    assert float(cfg.framework.visual_model.clean_target_loss_weight) > 0.0
    assert float(cfg.framework.visual_model.cosine_loss_weight) > 0.0
    assert int(cfg.framework.visual_model.num_inference_timesteps) == int(
        cfg.trainer.world_validation.num_inference_timesteps
    )
    assert float(cfg.trainer.learning_rate.wam_visual_head) == 1.0e-4
    assert float(cfg.trainer.learning_rate.wam_state_ctx) == 1.0e-4
    assert bool(cfg.trainer.world_validation.enabled)
    assert int(cfg.trainer.world_validation.interval) == 5000
    assert int(cfg.trainer.gradient_accumulation_steps) == 1
    assert bool(cfg.trainer.log_grad_norms)


def test_world_validation_reports_sampled_mse_cosine_and_copy_baseline() -> None:
    from starVLA.training.train_starvla import VLATrainer

    class _WorldEvalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def evaluate_wam_world_prediction(self, examples, **kwargs):
            assert examples and kwargs["task"] == "joint_detached"
            device = self.anchor.device
            return {
                "squared_error_sum": torch.tensor(8.0, device=device),
                "element_count": torch.tensor(4.0, device=device),
                "cosine_sum": torch.tensor(3.0, device=device),
                "token_count": torch.tensor(4.0, device=device),
                "copy_squared_error_sum": torch.tensor(16.0, device=device),
                "valid_sample_count": torch.tensor(1.0, device=device),
            }

    model = _WorldEvalModel().train()
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = model
    trainer.vla_val_dataloader = [[{}], [{}]]
    trainer.completed_steps = 5000
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "world_validation": {
                    "task": "joint_detached",
                    "num_batches": 2,
                    "num_samples": 1,
                    "num_inference_timesteps": 3,
                    "seed": 2027,
                }
            }
        }
    )
    trainer.accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=False,
        unwrap_model=lambda wrapped: wrapped,
    )
    metrics = VLATrainer.eval_world_model(trainer)
    assert metrics == {
        "val/world_mse": 2.0,
        "val/world_cosine": 0.75,
        "val/world_copy_mse": 4.0,
        "val/world_mse_vs_copy_ratio": 0.5,
        "val/world_valid_samples": 2.0,
    }
    assert model.training


def test_joint_world_optimizer_contract_rejects_zero_lr() -> None:
    from starVLA.training.train_starvla import VLATrainer

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wam_visual_head = torch.nn.Linear(3, 3)

    model = _Model()
    trainer = VLATrainer.__new__(VLATrainer)
    trainer._jointflow_tasks = ["joint_detached"]
    trainer.model = model
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    trainer.optimizer = torch.optim.AdamW(
        [{"params": model.wam_visual_head.parameters(), "lr": 1.0e-4, "name": "wam_visual_head"}]
    )
    VLATrainer._validate_joint_world_optimizer_contract(trainer)
    trainer.optimizer.param_groups[0]["lr"] = 0.0
    try:
        VLATrainer._validate_joint_world_optimizer_contract(trainer)
    except RuntimeError as exc:
        assert "LR must be positive" in str(exc)
    else:
        raise AssertionError("A zero-LR world predictor was accepted")


def test_joint_world_optimizer_contract_accepts_scheduler_warmup_zero_lr() -> None:
    """Step-zero warmup LR is zero, while its scheduler base LR remains trainable."""

    from starVLA.training.train_starvla import VLATrainer

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wam_visual_head = torch.nn.Linear(3, 3)
            self.wam_state_ctx = torch.nn.Linear(3, 3)

    model = _Model()
    trainer = VLATrainer.__new__(VLATrainer)
    trainer._jointflow_tasks = ["joint_detached"]
    trainer.model = model
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    trainer.optimizer = torch.optim.AdamW(
        [
            {
                "params": model.wam_visual_head.parameters(),
                "lr": 1.0e-4,
                "name": "wam_visual_head",
            },
            {
                "params": model.wam_state_ctx.parameters(),
                "lr": 1.0e-4,
                "name": "wam_state_ctx",
            },
        ]
    )
    torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _step: 0.0)
    assert all(group["lr"] == 0.0 for group in trainer.optimizer.param_groups)
    assert all(group["initial_lr"] == 1.0e-4 for group in trainer.optimizer.param_groups)
    VLATrainer._validate_joint_world_optimizer_contract(trainer)


def test_robotwin_two_stage_budget_is_80k_warmup_plus_20k_gate() -> None:
    config_dir = REPO_ROOT / "examples/Robotwin/train_files"
    job_dir = REPO_ROOT / "\u6267\u884c\u811a\u672c/RBT"
    baseline = OmegaConf.load(config_dir / "starvla_qwengroot_robotwin_fastwam.yaml")
    output_root = (
        "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/"
        "starvla_wam_robotwin_baselinepreserve"
    )
    pairs = (
        ("robotwin_wam_warmup_rand.yaml", "robotwin_wam_gate_rand2clean.yaml"),
        ("robotwin_wam_warmup_clean.yaml", "robotwin_wam_gate_clean2clean.yaml"),
    )
    for warmup_name, gate_name in pairs:
        warmup = OmegaConf.load(config_dir / warmup_name)
        gate = OmegaConf.load(config_dir / gate_name)
        warmup_steps = int(warmup.trainer.max_train_steps)
        gate_steps = int(gate.trainer.max_train_steps)
        assert warmup_steps == 80000
        assert gate_steps == 20000
        assert warmup_steps > gate_steps
        assert warmup_steps + gate_steps == 100000
        assert bool(warmup.datasets.vla_data.include_state) == bool(
            gate.datasets.vla_data.include_state
        )
        assert str(warmup.run_root_dir) == output_root
        assert str(gate.run_root_dir) == output_root
        assert "corrnoise" not in str(warmup.run_id).lower()
        assert "corrnoise" not in str(gate.run_id).lower()
        for cfg in (warmup, gate):
            assert not bool(cfg.framework.action_model.use_correlated_noise)
            assert not any(
                str(key).startswith("correlation_")
                for key in cfg.framework.action_model.keys()
            )
            assert bool(cfg.framework.wam.guidance.world_condition_on_state)
            assert bool(cfg.framework.wam.guidance.detach_world)
            assert bool(cfg.framework.wam.guidance.detached_prediction_eval_mode)
            assert bool(cfg.framework.wam.guidance.include_context_in_world_memory)
            assert int(cfg.datasets.vla_data.per_device_batch_size) == 12
            assert int(cfg.datasets.vla_data.num_workers) == 4
            assert int(cfg.trainer.world_validation.num_workers) == 2
            assert int(cfg.trainer.gradient_accumulation_steps) == 1
            assert int(cfg.trainer.expected_global_batch_size) == 768
            assert int(cfg.trainer.world_validation.interval) == 5000
            merged = merge_framework_config(QwenGR00TDefaultConfig, cfg)
            harness = SimpleNamespace(config=merged)
            Qwen_GR00T._validate_joint_e2e_contract(harness)
            Qwen_GR00T._validate_wam_two_stage_contract(harness)

        warmup_active = [
            str(name)
            for name, weight in warmup.framework.tasks.weights.items()
            if float(weight) > 0.0
        ]
        gate_active = [
            str(name)
            for name, weight in gate.framework.tasks.weights.items()
            if float(weight) > 0.0
        ]
        assert warmup_active == ["joint_detached"]
        assert gate_active == ["policy"]
        assert str(warmup.trainer.wam_two_stage_phase) == "predictor_warmup"
        assert str(gate.trainer.wam_two_stage_phase) == "gate_ft"
        assert str(warmup.trainer.wam_two_stage_recipe) == "baseline_preserving_v3"
        assert str(gate.trainer.wam_two_stage_recipe) == "baseline_preserving_v3"
        assert bool(warmup.framework.wam.guidance.action_world_bypass)
        assert not bool(warmup.framework.wam.action_gradient_checkpointing)
        assert not bool(warmup.framework.wam.guidance.detach_action_backbone)
        assert bool(warmup.framework.wam.guidance.detach_world_backbone)
        assert bool(warmup.framework.wam.guidance.baseline_action_context)
        assert not bool(gate.framework.wam.guidance.action_world_bypass)
        assert not bool(gate.framework.wam.guidance.detach_action_backbone)
        assert bool(gate.framework.wam.guidance.detach_world_backbone)
        assert bool(gate.framework.wam.guidance.baseline_action_context)
        assert str(gate.trainer.pretrained_checkpoint) == (
            f"{output_root}/{warmup.run_id}/final_model/pytorch_model.pt"
        )
        assert bool(warmup.trainer.save_full_training_state)
        assert bool(warmup.trainer.seed_before_model_init)
        assert bool(baseline.trainer.seed_before_model_init)
        assert bool(warmup.trainer.action_eval_enabled)
        assert int(warmup.trainer.eval_interval) == int(baseline.trainer.eval_interval)
        assert int(warmup.trainer.num_warmup_steps) == 2000
        assert int(warmup.trainer.num_warmup_steps) == int(
            baseline.trainer.num_warmup_steps
        )
        assert OmegaConf.to_container(warmup.framework.qwenvl) == OmegaConf.to_container(
            baseline.framework.qwenvl
        )
        assert OmegaConf.to_container(warmup.framework.action_model) == OmegaConf.to_container(
            baseline.framework.action_model
        )
        for lr_name in ("base", "qwen_vl_interface", "action_model"):
            assert float(warmup.trainer.learning_rate[lr_name]) == float(
                baseline.trainer.learning_rate[lr_name]
            )
        assert str(warmup.trainer.lr_scheduler_type) == str(
            baseline.trainer.lr_scheduler_type
        )
        assert OmegaConf.to_container(
            warmup.trainer.scheduler_specific_kwargs
        ) == OmegaConf.to_container(baseline.trainer.scheduler_specific_kwargs)
        assert bool(gate.trainer.reset_world_gates_after_pretrained_load)
        assert bool(gate.trainer.save_full_training_state)
        assert int(gate.trainer.num_warmup_steps) == 500
        assert str(gate.trainer.lr_scheduler_type) == "constant_with_warmup"
        assert OmegaConf.to_container(gate.trainer.scheduler_specific_kwargs) == {}
        frozen = {
            item.strip()
            for item in str(gate.trainer.freeze_modules).split(",")
            if item.strip()
        }
        assert {
            "qwen_vl_interface",
            "wam_visual_head",
            "wam_state_ctx",
            "wam_act_ctx",
        }.issubset(frozen)
        assert float(gate.trainer.learning_rate.action_model) == 2.0e-5
        assert float(gate.trainer.learning_rate.world_adapter) == 1.0e-4
        assert bool(gate.framework.wam.guidance.log_gate_openness)

        # Existing 88%-era checkpoints have no recipe marker and retain the
        # original action-detached/world-live contract. They must remain
        # constructible for historical evaluation.
        legacy = OmegaConf.create(OmegaConf.to_container(warmup, resolve=True))
        del legacy.trainer.wam_two_stage_recipe
        legacy.trainer.num_warmup_steps = 4000
        legacy.framework.wam.guidance.detach_action_backbone = True
        legacy.framework.wam.guidance.detach_world_backbone = False
        legacy_merged = merge_framework_config(QwenGR00TDefaultConfig, legacy)
        Qwen_GR00T._validate_wam_two_stage_contract(
            SimpleNamespace(config=legacy_merged)
        )

        gate_job = OmegaConf.load(job_dir / gate_name)
        required = gate_job.REQUIRED
        assert int(required.WORKER_MIN_NUM) == 8
        assert int(required.WORKER_MAX_NUM) == 8
        assert int(required.GPU_PER_WORKER) == 8
        assert int(required.environment.EXPECTED_NUM_MACHINES) == 8
        assert int(required.environment.GPUS_PER_NODE) == 8
        required_raw = OmegaConf.to_container(required, resolve=False)
        assert isinstance(required_raw, dict)
        assert "EXPECTED_NUM_MACHINES=8" in str(required_raw["RUN_SCRIPTS"])

        inferred = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "run_aidi_rbtw.sh"),
                "--infer-topology",
                str(config_dir / gate_name),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert inferred.stdout.strip().splitlines()[-1] == "8"

    rand = OmegaConf.load(config_dir / "robotwin_wam_gate_rand2clean.yaml")
    clean = OmegaConf.load(config_dir / "robotwin_wam_gate_clean2clean.yaml")
    # The saved historical rand warmup config is state-conditioned; Stage 2
    # must not silently change that checkpoint ABI.
    assert bool(rand.datasets.vla_data.include_state)
    assert bool(clean.datasets.vla_data.include_state)


def test_wam_two_stage_contract_rejects_corrnoise_and_unfrozen_gate_ft() -> None:
    path = REPO_ROOT / "examples/Robotwin/train_files/robotwin_wam_gate_rand2clean.yaml"
    for mutation, expected in (
        ("corrnoise", "use_correlated_noise"),
        ("unfrozen", "freeze_modules"),
        ("world_loss", "only policy"),
    ):
        cfg = OmegaConf.load(path)
        if mutation == "corrnoise":
            cfg.framework.action_model.use_correlated_noise = True
        elif mutation == "unfrozen":
            cfg.trainer.freeze_modules = "wam_visual_head"
        else:
            cfg.framework.tasks.weights.passive = 1.0
        merged = merge_framework_config(QwenGR00TDefaultConfig, cfg)
        harness = SimpleNamespace(config=merged)
        try:
            Qwen_GR00T._validate_wam_two_stage_contract(harness)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"Invalid gate-FT mutation {mutation} was accepted")


def test_gate_ft_checkpoint_restore_preserves_parent_provenance() -> None:
    """Inference construction must not invalidate the saved Stage-2 contract."""

    from starVLA.model.framework import base_framework as base_framework_module

    gate_config = OmegaConf.to_container(
        OmegaConf.load(
            REPO_ROOT
            / "examples/Robotwin/train_files/robotwin_wam_gate_rand2clean.yaml"
        ),
        resolve=True,
    )
    expected_parent = gate_config["trainer"]["pretrained_checkpoint"]
    captured = {}

    def build_tiny_framework(cfg):
        captured["parent"] = cfg.trainer.pretrained_checkpoint
        captured["phase"] = cfg.trainer.wam_two_stage_phase
        return torch.nn.Linear(2, 2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = Path(tmp_dir) / "steps_20000_pytorch_model.pt"
        reference = torch.nn.Linear(2, 2)
        torch.save(reference.state_dict(), checkpoint)
        with patch.object(
            base_framework_module,
            "read_mode_config",
            return_value=(gate_config, {"new_embodiment": {}}),
        ), patch.object(
            base_framework_module,
            "build_framework",
            side_effect=build_tiny_framework,
        ):
            restored = base_framework_module.baseframework.from_pretrained(str(checkpoint))

    assert isinstance(restored, torch.nn.Linear)
    assert captured == {"parent": expected_parent, "phase": "gate_ft"}


def test_gate_reset_changes_only_world_gate_parameters() -> None:
    from starVLA.training.train_starvla import VLATrainer

    class _Block(torch.nn.Module):
        def __init__(self, gate_value: float) -> None:
            super().__init__()
            self.world_gate = torch.nn.Parameter(torch.tensor([gate_value]))
            self.weight = torch.nn.Parameter(torch.tensor([gate_value + 1.0]))

    model = torch.nn.Sequential(_Block(0.4), _Block(-0.8))
    non_gate_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.endswith("world_gate")
    }
    summary = VLATrainer._reset_world_gate_parameters(model, value=0.0)
    assert summary["count"] == 2
    assert summary["before_openness"] > 0.0
    assert summary["after_openness"] == 0.0
    for name, parameter in model.named_parameters():
        if name.endswith("world_gate"):
            assert torch.count_nonzero(parameter).item() == 0
        else:
            torch.testing.assert_close(parameter, non_gate_before[name])


def test_full_training_state_discovery_requires_commit_marker() -> None:
    from starVLA.training.train_starvla import VLATrainer

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_dir = Path(tmp_dir)

        incomplete = checkpoint_dir / "steps_20000_state"
        incomplete.mkdir()
        (incomplete / "trainer_state.json").write_text(
            json.dumps({"steps": 20000, "world_size": 48}),
            encoding="utf-8",
        )

        valid = checkpoint_dir / "steps_10000_state"
        valid.mkdir()
        (valid / "trainer_state.json").write_text(
            json.dumps(
                {
                    "steps": 10000,
                    "world_size": 48,
                    "gradient_accumulation_steps": 1,
                }
            ),
            encoding="utf-8",
        )
        (valid / "_SUCCESS").touch()

        state_path, step = VLATrainer._get_latest_full_state_checkpoint(checkpoint_dir)
        assert state_path == str(valid)
        assert step == 10000

        try:
            VLATrainer._validate_full_state_checkpoint(incomplete)
        except FileNotFoundError as exc:
            assert "_SUCCESS" in str(exc)
        else:
            raise AssertionError("A partial full-state checkpoint was accepted")

        mismatched = checkpoint_dir / "steps_30000_state"
        mismatched.mkdir()
        (mismatched / "trainer_state.json").write_text(
            json.dumps({"steps": 29999}), encoding="utf-8"
        )
        (mismatched / "_SUCCESS").touch()
        try:
            VLATrainer._validate_full_state_checkpoint(mismatched)
        except ValueError as exc:
            assert "step mismatch" in str(exc)
        else:
            raise AssertionError("A mislabeled full-state checkpoint was accepted")


def test_accelerator_state_roundtrips_registered_raw_scheduler() -> None:
    from accelerate import Accelerator

    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.1, 1.0 - 0.1 * step),
    )
    accelerator = Accelerator(cpu=True)
    model, optimizer = accelerator.prepare(model, optimizer)
    # This mirrors VLATrainer.prepare_training: the raw scheduler retains the
    # established manual step semantics but participates in save/load state.
    accelerator.register_for_checkpointing(scheduler)

    loss = model(torch.ones(1, 2)).square().mean()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    saved_weight = accelerator.unwrap_model(model).weight.detach().clone()
    saved_epoch = scheduler.last_epoch
    saved_lr = scheduler.get_last_lr()

    with tempfile.TemporaryDirectory() as tmp_dir:
        accelerator.save_state(tmp_dir)
        assert (Path(tmp_dir) / "custom_checkpoint_0.pkl").is_file()

        with torch.no_grad():
            accelerator.unwrap_model(model).weight.add_(10.0)
        scheduler.step()
        assert scheduler.last_epoch != saved_epoch

        accelerator.load_state(tmp_dir)
        torch.testing.assert_close(
            accelerator.unwrap_model(model).weight.detach(), saved_weight
        )
        assert scheduler.last_epoch == saved_epoch
        assert scheduler.get_last_lr() == saved_lr


def test_joint_e2e_contract_rejects_gate_bypass_and_corrnoise() -> None:
    config_path = REPO_ROOT / "examples/RoboDojo/train_files/starvla_qwengroot_robodojo_wam_e2e.yaml"
    for field, expected_message in (
        ("suffix", "exclude_post_query_context"),
        ("corrnoise", "use_correlated_noise"),
    ):
        cfg = OmegaConf.load(config_path)
        if field == "suffix":
            cfg.framework.wam.guidance.exclude_post_query_context = False
        else:
            cfg.framework.action_model.use_correlated_noise = True
        merged = merge_framework_config(QwenGR00TDefaultConfig, cfg)
        try:
            Qwen_GR00T._validate_joint_e2e_contract(SimpleNamespace(config=merged))
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Invalid E2E {field} contract was not rejected")


def test_trainer_preserves_two_microbatch_gradient_accumulation() -> None:
    """The first micro-batch gradient must survive until the sync step."""

    from accelerate import Accelerator
    from accelerate.utils import GradientAccumulationPlugin

    from starVLA.training.train_starvla import VLATrainer

    class _ToyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, target):
            return {"action_loss": (self.weight - target).square()}

    class _Scheduler:
        def __init__(self) -> None:
            self.steps = 0

        def step(self) -> None:
            self.steps += 1

    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=2,
            # ZeRO-2 rejects DeepSpeedEngine.no_sync(); synchronizing each
            # micro-batch preserves accumulation while avoiding that path.
            sync_each_batch=True,
        ),
    )
    assert accelerator.gradient_state.plugin_kwargs["sync_each_batch"] is True
    model = _ToyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "gradient_clipping": None,
                "logging_frequency": 100,
            }
        }
    )
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = _Scheduler()
    trainer.accelerator = accelerator
    trainer.completed_steps = 0
    trainer._jointflow_tasks = None
    trainer._jointflow_weights = None
    trainer._jointflow_cur_task = None
    trainer._jointflow_resample = True
    trainer._log_grad_norms = False
    trainer._grad_groups = None
    trainer._using_deepspeed = False
    trainer._last_clip_norm = None
    trainer._grad_warned = False

    trainer._train_step(torch.tensor(1.0))
    assert accelerator.sync_gradients is False
    torch.testing.assert_close(accelerator.unwrap_model(model).weight.detach(), torch.tensor(0.0))

    trainer._train_step(torch.tensor(3.0))
    assert accelerator.sync_gradients is True
    # Gradients are (-2 + -6) / accumulation(2) = -4, hence w = 0.4.
    torch.testing.assert_close(accelerator.unwrap_model(model).weight.detach(), torch.tensor(0.4))
    assert trainer.lr_scheduler.steps == 1


def test_deepspeed_runtime_batch_contract_matches_yaml() -> None:
    from starVLA.training.train_starvla import VLATrainer

    class _Engine:
        @staticmethod
        def gradient_accumulation_steps():
            return 2

        @staticmethod
        def train_micro_batch_size_per_gpu():
            return 8

        @staticmethod
        def train_batch_size():
            return 768

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "gradient_accumulation_steps": 2,
                "expected_global_batch_size": 768,
            },
            "datasets": {"vla_data": {"per_device_batch_size": 8}},
        }
    )
    trainer.accelerator = SimpleNamespace(
        gradient_accumulation_steps=2,
        num_processes=48,
        is_main_process=False,
    )
    trainer._using_deepspeed = True
    trainer.model = _Engine()
    VLATrainer._validate_runtime_batch_contract(trainer)

    trainer.accelerator.num_processes = 32
    try:
        VLATrainer._validate_runtime_batch_contract(trainer)
    except RuntimeError as exc:
        assert "expected_global_batch_size" in str(exc)
    else:
        raise AssertionError("An undersized distributed world was accepted")
    trainer.accelerator.num_processes = 48

    trainer.model.train_batch_size = lambda: 384
    try:
        VLATrainer._validate_runtime_batch_contract(trainer)
    except RuntimeError as exc:
        assert "train_batch_size differs" in str(exc)
    else:
        raise AssertionError("DeepSpeed global-batch mismatch was not rejected")


def test_world_validation_loader_cannot_define_deepspeed_train_micro_batch() -> None:
    from starVLA.training.train_starvla import VLATrainer

    train_loader = SimpleNamespace(batch_size=12)
    val_loader = SimpleNamespace(batch_size=2)
    model = object()
    optimizer = object()
    calls: list[tuple[str, tuple[object, ...]]] = []

    class _Accelerator:
        def prepare(self, *components):
            calls.append(("prepare", components))
            return components

        def prepare_data_loader(self, loader):
            calls.append(("prepare_data_loader", (loader,)))
            return ("prepared_validation", loader)

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.accelerator = _Accelerator()
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.vla_train_dataloader = train_loader
    trainer.vla_val_dataloader = val_loader

    VLATrainer._prepare_distributed_components(trainer)

    assert calls == [
        ("prepare", (model, optimizer, train_loader)),
        ("prepare_data_loader", (val_loader,)),
    ]
    assert trainer.vla_train_dataloader is train_loader
    assert trainer.vla_val_dataloader == ("prepared_validation", val_loader)


if __name__ == "__main__":
    test_wam_action_loss_repeats_state_and_padding_like_baseline()
    test_wam_visual_loss_masks_padded_future_without_scale_drift()
    test_wam_visual_loss_applies_balanced_change_patch_weights()
    test_visual_jit_x_has_direct_clean_objective_and_finite_gradients()
    test_dino_preserves_fastwam_composite_as_24_by_20_grid()
    test_frozen_dino_teacher_is_not_checkpointed()
    test_action_dit_checkpointing_preserves_forward_and_gradients()
    test_joint_e2e_predicted_future_forward_is_unchanged_and_backward_is_ramped()
    test_joint_detached_action_gradient_cannot_reach_world_predictor()
    test_joint_e2e_one_backbone_forward_returns_action_and_world_losses()
    test_predictor_warmup_joint_batch_bypasses_world_only_for_action()
    test_world_gate_metrics_report_effective_tanh_openness()
    test_joint_e2e_trainer_metrics_expose_world_gate_to_wandb()
    test_two_stage_gate_metrics_expose_world_gate_to_wandb()
    test_dual_query_layout_is_causal_act_to_future_and_suffix_is_not_context()
    test_new_e2e_context_mask_closes_post_query_gate_bypass()
    test_action_and_world_memory_context_can_be_decoupled()
    test_yaml_task_weights_replace_framework_default_task_set()
    test_e2e_yaml_injects_zero_initialized_world_gates()
    test_dual_branch_no_world2action_has_no_action_injection_modules()
    test_no_world2action_runtime_contract_requires_physical_absence()
    test_robodojo_e2e_yaml_passes_early_joint_contract()
    test_joint_detached_iid_yaml_has_strict_world_learning_contract()
    test_world_validation_reports_sampled_mse_cosine_and_copy_baseline()
    test_joint_world_optimizer_contract_rejects_zero_lr()
    test_joint_world_optimizer_contract_accepts_scheduler_warmup_zero_lr()
    test_robotwin_two_stage_budget_is_80k_warmup_plus_20k_gate()
    test_wam_two_stage_contract_rejects_corrnoise_and_unfrozen_gate_ft()
    test_gate_ft_checkpoint_restore_preserves_parent_provenance()
    test_gate_reset_changes_only_world_gate_parameters()
    test_full_training_state_discovery_requires_commit_marker()
    test_accelerator_state_roundtrips_registered_raw_scheduler()
    test_joint_e2e_contract_rejects_gate_bypass_and_corrnoise()
    test_trainer_preserves_two_microbatch_gradient_accumulation()
    test_deepspeed_runtime_batch_contract_matches_yaml()
    test_world_validation_loader_cannot_define_deepspeed_train_micro_batch()
