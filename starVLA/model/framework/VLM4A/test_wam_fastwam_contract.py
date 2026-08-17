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
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T, QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.QwenWorldActionMoT import QwenWorldActionMoT
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

    square = backbone.preprocess_batch([image], image_size=[256, 256])
    assert square.shape == (1, 3, 256, 256)
    assert backbone(square).shape == (1, 256, 8)


def test_mot_current_dino_encodes_full_480_token_clean_prefix_once() -> None:
    class Harness:
        _pool_dino = QwenWorldActionMoT._pool_dino
        _finalize_dino = QwenWorldActionMoT._finalize_dino
        _encode_current_dino = QwenWorldActionMoT._encode_current_dino

        def __init__(self) -> None:
            self.dino_pool = 2
            self.current_dino_pool = 1
            self.num_current_world_views = 1
            self._dino_spec = {
                "image_size": [384, 320],
                "patch_size": 16,
            }
            self._dino_mean = torch.zeros(8)
            self._dino_std = torch.ones(8)
            self.calls = 0

        def _encode_dino_raw(self, batch_views, *, expected_views=None):
            assert expected_views == 1
            self.calls += 1
            batch = len(batch_views)
            raw = torch.arange(
                batch * 480 * 8,
                dtype=torch.float32,
            ).reshape(batch, 480, 8)
            return raw, 1

    harness = Harness()
    current = harness._encode_current_dino([[object()], [object()]])
    assert harness.calls == 1
    assert current.shape == (2, 480, 8)
    torch.testing.assert_close(
        current[0],
        torch.arange(480 * 8, dtype=torch.float32).reshape(480, 8),
    )


def test_mot_uses_two_current_views_and_one_future_view() -> None:
    class Harness:
        _pool_dino = QwenWorldActionMoT._pool_dino
        _finalize_dino = QwenWorldActionMoT._finalize_dino
        _encode_dino = QwenWorldActionMoT._encode_dino
        _encode_current_dino = QwenWorldActionMoT._encode_current_dino

        def __init__(self) -> None:
            self.dino_pool = 1
            self.current_dino_pool = None
            self.num_world_views = 1
            self.num_current_world_views = 2
            self.future_dino_image_size = (224, 224)
            self._dino_spec = {
                "image_size": (224, 224),
                "patch_size": 16,
            }
            self._dino_mean = torch.zeros(8)
            self._dino_std = torch.ones(8)

        def _encode_dino_raw(
            self,
            batch_views,
            *,
            image_size=None,
            expected_views=None,
        ):
            size = self._dino_spec["image_size"] if image_size is None else image_size
            rows, columns = dino_patch_grid(size, self._dino_spec["patch_size"])
            views = len(batch_views[0])
            assert views == expected_views
            return (
                torch.zeros(len(batch_views) * views, rows * columns, 8),
                views,
            )

    harness = Harness()
    current = harness._encode_current_dino(
        [[object(), object()], [object(), object()]]
    )
    future = harness._encode_dino([[object()], [object()]])

    assert current.shape == (2, 392, 8)
    assert future.shape == (2, 196, 8)


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
    _wam_action_loss_from_guided_context = Qwen_GR00T._wam_action_loss_from_guided_context
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
        self.detach_action_query_calls = []
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

    def _wam_guided_backbone(self, examples, task, *, detach_action_query=False):
        assert len(examples) == 1 and task == self.expected_task
        self.backbone_calls += 1
        self.detach_action_query_calls.append(bool(detach_action_query))
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


def test_causal_query_warmup_uses_one_causal_backbone_pass() -> None:
    harness = _JointE2EForwardHarness()
    harness.expected_task = "joint_detached"
    harness.expected_action_world_bypass = True
    harness.wam_guidance.update(
        {
            "detach_world": True,
            "action_world_bypass": True,
            "detach_action_backbone": False,
            "detach_world_backbone": False,
            "future_query_through_qwen": True,
            "separate_world_backbone_pass": False,
            "detach_action_query_in_world_pass": False,
        }
    )
    output = Qwen_GR00T._wam_guided_forward(
        harness,
        examples=[{}],
        task="joint_detached",
        global_step=20,
    )
    assert harness.backbone_calls == 1
    assert harness.native_context_calls == 0
    assert harness.detach_action_query_calls == [False]
    assert harness.world_signal_scales == []
    torch.testing.assert_close(output["world_loss_raw"], torch.tensor(4.0))
    torch.testing.assert_close(output["action_world_grad_scale"], torch.tensor(0.0))
    assert harness.action_world_tokens is None
    assert set(key for key in output if key.endswith("_loss")) == {"action_loss", "world_loss"}
    torch.testing.assert_close(output["action_world_bypassed"], torch.tensor(1.0))
    torch.testing.assert_close(output["action_backbone_detached"], torch.tensor(0.0))
    torch.testing.assert_close(output["world_backbone_detached"], torch.tensor(0.0))
    torch.testing.assert_close(output["baseline_action_context"], torch.tensor(0.0))


def test_causal_query_gate_policy_uses_act_hidden_not_native_context() -> None:
    """Gate FT and inference-facing policy loss must keep ACT on the action path."""

    harness = _JointE2EForwardHarness()
    harness.expected_task = "policy"
    harness.wam_guidance.update(
        {
            "baseline_action_context": False,
            "future_query_through_qwen": True,
            "separate_world_backbone_pass": False,
            "detach_action_query_in_world_pass": False,
        }
    )
    harness._wam_maybe_logged_gate_metrics = lambda: {}
    output = Qwen_GR00T._wam_guided_forward(
        harness,
        examples=[{}],
        task="policy",
    )
    assert harness.backbone_calls == 1
    assert harness.native_context_calls == 0
    assert harness.action_world_tokens is not None
    torch.testing.assert_close(output["baseline_action_context"], torch.tensor(0.0))


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


def test_robodojo_rynnbrain_uses_maintained_h25_configs() -> None:
    train_root = REPO_ROOT / "examples/RoboDojo/train_files"
    train_dir = train_root / "released_rynn50k"
    contracts = {
        "rynn_base_h25_50k.yaml": (False, False, True, False, False),
        "rynn_base_history_h25_mem_50k.yaml": (False, True, False, False, True),
        "rynn_base_text_h25_mem_50k.yaml": (True, True, False, True, False),
        "rynn_base_text_h25_mem_bf16_50k.yaml": (True, True, False, False, False),
    }
    configs = {name: OmegaConf.load(train_dir / name) for name in contracts}

    for name, cfg in configs.items():
        (
            planner_text_enabled,
            data_text_enabled,
            action_eval_enabled,
            fp32_shell,
            mem_encoder,
        ) = contracts[name]
        assert str(cfg.framework.name) == "QwenWorldActionMoT"
        assert bool(cfg.framework.enable_world_action_mot)
        assert "wam" not in cfg.framework
        assert "jointflow" not in cfg.framework
        assert "wam_two_stage_phase" not in cfg.trainer
        assert "wam_two_stage_recipe" not in cfg.trainer
        assert cfg.trainer.pretrained_checkpoint is None
        assert not bool(cfg.trainer.is_resume)
        assert not bool(cfg.trainer.world_validation.enabled)
        assert int(cfg.trainer.max_train_steps) == 50000
        assert int(cfg.trainer.expected_global_batch_size) == 768
        assert int(cfg.trainer.gradient_accumulation_steps) == 1
        assert int(cfg.datasets.vla_data.per_device_batch_size) == 12
        assert bool(cfg.datasets.vla_data.include_state)
        assert int(cfg.framework.action_model.state_dim) == 14
        assert int(cfg.framework.action_model.action_horizon) == 25
        assert int(cfg.framework.planner.num_action_queries) == 25
        assert int(cfg.framework.planner.num_world_queries) == 16
        assert str(cfg.framework.world_action_mot.architecture) == "causal_dino_mot"
        assert str(cfg.framework.world_action_mot.interaction_mode) == "base"
        assert int(cfg.framework.world_action_mot.world_hidden_size) == 512
        assert int(cfg.framework.world_action_mot.world_ffn_dim) == 2048
        assert int(cfg.framework.world_action_mot.action_hidden_size) == 1024
        assert int(cfg.framework.world_action_mot.action_ffn_dim) == 4096
        assert int(cfg.framework.world_action_mot.num_layers) == 30
        assert int(cfg.framework.world_action_mot.num_attention_heads) == 24
        assert int(cfg.framework.world_action_mot.attention_head_dim) == 128
        assert not bool(cfg.framework.world_action_mot.layerwise_planner_coupling)
        assert "world_condition_on_state" not in cfg.framework.world_action_mot
        assert int(cfg.framework.qwenvl.truncate_vlm_layers) == 0
        assert int(cfg.framework.world_action_mot.world_num_train_timesteps) == 1000
        assert int(cfg.framework.world_action_mot.action_num_train_timesteps) == 1000
        assert str(cfg.framework.world_action_mot.action_prediction_type) == "velocity"
        assert str(cfg.framework.world_action_mot.action_velocity_target) == "noise_minus_clean"
        assert float(cfg.framework.world_action_mot.jit_t_eps) == 0.05
        assert int(cfg.framework.world_action_mot.repeated_diffusion_steps) == 1
        assert float(cfg.framework.world_action_mot.action_loss_weight) == 1.0
        assert float(cfg.framework.world_action_mot.world_loss_weight) == 1.0
        assert (
            bool(cfg.framework.planner.text_supervision.enabled)
            is planner_text_enabled
        )
        assert float(cfg.framework.world_action_mot.text_loss_weight) == (
            0.005 if planner_text_enabled else 0.0
        )
        assert bool(cfg.trainer.seed_before_model_init)
        assert bool(cfg.trainer.action_eval_enabled) is action_eval_enabled
        assert bool(cfg.framework.dino.force_online)
        assert str(cfg.framework.dino.weights).endswith("/DINO-B/")
        assert cfg.framework.dino.current_dino_pool is None
        assert cfg.framework.dino.get("action_dino_pool", None) is None
        assert (
            bool(cfg.datasets.vla_data.text_annotations.enabled)
            is data_text_enabled
        )
        assert str(
            cfg.framework.world_action_mot.get(
                "action_precision_mode",
                "inherit",
            )
        ) == ("fp32_shell" if fp32_shell else "inherit")
        assert ("mem_vision_encoder" in cfg.framework.qwenvl) is mem_encoder

    for rynn in configs.values():
        assert str(rynn.framework.qwenvl.attn_implementation) == "sdpa"
        assert bool(rynn.framework.qwenvl.require_attn_implementation)
        assert str(rynn.framework.qwenvl.base_vlm).rstrip("/").endswith(
            "/rynnbrain1.1-2B"
        )
        assert not bool(rynn.framework.qwenvl.enable_thinking)
        assert str(rynn.run_id).lower().startswith("starvla_rynnbrain11_")

    removed = (
        "rynn_base_L.yaml",
        "rynn_base_L_text.yaml",
        "rynn_base_L_robotwin.yaml",
        "rynn_base_L_resume.yaml",
        "rynn_joint.yaml",
        "rynn_joint_text.yaml",
        "mot_base.yaml",
        "mot_base_text.yaml",
        "mot_joint.yaml",
        "mot_joint_text.yaml",
        "rynn_action_only_gen.yaml",
        "starvla_qwen3_robodojo_world_action_mot.yaml",
        "starvla_rynnbrain11_robodojo_world_action_mot.yaml",
        "starvla_qwengroot_robodojo_wam_warmup.yaml",
        "starvla_qwengroot_robodojo_wam_gate.yaml",
        "starvla_rynnbrain11_robodojo_causal_query_warmup.yaml",
        "starvla_rynnbrain11_robodojo_causal_query_gate.yaml",
    )
    assert all(not (train_root / name).exists() for name in removed)


def test_robodojo_mot_ignores_state_when_the_checkpoint_is_query_only() -> None:
    stateless = SimpleNamespace(include_state=False)
    assert (
        QwenWorldActionMoT._stack_state(
            stateless,
            [{"state": np.arange(14, dtype=np.float32)[None]}],
            torch.float32,
        )
        is None
    )


def test_legacy_robodojo_mot_stacks_current_state_as_one_normalized_token() -> None:
    harness = SimpleNamespace(
        include_state=True,
        device=torch.device("cpu"),
        config=SimpleNamespace(
            framework=SimpleNamespace(
                action_model=SimpleNamespace(state_dim=14)
            )
        ),
    )
    examples = [
        {"state": np.arange(14, dtype=np.float32)[None]},
        {"state": (np.arange(14, dtype=np.float32) + 10)[None]},
    ]
    state = QwenWorldActionMoT._stack_state(harness, examples, torch.float32)
    assert state.shape == (2, 1, 14)
    assert torch.equal(state[0, 0], torch.arange(14, dtype=torch.float32))
    assert torch.equal(state[1, 0], torch.arange(14, dtype=torch.float32) + 10)


def test_robodojo_text_target_rejects_missing_annotations() -> None:
    harness = SimpleNamespace(
        wam_text_supervision={
            "enabled": True,
            "subtask_field": "subtask_text",
            "completed_subtask_field": "completed_subtask_text",
            "prompt_template": "Task: {instruction}",
            "response_template": (
                "Current subtask: {subtask_text}\n"
                "Completed subtask: {completed_subtask_text}"
            ),
        }
    )
    with pytest.raises(ValueError, match="non-empty annotations"):
        Qwen_GR00T._wam_text_target(
            harness,
            {
                "lang": "Build a tower.",
                "subtask_text": "",
                "completed_subtask_text": "",
            },
        )
    prompt, answer = Qwen_GR00T._wam_text_target(
        harness,
        {
            "lang": "Build a tower.",
            "subtask_text": "Place the lower board.",
            "completed_subtask_text": "None.",
        },
    )
    assert prompt == "Task: Build a tower."
    assert answer == (
        "Current subtask: Place the lower board.\nCompleted subtask: None."
    )


def test_assistant_only_text_labels_exclude_prompt_and_left_padding() -> None:
    input_ids = torch.tensor(
        [
            [0, 0, 10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24, 25, 26],
        ]
    )
    attention = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )
    labels = Qwen_GR00T._assistant_only_labels(
        input_ids,
        attention,
        torch.tensor([3, 4]),
    )
    assert labels.tolist() == [
        [-100, -100, -100, -100, -100, 13, 14],
        [-100, -100, -100, -100, 24, 25, 26],
    ]


def test_text_loss_uses_every_row_and_configured_weight() -> None:
    class Batch(dict):
        def to(self, device):
            return Batch(
                {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in self.items()
                }
            )

    class Processor:
        tokenizer = SimpleNamespace(padding_side="right")

        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            del kwargs
            assert len(messages) == 2
            if add_generation_prompt:
                return Batch(
                    {
                        "input_ids": torch.tensor([[1, 2, 3, 4]] * 2),
                        "attention_mask": torch.ones(2, 4, dtype=torch.long),
                    }
                )
            return Batch(
                {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]] * 2),
                    "attention_mask": torch.ones(2, 6, dtype=torch.long),
                }
            )

    class TinyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))
            self.processor = Processor()
            self.model = SimpleNamespace(device=torch.device("cpu"))
            self.seen_labels = None

        def forward(self, **kwargs):
            self.seen_labels = kwargs["labels"].detach().clone()
            return SimpleNamespace(loss=self.anchor * 2.0)

    class Harness:
        wam_text_loss_weight = 0.005
        wam_text_supervision = {
            "enabled": True,
            "subtask_field": "subtask_text",
            "completed_subtask_field": "completed_subtask_text",
            "prompt_template": "{instruction}",
            "response_template": "{subtask_text} | {completed_subtask_text}",
        }
        qwen_vl_interface = TinyQwen()
        _wam_text_target = Qwen_GR00T._wam_text_target
        _assistant_only_labels = staticmethod(Qwen_GR00T._assistant_only_labels)
        _wam_text_loss = Qwen_GR00T._wam_text_loss

        @staticmethod
        def _wam_views(example):
            del example
            return [], None

    examples = [
        {
            "lang": "first annotation",
            "subtask_text": "move block",
            "completed_subtask_text": "None.",
        },
        {
            "lang": "second annotation",
            "subtask_text": "place block",
            "completed_subtask_text": "moved block",
        },
    ]
    harness = Harness()
    weighted, raw, count = harness._wam_text_loss(examples)
    torch.testing.assert_close(raw, torch.tensor(2.0))
    torch.testing.assert_close(weighted, torch.tensor(0.01))
    assert count == 2
    assert harness.qwen_vl_interface.seen_labels.tolist() == [
        [-100, -100, -100, -100, 5, 6],
        [-100, -100, -100, -100, 5, 6],
    ]


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


def test_causal_shared_query_optimizer_contract_proves_action_dit_is_updated() -> None:
    """causal one-pass must fail before launch if even one live Action-DiT parameter is omitted."""

    from starVLA.training.train_starvla import VLATrainer

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qwen_vl_interface = torch.nn.Linear(3, 3)
            self.action_queries = torch.nn.Embedding(2, 3)
            self.action_model = torch.nn.Linear(3, 3)
            self.future_dino_queries = torch.nn.Embedding(2, 3)
            self.wam_visual_head = torch.nn.Linear(3, 3)
            self.wam_state_ctx = torch.nn.Linear(3, 3)

    model = _Model()
    trainer = VLATrainer.__new__(VLATrainer)
    trainer._jointflow_tasks = ["joint_detached"]
    trainer.model = model
    trainer.config = OmegaConf.create(
        {"trainer": {"wam_two_stage_recipe": "causal_action_world_queries_v1"}}
    )
    trainer.accelerator = SimpleNamespace(is_main_process=False)
    trainer.optimizer = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": 1.0e-4, "name": "all_v5"}]
    )
    VLATrainer._validate_joint_world_optimizer_contract(trainer)

    omitted = next(model.action_model.parameters())
    trainer.optimizer.param_groups[0]["params"] = [
        parameter
        for parameter in trainer.optimizer.param_groups[0]["params"]
        if parameter is not omitted
    ]
    with pytest.raises(RuntimeError, match="causal action-path parameters are absent"):
        VLATrainer._validate_joint_world_optimizer_contract(trainer)


def test_isolated_query_two_stage_yaml_contract() -> None:
    """The new rand pair keeps both pretraining query families and strict ownership."""

    config_dir = REPO_ROOT / "examples/Robotwin/train_files"
    warmup = OmegaConf.load(config_dir / "robotwin_wam_isolated_warmup_rand.yaml")
    gate = OmegaConf.load(config_dir / "robotwin_wam_isolated_gate_ft_rand.yaml")
    for cfg in (warmup, gate):
        merged = merge_framework_config(QwenGR00TDefaultConfig, cfg)
        harness = SimpleNamespace(config=merged)
        Qwen_GR00T._validate_joint_e2e_contract(harness)
        Qwen_GR00T._validate_wam_two_stage_contract(harness)
        assert str(cfg.trainer.wam_two_stage_recipe) == "isolated_queries_v4"
        assert bool(cfg.framework.wam.guidance.pretraining_aligned_queries)
        assert not bool(cfg.framework.wam.guidance.baseline_action_context)
        assert bool(cfg.datasets.vla_data.include_state)
        assert not bool(cfg.framework.action_model.use_correlated_noise)
        assert int(cfg.datasets.vla_data.per_device_batch_size) == 12
        assert int(cfg.trainer.expected_global_batch_size) == 768

    assert bool(warmup.framework.wam.guidance.action_world_bypass)
    assert bool(warmup.framework.wam.guidance.freeze_world_to_action_in_warmup)
    assert bool(warmup.trainer.independent_gradient_clipping.enabled)
    assert float(warmup.trainer.independent_gradient_clipping.action) == 1.0
    assert float(warmup.trainer.independent_gradient_clipping.world) == 1.0
    assert warmup.trainer.gradient_clipping is None
    assert str(warmup.trainer.freeze_modules) == "wam_act_ctx"

    assert not bool(gate.framework.wam.guidance.action_world_bypass)
    assert not bool(gate.framework.wam.guidance.freeze_world_to_action_in_warmup)
    frozen = {item.strip() for item in str(gate.trainer.freeze_modules).split(",")}
    assert {
        "qwen_vl_interface",
        "wam_visual_head",
        "wam_state_ctx",
        "wam_act_ctx",
        "future_dino_queries",
    }.issubset(frozen)
    assert "action_queries" not in frozen
    assert float(gate.trainer.learning_rate.world_gates) == 1.0e-4
    assert float(gate.trainer.learning_rate.world_to_action_adapters) == 1.0e-4
    assert float(gate.trainer.learning_rate.action_model) == 1.0e-5
    assert float(gate.trainer.learning_rate.action_queries) == 1.0e-5
    assert int(gate.trainer.num_warmup_steps) == 500
    assert str(gate.trainer.pretrained_checkpoint) == (
        f"{warmup.run_root_dir}/{warmup.run_id}/final_model/pytorch_model.pt"
    )
    job_dir = REPO_ROOT / "执行脚本/RBT"
    for name in (
        "robotwin_wam_isolated_warmup_rand.yaml",
        "robotwin_wam_isolated_gate_ft_rand.yaml",
    ):
        job = OmegaConf.load(job_dir / name)
        assert int(job.REQUIRED.WORKER_MIN_NUM) == 8
        assert int(job.REQUIRED.WORKER_MAX_NUM) == 8
        assert int(job.REQUIRED.GPU_PER_WORKER) == 8
        assert int(job.REQUIRED.environment.EXPECTED_NUM_MACHINES) == 8
        assert int(job.REQUIRED.environment.GPUS_PER_NODE) == 8
        raw_job = OmegaConf.to_container(job, resolve=False)
        assert raw_job["REQUIRED"]["RUN_SCRIPTS"] == (
            "EXPECTED_NUM_MACHINES=8 ${WORKING_PATH}/run_aidi_rbtw.sh "
            f"examples/Robotwin/train_files/{name}"
        )
    runbook = (job_dir / "run.sh").read_text(encoding="utf-8")
    assert "-f robotwin_wam_isolated_warmup_rand.yaml" in runbook
    assert "-f robotwin_wam_isolated_gate_ft_rand.yaml" in runbook


def test_pretraining_query_banks_have_disjoint_gradient_ownership() -> None:
    """World loss updates FUTURE query but cannot update the Qwen tensor."""

    from starVLA.model.framework.VLM4A.jointflow.joint_modules import (
        ActionQueryTokenBank,
        FutureDinoQueryTokenBank,
    )

    class QueryHarness:
        wam_pretraining_aligned_queries = True
        wam_n_flow = 2
        _jointflow_module_dtype = staticmethod(Qwen_GR00T._jointflow_module_dtype)
        _wam_future_query_condition = Qwen_GR00T._wam_future_query_condition
        _wam_query_embedding_override = Qwen_GR00T._wam_query_embedding_override

        def __init__(self) -> None:
            self.action_queries = ActionQueryTokenBank(3, 4)
            self.future_dino_queries = FutureDinoQueryTokenBank(2, 4)

            class TinyQwen(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.embedding = torch.nn.Embedding(12, 4)
                    self.projection = torch.nn.Linear(4, 4, bias=False)

                def get_input_embeddings(self):
                    return self.embedding

                def forward(self, input_ids):
                    return self.projection(self.embedding(input_ids))

            self.qwen_vl_interface = SimpleNamespace(model=TinyQwen())

    harness = QueryHarness()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    act_mask = torch.tensor([[False, True, True, True, False, False]])
    world_mask = torch.tensor([[False, False, False, False, True, True]])
    with harness._wam_query_embedding_override(act_mask, world_mask):
        hidden = harness.qwen_vl_interface.model(input_ids)
    action_condition = hidden[act_mask].view(1, 3, 4)
    qwen_world = hidden[world_mask].view(1, 2, 4)

    action_condition.square().mean().backward(retain_graph=True)
    assert harness.qwen_vl_interface.model.projection.weight.grad is not None
    assert harness.action_queries.action_query.weight.grad is not None
    future_grad = harness.future_dino_queries.future_dino_query.weight.grad
    assert future_grad is None or float(future_grad.abs().sum()) == 0.0

    for parameter in harness.qwen_vl_interface.model.parameters():
        parameter.grad = None
    harness.action_queries.action_query.weight.grad = None
    harness.future_dino_queries.future_dino_query.weight.grad = None
    world_condition = harness._wam_future_query_condition(qwen_world.detach())
    world_condition.square().mean().backward()
    assert harness.qwen_vl_interface.model.projection.weight.grad is None
    assert harness.action_queries.action_query.weight.grad is None
    assert harness.future_dino_queries.future_dino_query.weight.grad is not None

    # Stage 2 freezes Qwen weights, but ACT queries remain differentiable
    # through that frozen backbone and are still optimized by action loss.
    for parameter in harness.qwen_vl_interface.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    harness.action_queries.action_query.weight.grad = None
    with harness._wam_query_embedding_override(act_mask, world_mask):
        frozen_hidden = harness.qwen_vl_interface.model(input_ids)
    frozen_hidden[act_mask].square().mean().backward()
    assert all(
        parameter.grad is None
        for parameter in harness.qwen_vl_interface.model.parameters()
    )
    assert harness.action_queries.action_query.weight.grad is not None


def test_causal_dual_query_layout_rejects_future_before_action() -> None:
    act = torch.tensor([[False, True, True, False, False]])
    future = torch.tensor([[False, False, False, True, True]])
    Qwen_GR00T._wam_validate_dual_query_layout(act, future, 2, 2)
    with pytest.raises(ValueError, match="ACT placeholder before every FUTURE"):
        Qwen_GR00T._wam_validate_dual_query_layout(future, act, 2, 2)


def test_v5_physically_appends_act_then_future_as_final_2d_mask_suffix() -> None:
    inputs = {
        "input_ids": torch.tensor([[0, 0, 11, 12], [0, 21, 22, 23]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]),
        "position_ids": torch.arange(4).repeat(2, 1),
    }
    original_context = inputs["input_ids"].clone()
    act, future, attention = Qwen_GR00T._wam_append_causal_query_suffix(
        inputs,
        act_token_id=90,
        future_token_id=91,
        n_act=3,
        n_future=2,
    )

    assert inputs["input_ids"][:, :4].equal(original_context)
    assert inputs["input_ids"][:, 4:7].equal(torch.full((2, 3), 90))
    assert inputs["input_ids"][:, 7:].equal(torch.full((2, 2), 91))
    assert act[:, 4:7].all() and not act[:, :4].any() and not act[:, 7:].any()
    assert future[:, 7:].all() and not future[:, :7].any()
    assert attention.ndim == 2 and tuple(attention.shape) == (2, 9)
    assert attention[:, -5:].all()
    assert "position_ids" not in inputs
    Qwen_GR00T._wam_validate_dual_query_layout(act, future, 3, 2)


def test_action_last_suffix_places_visual_queries_before_final_action_queries() -> None:
    inputs = {
        "input_ids": torch.tensor([[0, 11, 12, 13]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1]]),
    }
    act, future, attention = Qwen_GR00T._wam_append_causal_query_suffix(
        inputs,
        act_token_id=90,
        future_token_id=91,
        n_act=3,
        n_future=2,
        action_query_last=True,
    )

    assert inputs["input_ids"][:, 4:6].equal(torch.full((1, 2), 91))
    assert inputs["input_ids"][:, 6:].equal(torch.full((1, 3), 90))
    assert future[:, 4:6].all() and not future[:, :4].any() and not future[:, 6:].any()
    assert act[:, 6:].all() and not act[:, :6].any()
    assert attention[:, -5:].all()
    Qwen_GR00T._wam_validate_dual_query_layout(
        act,
        future,
        expected_act=3,
        expected_future=2,
        action_query_last=True,
    )


def test_causal_query_one_causal_pass_has_required_gradient_routes() -> None:
    """Final ACTION reads visual queries; world loss cannot update ACTION/Action DiT."""

    from starVLA.model.framework.VLM4A.jointflow.joint_modules import (
        ActionQueryTokenBank,
        FutureDinoQueryTokenBank,
    )

    class QueryHarness:
        wam_pretraining_aligned_queries = True
        wam_future_query_through_qwen = True
        _wam_query_embedding_override = Qwen_GR00T._wam_query_embedding_override
        _wam_future_query_condition = Qwen_GR00T._wam_future_query_condition

        def __init__(self) -> None:
            self.action_queries = ActionQueryTokenBank(3, 4)
            self.future_dino_queries = FutureDinoQueryTokenBank(2, 4)

            class TinyCausalQwen(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.embedding = torch.nn.Embedding(12, 4)
                    self.projection = torch.nn.Linear(4, 4, bias=False)

                def get_input_embeddings(self):
                    return self.embedding

                def forward(self, input_ids):
                    # Exact dependency structure of a causal stack for this
                    # test: position i is a function only of positions <= i.
                    return self.projection(self.embedding(input_ids)).cumsum(dim=1)

            self.qwen_vl_interface = SimpleNamespace(model=TinyCausalQwen())
            self.action_head = torch.nn.Linear(4, 4, bias=False)
            self.world_head = torch.nn.Linear(4, 4, bias=False)

    harness = QueryHarness()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    future_mask = torch.tensor([[False, False, True, True, False, False, False]])
    act_mask = torch.tensor([[False, False, False, False, True, True, True]])

    with harness._wam_query_embedding_override(act_mask, future_mask):
        hidden = harness.qwen_vl_interface.model(input_ids)
    action_loss = harness.action_head(hidden[act_mask]).square().mean()
    world_condition = harness._wam_future_query_condition(
        hidden[future_mask].view(1, 2, 4)
    )
    world_loss = harness.world_head(world_condition).square().mean()
    tracked = (
        harness.qwen_vl_interface.model.projection.weight,
        harness.action_queries.action_query.weight,
        harness.future_dino_queries.future_dino_query.weight,
        harness.action_head.weight,
        harness.world_head.weight,
    )
    action_grads = torch.autograd.grad(
        action_loss,
        tracked,
        retain_graph=True,
        allow_unused=True,
    )
    world_grads = torch.autograd.grad(world_loss, tracked, allow_unused=True)

    assert action_grads[0] is not None and action_grads[0].abs().sum() > 0
    assert action_grads[1] is not None and action_grads[1].abs().sum() > 0
    # ACTION is last, so action loss reaches the preceding visual-query bank.
    assert action_grads[2] is not None and action_grads[2].abs().sum() > 0
    assert action_grads[3] is not None and action_grads[3].abs().sum() > 0
    assert action_grads[4] is None or float(action_grads[4].abs().sum()) == 0.0
    assert world_grads[0] is not None and world_grads[0].abs().sum() > 0
    # Earlier visual queries cannot read the later ACTION query bank.
    assert world_grads[1] is None or float(world_grads[1].abs().sum()) == 0.0
    assert world_grads[2] is not None and world_grads[2].abs().sum() > 0
    # The world objective never executes the action head/Action DiT.
    assert world_grads[3] is None or float(world_grads[3].abs().sum()) == 0.0
    assert world_grads[4] is not None and world_grads[4].abs().sum() > 0


def test_ft_optimizer_groups_separate_gate_adapter_action_and_query_lrs() -> None:
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = torch.nn.Linear(4, 4)
            self.world_norm = torch.nn.LayerNorm(4)
            self.world_attn = torch.nn.Linear(4, 4)
            self.world_gate = torch.nn.Parameter(torch.zeros(1))

    class ActionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.transformer_blocks = torch.nn.ModuleList([Block()])

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_model = ActionModel()
            self.action_queries = torch.nn.Embedding(3, 4)
            self.world_adapter = torch.nn.Linear(4, 4)
            self.qwen_vl_interface = torch.nn.Linear(4, 4)
            self.wam_visual_head = torch.nn.Linear(4, 4)
            self.wam_state_ctx = torch.nn.Linear(4, 4)
            self.wam_act_ctx = torch.nn.Linear(4, 4)
            self.future_dino_queries = torch.nn.Embedding(2, 4)

    model = Model()
    cfg = OmegaConf.create(
        {
            "trainer": {
                "freeze_modules": (
                    "qwen_vl_interface,wam_visual_head,wam_state_ctx,"
                    "wam_act_ctx,future_dino_queries"
                ),
                "learning_rate": {
                    "base": 1.0e-5,
                    "world_gates": 1.0e-4,
                    "world_to_action_adapters": 1.0e-4,
                    "action_queries": 1.0e-5,
                    "action_model": 1.0e-5,
                },
            }
        }
    )
    groups = build_param_lr_groups(model, cfg)
    by_name = {group["name"]: group for group in groups}
    assert set(by_name) == {
        "world_gates",
        "world_to_action_adapters",
        "action_queries",
        "action_model",
    }
    assert float(by_name["world_gates"]["lr"]) == 1.0e-4
    assert float(by_name["world_to_action_adapters"]["lr"]) == 1.0e-4
    assert float(by_name["action_model"]["lr"]) == 1.0e-5
    assert float(by_name["action_queries"]["lr"]) == 1.0e-5
    all_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids))
    assert {group["gradient_partition"] for group in groups} == {"action"}


def test_warmup_optimizer_groups_are_disjoint_action_and_world_partitions() -> None:
    """Strict warmup cannot hide a mixed branch in the base LR group."""

    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.core = torch.nn.Linear(4, 4)
            self.world_norm = torch.nn.LayerNorm(4)
            self.world_attn = torch.nn.Linear(4, 4)
            self.world_gate = torch.nn.Parameter(torch.zeros(1))

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qwen_vl_interface = torch.nn.Linear(4, 4)
            self.action_queries = torch.nn.Embedding(3, 4)
            self.action_model = torch.nn.Module()
            self.action_model.model = torch.nn.Module()
            self.action_model.model.transformer_blocks = torch.nn.ModuleList([Block()])
            self.future_dino_queries = torch.nn.Embedding(2, 4)
            self.wam_visual_head = torch.nn.Linear(4, 4)
            self.wam_state_ctx = torch.nn.Linear(4, 4)
            self.wam_act_ctx = torch.nn.Linear(4, 4)
            self.world_adapter = torch.nn.Linear(4, 4)
            for name, parameter in self.named_parameters():
                if (
                    name.startswith("world_adapter.")
                    or ".world_attn." in name
                    or ".world_norm." in name
                    or name.endswith(".world_gate")
                ):
                    parameter.requires_grad_(False)

    model = Model()
    cfg = OmegaConf.create(
        {
            "trainer": {
                "freeze_modules": "wam_act_ctx",
                "learning_rate": {
                    "base": 1.0e-5,
                    "qwen_vl_interface": 1.0e-5,
                    "action_queries": 1.0e-4,
                    "action_model": 1.0e-4,
                    "future_dino_queries": 1.0e-4,
                    "wam_visual_head": 1.0e-4,
                    "wam_state_ctx": 1.0e-4,
                },
            }
        }
    )
    groups = build_param_lr_groups(model, cfg)
    assert {group["gradient_partition"] for group in groups} == {
        "action",
        "world",
    }
    assert all(group["gradient_partition"] != "mixed" for group in groups)
    assert {group["name"] for group in groups} == {
        "qwen_vl_interface",
        "action_queries",
        "action_model",
        "future_dino_queries",
        "wam_visual_head",
        "wam_state_ctx",
    }
    all_ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(all_ids) == len(set(all_ids))


def test_zero2_branch_clipper_scales_action_and_world_independently() -> None:
    from starVLA.training.train_starvla import VLATrainer

    action_grad = torch.tensor([3.0, 4.0])
    world_grad = torch.tensor([0.0, 10.0])

    class Inner:
        param_groups = [
            {"gradient_partition": "action"},
            {"gradient_partition": "world"},
        ]

    class Zero:
        optimizer = Inner()
        averaged_gradients = {0: [action_grad], 1: [world_grad]}
        params_in_partition = {0: [torch.nn.Parameter(torch.zeros(2))], 1: [torch.nn.Parameter(torch.zeros(2))]}
        loss_scale = 1.0

        @staticmethod
        def get_grad_norm_direct(gradients, _params):
            return torch.linalg.vector_norm(torch.cat([gradient.flatten() for gradient in gradients]))

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = SimpleNamespace(optimizer=Zero())
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer._independent_clip_limits = {"action": 1.0, "world": 2.0}
    metrics = VLATrainer._clip_independent_zero2_partitions(trainer)
    torch.testing.assert_close(action_grad, torch.tensor([0.6, 0.8]), atol=2.0e-6, rtol=0)
    torch.testing.assert_close(world_grad, torch.tensor([0.0, 2.0]), atol=2.0e-6, rtol=0)
    torch.testing.assert_close(metrics["train/grad_norm_preclip/action"], torch.tensor(5.0))
    torch.testing.assert_close(metrics["train/grad_norm_preclip/world"], torch.tensor(10.0))


@pytest.mark.skip(reason="retired RobotWin warmup/gate configs were removed")
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


@pytest.mark.skip(reason="retired RobotWin warmup/gate configs were removed")
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


def test_removed_robodojo_warmup_config_stays_removed() -> None:
    # The current RoboDojo surface contains only the causal-DINO MoT configs.
    config_path = REPO_ROOT / "examples/RoboDojo/train_files/starvla_qwen3_robodojo_causal_query_warmup.yaml"
    assert not config_path.exists()


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


def test_event_semantic_loader_cannot_define_deepspeed_train_micro_batch() -> None:
    from starVLA.training.train_starvla import VLATrainer

    train_loader = SimpleNamespace(batch_size=12)
    semantic_loader = SimpleNamespace(batch_size=None)
    model = object()
    optimizer = object()
    calls: list[tuple[str, tuple[object, ...]]] = []

    class _Accelerator:
        def prepare(self, *components):
            calls.append(("prepare", components))
            return components

        def prepare_data_loader(self, loader):
            calls.append(("prepare_data_loader", (loader,)))
            return ("prepared_loader", loader)

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.accelerator = _Accelerator()
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.vla_train_dataloader = train_loader
    trainer.semantic_train_dataloader = semantic_loader
    trainer.vla_val_dataloader = None

    VLATrainer._prepare_distributed_components(trainer)

    assert calls == [
        ("prepare", (model, optimizer, train_loader)),
        ("prepare_data_loader", (semantic_loader,)),
    ]
    assert trainer.vla_train_dataloader is train_loader
    assert trainer.semantic_train_dataloader == (
        "prepared_loader",
        semantic_loader,
    )


def test_event_semantic_iterator_keeps_restored_sampler_epoch() -> None:
    from starVLA.training.train_starvla import VLATrainer

    trainer = VLATrainer.__new__(VLATrainer)
    trainer.vla_train_dataloader = ["physical"]
    trainer.semantic_train_dataloader = ["semantic"]
    trainer.semantic_batch_sampler = SimpleNamespace(epoch=17)

    VLATrainer._create_data_iterators(trainer)

    assert trainer.semantic_epoch_count == 17
    assert next(trainer.semantic_iter) == "semantic"


def test_rynn_mem_packs_history_privately_and_keeps_one_vlm_image_span() -> None:
    history_frames = [object() for _ in range(5)]
    current_frame = object()

    class FakeImageProcessor:
        def __call__(self, *, images, return_tensors):
            assert return_tensors == "pt"
            assert images == [*history_frames, current_frame]
            return {
                "pixel_values": torch.randn(6 * 4, 12),
                "image_grid_thw": torch.tensor([[1, 2, 2]] * 6),
            }

    harness = SimpleNamespace(
        text_history_enabled=True,
        mem_vision_encoder_enabled=True,
        mem_vision_encoder={"num_frames": 6},
        qwen_vl_interface=SimpleNamespace(
            processor=SimpleNamespace(image_processor=FakeImageProcessor())
        ),
        _planner_history_views=lambda _example: list(history_frames),
        _planner_current_views=lambda _example: [current_frame],
    )

    message = QwenWorldActionMoT._planner_user_message(
        harness,
        {},
        "plan",
    )
    image_items = [
        item
        for item in message["content"]
        if item.get("type") == "image"
    ]
    assert len(image_items) == 1
    assert image_items[0]["image"] is current_frame

    inputs = {
        "pixel_values": torch.randn(4, 12),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    packed = QwenWorldActionMoT._prepare_mem_vision_inputs(
        harness,
        inputs,
        [{}],
    )
    assert packed["pixel_values"].shape[0] == 24
    assert packed["image_grid_thw"].shape == (1, 3)
    assert packed["_starvla_mem_image_grid_thw"].shape == (6, 3)


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
    test_causal_query_recipe_has_no_action_injection_modules()
    test_no_world2action_runtime_contract_requires_physical_absence()
    test_robodojo_warmup_and_gate_yaml_pass_early_contracts()
    test_causal_query_yaml_has_strict_world_learning_contract()
    test_world_validation_reports_sampled_mse_cosine_and_copy_baseline()
    test_joint_world_optimizer_contract_rejects_zero_lr()
    test_joint_world_optimizer_contract_accepts_scheduler_warmup_zero_lr()
    test_gate_reset_changes_only_world_gate_parameters()
    test_full_training_state_discovery_requires_commit_marker()
    test_accelerator_state_roundtrips_registered_raw_scheduler()
    test_joint_e2e_contract_rejects_gate_bypass_and_corrnoise()
    test_trainer_preserves_two_microbatch_gradient_accumulation()
    test_deepspeed_runtime_batch_contract_matches_yaml()
    test_world_validation_loader_cannot_define_deepspeed_train_micro_batch()
