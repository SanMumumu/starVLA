"""Small unit checks for WAM/FastWAM loss-contract plumbing."""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.framework.VLM4A.jointflow.dino_v3 import DINOv3Backbone, dino_num_patches, dino_patch_grid
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


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


if __name__ == "__main__":
    test_wam_action_loss_repeats_state_and_padding_like_baseline()
    test_wam_visual_loss_masks_padded_future_without_scale_drift()
    test_dino_preserves_fastwam_composite_as_24_by_20_grid()
    test_frozen_dino_teacher_is_not_checkpointed()
    test_action_dit_checkpointing_preserves_forward_and_gradients()
