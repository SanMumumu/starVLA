"""Small unit checks for WAM/FastWAM loss-contract plumbing."""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T, QwenGR00TDefaultConfig
from starVLA.model.framework.VLM4A.jointflow.dino_v3 import DINOv3Backbone, dino_num_patches, dino_patch_grid
from starVLA.model.framework.VLM4A.wam_guidance import linear_gradient_ramp
from starVLA.model.framework.share_tools import merge_framework_config
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


class _JointE2EForwardHarness:
    _wam_action_world_grad_scale = Qwen_GR00T._wam_action_world_grad_scale

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

    def _wam_guided_backbone(self, examples, task):
        assert len(examples) == 1 and task == "joint_e2e"
        self.backbone_calls += 1
        h_act = torch.ones(1, 2, 4, requires_grad=True)
        h_future = torch.full((1, 3, 4), 2.0, requires_grad=True)
        hidden = torch.ones(1, 5, 4, requires_grad=True)
        attention = torch.ones(1, 5, dtype=torch.bool)
        placeholders = torch.zeros(1, 5, dtype=torch.bool)
        return h_act, h_future, hidden, attention, placeholders

    def _wam_world_target(self, examples):
        assert len(examples) == 1
        return torch.zeros(1, 3, 4)

    @staticmethod
    def _jointflow_module_dtype(module, fallback):
        return next(module.parameters()).dtype

    def _wam_visual_loss(self, cond, target, examples):
        assert cond.shape == target.shape == (1, 3, 4)
        return (cond - target).square().mean()

    def _build_world_signal(self, h_future, examples, action_world_grad_scale=1.0):
        self.world_signal_scales.append(float(action_world_grad_scale))
        return 3.0 * h_future

    def _assemble_guided_inputs(self, mode, h_act, h_future, hidden, attn, ph_mask, world_tokens):
        assert mode == "dual_xattn"
        self.action_world_tokens = world_tokens
        return h_act, torch.ones(h_act.shape[:2], dtype=torch.bool), world_tokens, None, None

    def _stack_jointflow_field(self, examples, key, required):
        assert key == "action" and required
        return torch.ones(1, 2, 2)

    @staticmethod
    def _wam_action_state_and_mask(examples):
        return None, torch.zeros(1, 2, dtype=torch.bool)

    @staticmethod
    def _wam_action_loss(mem, actions, state, mem_mask, action_is_pad, **kwargs):
        assert state is None and kwargs["guidance_mode"] == "dual_xattn"
        assert kwargs["world_embs"] is not None
        return mem.square().mean() + kwargs["world_embs"].square().mean()

    @staticmethod
    def _wam_guided_unused_anchor(task, loss):
        assert task == "joint_e2e"
        return loss.new_zeros(())

    @staticmethod
    def _wam_world_gate_metrics():
        return {
            "world_gate_openness": torch.tensor(0.25),
            "world_gate_signed_mean": torch.tensor(-0.125),
            "world_gate_max_openness": torch.tensor(0.5),
        }


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


def test_e2e_yamls_pass_early_joint_contract() -> None:
    config_paths = (
        REPO_ROOT / "examples/Robotwin/train_files/robotwin_wam_e2e_rand.yaml",
        REPO_ROOT
        / "examples/Robotwin/train_files/robotwin_wam_e2e_rand_worldmem_nocontext.yaml",
        REPO_ROOT / "examples/Robotwin/train_files/robotwin_wam_e2e_clean.yaml",
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
        if config_path.name == "robotwin_wam_e2e_rand_worldmem_nocontext.yaml":
            guidance = cfg.framework.wam.guidance
            assert bool(guidance.include_context_in_action_memory)
            assert not bool(guidance.include_context_in_world_memory)


def test_robotwin_two_stage_budget_is_80k_warmup_plus_20k_gate() -> None:
    config_dir = REPO_ROOT / "examples/Robotwin/train_files"
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
        assert str(gate.run_id).endswith("_20k")
        assert str(gate.trainer.pretrained_checkpoint).endswith(
            f"/{warmup.run_id}/final_model/pytorch_model.pt"
        )


def test_joint_e2e_contract_rejects_gate_bypass_and_corrnoise() -> None:
    config_path = REPO_ROOT / "examples/Robotwin/train_files/robotwin_wam_e2e_rand.yaml"
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
            "trainer": {"gradient_accumulation_steps": 2},
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

    trainer.model.train_batch_size = lambda: 384
    try:
        VLATrainer._validate_runtime_batch_contract(trainer)
    except RuntimeError as exc:
        assert "train_batch_size differs" in str(exc)
    else:
        raise AssertionError("DeepSpeed global-batch mismatch was not rejected")


if __name__ == "__main__":
    test_wam_action_loss_repeats_state_and_padding_like_baseline()
    test_wam_visual_loss_masks_padded_future_without_scale_drift()
    test_dino_preserves_fastwam_composite_as_24_by_20_grid()
    test_frozen_dino_teacher_is_not_checkpointed()
    test_action_dit_checkpointing_preserves_forward_and_gradients()
    test_joint_e2e_predicted_future_forward_is_unchanged_and_backward_is_ramped()
    test_joint_e2e_one_backbone_forward_returns_action_and_world_losses()
    test_world_gate_metrics_report_effective_tanh_openness()
    test_joint_e2e_trainer_metrics_expose_world_gate_to_wandb()
    test_two_stage_gate_metrics_expose_world_gate_to_wandb()
    test_dual_query_layout_is_causal_act_to_future_and_suffix_is_not_context()
    test_new_e2e_context_mask_closes_post_query_gate_bypass()
    test_action_and_world_memory_context_can_be_decoupled()
    test_yaml_task_weights_replace_framework_default_task_set()
    test_e2e_yaml_injects_zero_initialized_world_gates()
    test_e2e_yamls_pass_early_joint_contract()
    test_robotwin_two_stage_budget_is_80k_warmup_plus_20k_gate()
    test_joint_e2e_contract_rejects_gate_bypass_and_corrnoise()
    test_trainer_preserves_two_microbatch_gradient_accumulation()
    test_deepspeed_runtime_batch_contract_matches_yaml()
