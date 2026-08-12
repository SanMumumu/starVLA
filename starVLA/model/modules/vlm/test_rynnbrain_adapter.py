from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors import safe_open

from starVLA.model.modules.vlm import resolve_vlm_family


REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_RYNN = REPO_ROOT / "执行脚本/rynnbrain1.1-2B"
CLUSTER_RYNN = "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/rynnbrain1.1-2B/"


def test_rynnbrain_local_and_cluster_paths_route_to_qwen35() -> None:
    assert resolve_vlm_family(str(LOCAL_RYNN)) == "qwen3_5"
    assert resolve_vlm_family(CLUSTER_RYNN) == "qwen3_5"
    assert resolve_vlm_family("Qwen/Qwen3.5-2B") == "qwen3_5"


def test_downloaded_rynnbrain_checkpoint_has_expected_starvla_abi() -> None:
    config = json.loads((LOCAL_RYNN / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert config["model_type"] == "qwen3_5"
    assert int(config["text_config"]["hidden_size"]) == 2048
    assert int(config["text_config"]["num_hidden_layers"]) == 24
    weights = LOCAL_RYNN / "model.safetensors"
    assert weights.stat().st_size > 4_000_000_000
    with safe_open(weights, framework="pt", device="cpu") as tensors:
        assert len(tensors.keys()) == 617
        assert tensors.get_slice(
            "model.language_model.embed_tokens.weight"
        ).get_shape() == [248320, 2048]
        assert tensors.get_slice(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        ).get_shape() == [6144, 2048]
        assert tensors.get_slice(
            "model.visual.patch_embed.proj.weight"
        ).get_shape() == [1024, 3, 2, 16, 16]


def test_qwen35_interface_enables_training_contract(monkeypatch) -> None:
    import starVLA.model.modules.vlm.QWen3_5 as module

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="qwen3_5",
                text_config=SimpleNamespace(hidden_size=2048),
            )
            self.gradient_checkpointing_kwargs = None
            self.input_grads_enabled = False

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

        def enable_input_require_grads(self):
            self.input_grads_enabled = True

    fake_model = FakeModel()
    captured = {}

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)
            return fake_model

    fake_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="right")
    )
    monkeypatch.setattr(
        module.transformers,
        "Qwen3_5ForConditionalGeneration",
        FakeModelClass,
        raising=False,
    )
    monkeypatch.setattr(
        module.AutoProcessor,
        "from_pretrained",
        lambda _model_id: fake_processor,
    )
    cfg = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {
                    "base_vlm": CLUSTER_RYNN,
                    "attn_implementation": "sdpa",
                    "enable_gradient_checkpointing": True,
                }
            }
        }
    )

    interface = module._QWen3_5_VL_Interface(cfg)
    assert interface.model is fake_model
    assert interface.processor.tokenizer.padding_side == "left"
    assert interface.attn_implementation == "sdpa"
    assert captured["model_id"] == CLUSTER_RYNN
    assert captured["attn_implementation"] == "sdpa"
    assert captured["dtype"] == torch.bfloat16
    assert fake_model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert fake_model.input_grads_enabled
    assert int(interface.model.config.hidden_size) == 2048


def test_qwen35_safe_causal_conv_replaces_only_conv_extension(monkeypatch) -> None:
    import starVLA.model.modules.vlm.QWen3_5 as module

    fast_conv = lambda *args, **kwargs: None
    fast_update = lambda *args, **kwargs: None

    class LinearAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.causal_conv1d_fn = fast_conv
            self.causal_conv1d_update = fast_update
            self.chunk_gated_delta_rule = object()

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attention = LinearAttention()

    fake_hf_module = SimpleNamespace(
        causal_conv1d_fn=fast_conv,
        causal_conv1d_update=fast_update,
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: fake_hf_module,
    )
    model = FakeModel()
    original_fla = model.linear_attention.chunk_gated_delta_rule

    replaced = module._install_safe_qwen35_causal_conv(model)

    assert replaced == 1
    assert fake_hf_module.causal_conv1d_fn is module._safe_causal_conv1d_fn
    assert (
        fake_hf_module.causal_conv1d_update
        is module._safe_causal_conv1d_update
    )
    assert (
        model.linear_attention.causal_conv1d_fn
        is module._safe_causal_conv1d_fn
    )
    assert (
        model.linear_attention.causal_conv1d_update
        is module._safe_causal_conv1d_update
    )
    assert model.linear_attention.chunk_gated_delta_rule is original_fla

    torch.manual_seed(7)
    x = torch.randn(2, 3, 5)
    weight = torch.randn(3, 4)
    bias = torch.randn(3)
    actual = module._safe_causal_conv1d_fn(x, weight, bias, "silu")
    expected = F.silu(
        F.conv1d(
            x,
            weight.unsqueeze(1),
            bias,
            padding=3,
            groups=3,
        )[..., :5]
    )
    torch.testing.assert_close(actual, expected)


def test_qwen35_safe_fla_replaces_delta_rule_and_fused_norm(
    monkeypatch,
) -> None:
    import starVLA.model.modules.vlm.QWen3_5 as module

    fast_chunk = lambda *args, **kwargs: None
    fast_recurrent = lambda *args, **kwargs: None

    def torch_chunk(*args, **kwargs):
        return args, kwargs

    def torch_recurrent(*args, **kwargs):
        return args, kwargs

    class ReferenceNorm(torch.nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

    class FusedNorm(torch.nn.Module):
        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.arange(hidden_size, dtype=torch.float32)
            )

    class LinearAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head_v_dim = 4
            self.layer_norm_epsilon = 1.0e-5
            self.chunk_gated_delta_rule = fast_chunk
            self.recurrent_gated_delta_rule = fast_recurrent
            self.norm = FusedNorm(self.head_v_dim)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_attention = LinearAttention()

    fake_hf_module = SimpleNamespace(
        torch_chunk_gated_delta_rule=torch_chunk,
        torch_recurrent_gated_delta_rule=torch_recurrent,
        Qwen3_5RMSNormGated=ReferenceNorm,
    )
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: fake_hf_module,
    )
    model = FakeModel()
    expected_weight = model.linear_attention.norm.weight.detach().clone()

    replaced = module._install_safe_qwen35_fla(model)

    assert replaced == 1
    assert model.linear_attention.chunk_gated_delta_rule is torch_chunk
    assert model.linear_attention.recurrent_gated_delta_rule is torch_recurrent
    assert isinstance(model.linear_attention.norm, ReferenceNorm)
    assert model.linear_attention.norm.variance_epsilon == 1.0e-5
    torch.testing.assert_close(
        model.linear_attention.norm.weight,
        expected_weight,
    )


def test_qwen35_safe_kernels_are_selected_before_model_construction(
    monkeypatch,
) -> None:
    import starVLA.model.modules.vlm.QWen3_5 as module

    fake_hf_module = SimpleNamespace(
        causal_conv1d_fn=object(),
        causal_conv1d_update=object(),
        chunk_gated_delta_rule=object(),
        fused_recurrent_gated_delta_rule=object(),
        FusedRMSNormGated=object(),
        is_fast_path_available=True,
        flash_attention_sentinel=object(),
    )
    original_flash_attention = fake_hf_module.flash_attention_sentinel
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: fake_hf_module,
    )

    module._prepare_safe_qwen35_kernels(
        type("FakeModelClass", (), {}),
        safe_causal_conv=True,
        safe_fla=True,
    )

    assert fake_hf_module.causal_conv1d_fn is module._safe_causal_conv1d_fn
    assert (
        fake_hf_module.causal_conv1d_update
        is module._safe_causal_conv1d_update
    )
    assert fake_hf_module.chunk_gated_delta_rule is None
    assert fake_hf_module.fused_recurrent_gated_delta_rule is None
    assert fake_hf_module.FusedRMSNormGated is None
    assert fake_hf_module.is_fast_path_available is False
    assert fake_hf_module.flash_attention_sentinel is original_flash_attention


def test_qwen35_attention_override_is_inference_only_environment_choice(
    monkeypatch,
) -> None:
    import starVLA.model.modules.vlm.QWen3_5 as module

    monkeypatch.delenv("STARVLA_QWEN35_ATTN_IMPLEMENTATION", raising=False)
    assert module._resolve_qwen35_attn_implementation(
        "flash_attention_2"
    ) == ("flash_attention_2", "config")

    monkeypatch.setenv("STARVLA_QWEN35_ATTN_IMPLEMENTATION", "sdpa")
    assert module._resolve_qwen35_attn_implementation(
        "flash_attention_2"
    ) == ("sdpa", "environment")

    monkeypatch.setenv("STARVLA_QWEN35_ATTN_IMPLEMENTATION", "invalid")
    with pytest.raises(ValueError, match="must be one of"):
        module._resolve_qwen35_attn_implementation("flash_attention_2")


def test_rynn_mem_patch_is_opt_in_zero_parameter_and_shape_preserving(
    monkeypatch,
) -> None:
    import starVLA.model.modules.vlm.rynn_mem_encoder as mem

    hidden_size = 8
    num_heads = 2

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.scaling = (hidden_size // num_heads) ** -0.5
            self.attention_dropout = 0.0
            self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3)
            self.proj = torch.nn.Linear(hidden_size, hidden_size)

    class FakeBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(hidden_size)
            self.norm2 = torch.nn.LayerNorm(hidden_size)
            self.attn = FakeAttention()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size * 2),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size * 2, hidden_size),
            )

        def forward(self, hidden_states, **_kwargs):
            return hidden_states

    class FakeVisual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([FakeBlock() for _ in range(8)])
            self.spatial_merge_size = 1

        def forward(self, hidden_states, grid_thw, **_kwargs):
            expected = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
            assert hidden_states.shape[0] == expected
            return SimpleNamespace(
                last_hidden_state=hidden_states,
                pooler_output=hidden_states,
            )

    class FakeBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = FakeVisual()

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeBackbone()

    def identity_rotary(query, key, _cos, _sin):
        return query, key

    monkeypatch.setattr(
        mem.importlib,
        "import_module",
        lambda _name: SimpleNamespace(
            apply_rotary_pos_emb_vision=identity_rotary
        ),
    )
    model = FakeModel()
    before_keys = tuple(model.state_dict())
    before_parameters = sum(parameter.numel() for parameter in model.parameters())

    summary = mem.apply_rynn_mem_encoder_patch(
        model,
        {
            "enabled": True,
            "num_frames": 6,
            "spacetime_layer_stride": 4,
            "time_embed_base": 100.0,
        },
    )

    assert summary["patched_indices"] == [3, 7]
    assert summary["added_parameters"] == 0
    assert summary["language_frame_tokens"] == "current_only"
    assert tuple(model.state_dict()) == before_keys
    assert sum(parameter.numel() for parameter in model.parameters()) == before_parameters
    assert not hasattr(model.model.visual.blocks[2], "_starvla_mem_enabled")
    assert model.model.visual.blocks[3]._starvla_mem_enabled

    batch_size, num_frames, tokens_per_frame = 2, 6, 3
    hidden = torch.randn(
        batch_size * num_frames * tokens_per_frame,
        hidden_size,
        requires_grad=True,
    )
    cu_seqlens = torch.arange(
        batch_size * num_frames + 1,
        dtype=torch.int32,
    ) * tokens_per_frame
    cos = torch.ones(hidden.shape[0], hidden_size // num_heads)
    sin = torch.zeros_like(cos)
    output = model.model.visual.blocks[3](
        hidden,
        cu_seqlens=cu_seqlens,
        position_embeddings=(cos, sin),
    )
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None

    # The vision tower sees all B*K frames, but its public output retains only
    # frame K-1 for each sample, matching one current-image placeholder span.
    mem_grid = torch.tensor([[1, 1, tokens_per_frame]] * 12)
    current_grid = torch.tensor([[1, 1, tokens_per_frame]] * batch_size)
    vision_hidden = torch.arange(
        batch_size * num_frames * tokens_per_frame * hidden_size,
        dtype=torch.float32,
    ).view(-1, hidden_size)
    model.model.visual._starvla_mem_active_grid_thw = mem_grid
    vision_output = model.model.visual(
        vision_hidden,
        grid_thw=current_grid,
    )
    assert vision_output.pooler_output.shape == (
        batch_size * tokens_per_frame,
        hidden_size,
    )
    expected_current = vision_hidden.view(
        batch_size,
        num_frames,
        tokens_per_frame,
        hidden_size,
    )[:, -1].reshape(batch_size * tokens_per_frame, hidden_size)
    torch.testing.assert_close(vision_output.pooler_output, expected_current)
    torch.testing.assert_close(vision_output.last_hidden_state, expected_current)


def test_rynn_mem_disabled_does_not_inspect_or_modify_model() -> None:
    import starVLA.model.modules.vlm.rynn_mem_encoder as mem

    sentinel = torch.nn.Linear(2, 2)
    before = tuple(sentinel.state_dict())
    summary = mem.apply_rynn_mem_encoder_patch(sentinel, {"enabled": False})
    assert summary["enabled"] is False
    assert tuple(sentinel.state_dict()) == before
