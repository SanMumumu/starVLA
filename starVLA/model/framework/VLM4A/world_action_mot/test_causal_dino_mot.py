from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.world_action_mot.causal_dino_mot import (
    CausalDINOActionLayer,
    CausalDINOActionMoT,
)


TRAIN_CONFIG_DIR = REPO_ROOT / "examples" / "RoboDojo" / "train_files"


def _tiny_model(interaction_mode: str) -> CausalDINOActionMoT:
    return CausalDINOActionMoT(
        planner_dim=12,
        world_dim=8,
        action_config={
            "action_horizon": 4,
            "action_dim": 3,
            "state_dim": 3,
        },
        mot_config={
            "interaction_mode": interaction_mode,
            "world_attention_mask_mode": "first_frame_causal",
            "world_hidden_size": 24,
            "action_hidden_size": 16,
            "world_ffn_dim": 48,
            "action_ffn_dim": 32,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "time_frequency_dim": 8,
            "world_grid_height": 2,
            "world_grid_width": 3,
            "num_inference_timesteps": 2,
            "enable_gradient_checkpointing": False,
        },
    )


def test_base_and_joint_masks_differ_only_for_action_reading_future() -> None:
    base = _tiny_model("base")
    joint = _tiny_model("joint")
    current = base.world_tokens
    future_end = 2 * current

    difference = base.physical_attention_mask ^ joint.physical_attention_mask
    expected = torch.zeros_like(difference)
    expected[future_end:, current:future_end] = True
    assert torch.equal(difference, expected)

    # Current DINO cannot read future DINO; future DINO sees both world frames.
    assert not base.physical_attention_mask[:current, current:future_end].any()
    assert base.physical_attention_mask[current:future_end, :future_end].all()
    # Neither mode permits the world stream to read action.
    assert not base.physical_attention_mask[:future_end, future_end:].any()
    assert not joint.physical_attention_mask[:future_end, future_end:].any()


def test_experts_keep_independent_qkv_ffn_and_rope() -> None:
    model = _tiny_model("joint")
    assert all(isinstance(layer, CausalDINOActionLayer) for layer in model.layers)
    for layer in model.layers:
        assert layer.world.self_attn.q is not layer.action.self_attn.q
        assert layer.world.self_attn.k is not layer.action.self_attn.k
        assert layer.world.self_attn.v is not layer.action.self_attn.v
        assert layer.world.ffn is not layer.action.ffn
    assert model.world_frequencies.shape == (
        2 * model.world_tokens,
        1,
        model.head_dim // 2,
    )
    assert model.action_frequencies.shape == (
        model.action_horizon,
        1,
        model.head_dim // 2,
    )
    assert model.world_frame_embedding is not None
    torch.testing.assert_close(
        model.world_frequencies[: model.world_tokens],
        model.world_frequencies[model.world_tokens :],
    )
    world_frequencies = model.world_frequencies.clone()
    action_frequencies = model.action_frequencies.clone()
    model.to(dtype=torch.bfloat16)
    assert model.world_input.weight.dtype == torch.bfloat16
    assert model.world_frequencies.is_complex()
    assert model.action_frequencies.is_complex()
    torch.testing.assert_close(model.world_frequencies, world_frequencies)
    torch.testing.assert_close(model.action_frequencies, action_frequencies)


def test_tiny_train_and_sample_paths_share_shapes_state_and_dtype() -> None:
    model = _tiny_model("base")
    batch = 2
    inputs = {
        "action_plan": torch.randn(batch, 4, 12),
        "world_plan": torch.randn(batch, 4, 12),
        "current_world": torch.randn(batch, model.world_tokens, 8),
        "current_state": torch.randn(batch, 1, 3),
    }
    output = model.forward_train(
        **inputs,
        target_action=torch.randn(batch, 4, 3),
        target_world=torch.randn(batch, model.world_tokens, 8),
        action_is_pad=torch.zeros(batch, 4, dtype=torch.bool),
        future_valid=torch.ones(batch),
    )
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])
    output["loss"].backward()

    model.eval()
    action, future_world = model.sample(**inputs, seed=7)
    assert action.shape == (batch, 4, 3)
    assert future_world.shape == (batch, model.world_tokens, 8)
    assert action.dtype == model.action_input.weight.dtype
    assert future_world.dtype == model.world_input.weight.dtype


def test_rejects_noncausal_world_mask_mode() -> None:
    with pytest.raises(ValueError, match="first_frame_causal"):
        CausalDINOActionMoT(
            planner_dim=12,
            world_dim=8,
            action_config={
                "action_horizon": 4,
                "action_dim": 3,
                "state_dim": 0,
            },
            mot_config={
                "interaction_mode": "base",
                "world_attention_mask_mode": "bidirectional",
                "world_hidden_size": 24,
                "action_hidden_size": 16,
                "world_ffn_dim": 48,
                "action_ffn_dim": 32,
                "num_layers": 1,
                "num_attention_heads": 2,
                "attention_head_dim": 12,
                "time_frequency_dim": 8,
                "world_grid_height": 2,
                "world_grid_width": 3,
            },
        )


@pytest.mark.parametrize(
    ("filename", "interaction_mode"),
    [
        ("mot_base.yaml", "base"),
        ("mot_joint.yaml", "joint"),
        ("rynn_base.yaml", "base"),
        ("rynn_joint.yaml", "joint"),
    ],
)
def test_training_configs_pin_the_physical_contract(
    filename: str,
    interaction_mode: str,
) -> None:
    path = TRAIN_CONFIG_DIR / filename
    config_text = path.read_text(encoding="utf-8")
    config = OmegaConf.load(path)
    physical = config.framework.world_action_mot

    assert "fastwam" not in config_text.lower()
    assert physical.architecture == "causal_dino_mot"
    assert physical.interaction_mode == interaction_mode
    assert physical.world_attention_mask_mode == "first_frame_causal"
    assert (physical.world_hidden_size, physical.action_hidden_size) == (512, 1024)
    assert (physical.world_ffn_dim, physical.action_ffn_dim) == (2048, 4096)
    assert (
        physical.num_layers,
        physical.num_attention_heads,
        physical.attention_head_dim,
    ) == (30, 24, 128)
    assert (
        physical.world_train_shift,
        physical.world_infer_shift,
        physical.action_train_shift,
        physical.action_infer_shift,
    ) == (5.0, 5.0, 5.0, 5.0)
    assert (
        physical.world_num_train_timesteps,
        physical.action_num_train_timesteps,
    ) == (1000, 1000)
    assert config.framework.dino.embed_dim == 768
    assert config.framework.dino.dino_pool == 2
    assert config.datasets.vla_data.image_layout == "tri_view_composite"
    if filename.startswith("rynn_"):
        assert config.framework.qwenvl.attn_implementation == "sdpa"
        assert str(config.framework.qwenvl.base_vlm).rstrip("/").endswith(
            "/rynnbrain1.1-2B"
        )
        assert not config.framework.qwenvl.enable_thinking
    else:
        assert config.framework.qwenvl.attn_implementation == "flash_attention_2"
        assert str(config.framework.qwenvl.base_vlm).rstrip("/").endswith(
            "/Qwen3-VL-2B-Instruct"
        )


def test_exact_yaml_topology_is_about_1b_action_and_450m_world() -> None:
    config = OmegaConf.load(TRAIN_CONFIG_DIR / "mot_joint.yaml")
    physical = config.framework.world_action_mot
    planner_dim = int(config.framework.qwenvl.vl_hidden_dim)
    world_dim = int(config.framework.dino.embed_dim)
    action_dim = int(config.framework.action_model.action_dim)
    inner_dim = (
        int(physical.num_attention_heads) * int(physical.attention_head_dim)
    )
    layers = int(physical.num_layers)

    def block_parameters(hidden_size: int, ffn_dim: int) -> int:
        # Two independent attention modules (self and cross), affine norm3,
        # two-layer FFN and six-way time modulation.
        return (
            8 * hidden_size * inner_dim
            + 10 * inner_dim
            + 11 * hidden_size
            + 2 * hidden_size * ffn_dim
            + ffn_dim
        )

    action_hidden = int(physical.action_hidden_size)
    action_ffn = int(physical.action_ffn_dim)
    action_parameters = layers * block_parameters(action_hidden, action_ffn)
    action_parameters += action_dim * action_hidden + action_hidden
    action_parameters += (
        planner_dim * action_hidden
        + action_hidden
        + action_hidden * action_hidden
        + action_hidden
    )
    action_parameters += (
        int(physical.time_frequency_dim) * action_hidden
        + action_hidden
        + action_hidden * action_hidden
        + action_hidden
    )
    action_parameters += action_hidden * (6 * action_hidden) + 6 * action_hidden
    action_parameters += action_hidden * action_dim + action_dim

    world_hidden = int(physical.world_hidden_size)
    world_ffn = int(physical.world_ffn_dim)
    world_parameters = layers * block_parameters(world_hidden, world_ffn)
    world_parameters += world_dim * world_hidden + world_hidden
    world_parameters += (
        planner_dim * world_hidden
        + world_hidden
        + world_hidden * world_hidden
        + world_hidden
    )
    world_parameters += (
        int(physical.time_frequency_dim) * world_hidden
        + world_hidden
        + world_hidden * world_hidden
        + world_hidden
    )
    world_parameters += world_hidden * (6 * world_hidden) + 6 * world_hidden
    world_parameters += world_hidden * world_dim + world_dim + 2 * world_hidden
    world_parameters += 2 * world_hidden  # current/future frame type

    assert action_parameters == 1_018_803_214
    assert world_parameters == 445_625_600
