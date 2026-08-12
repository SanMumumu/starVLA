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


def _tiny_model(
    interaction_mode: str,
    **mot_overrides,
) -> CausalDINOActionMoT:
    current_world_grid = mot_overrides.pop("current_world_grid", None)
    state_dim = mot_overrides.pop("state_dim", 3)
    mot_config = {
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
    }
    mot_config.update(mot_overrides)
    return CausalDINOActionMoT(
        planner_dim=12,
        world_dim=8,
        action_config={
            "action_horizon": 4,
            "action_dim": 3,
            "state_dim": state_dim,
        },
        mot_config=mot_config,
        current_world_grid=current_world_grid,
    )


def test_base_and_joint_masks_differ_only_for_action_reading_future() -> None:
    base = _tiny_model("base")
    joint = _tiny_model("joint")
    current = base.current_world_tokens
    future_end = current + base.world_tokens

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
        model.current_world_tokens + model.world_tokens,
        1,
        model.head_dim // 2,
    )
    assert model.action_frequencies.shape == (
        model.action_horizon,
        1,
        model.head_dim // 2,
    )
    assert model.world_frequencies.dtype == torch.complex128
    assert model.action_frequencies.dtype == torch.complex128
    assert model.world_frame_embedding is not None
    torch.testing.assert_close(
        model.world_frequencies[: model.world_tokens],
        model.world_frequencies[model.world_tokens :],
    )
    world_frequencies = model.world_frequencies.clone()
    action_frequencies = model.action_frequencies.clone()
    model.to(dtype=torch.bfloat16)
    assert model.world_input.weight.dtype == torch.bfloat16
    assert model.action_input.weight.dtype == torch.bfloat16
    assert model.action_output.weight.dtype == torch.bfloat16
    assert model.layers[0].action.norm3.weight.dtype == torch.bfloat16
    assert model.action_flow_dtype() == torch.bfloat16
    assert model.world_frequencies.is_complex()
    assert model.action_frequencies.is_complex()
    torch.testing.assert_close(model.world_frequencies, world_frequencies)
    torch.testing.assert_close(model.action_frequencies, action_frequencies)

    layerwise = _tiny_model(
        "joint",
        layerwise_planner_coupling=True,
    )
    assert layerwise.world_frequencies.dtype == torch.complex64
    assert layerwise.action_frequencies.dtype == torch.complex64


def test_fp32_action_shell_keeps_bf16_core_and_fp32_flow_after_global_cast() -> None:
    model = _tiny_model(
        "base",
        action_precision_mode="fp32_shell",
        action_prediction_type="velocity",
        action_velocity_target="noise_minus_clean",
        world_loss_weight=0.1,
    ).to(dtype=torch.bfloat16)

    action_layer = model.layers[0].action
    # Transformer compute stays bf16.
    assert action_layer.self_attn.q.weight.dtype == torch.bfloat16
    assert action_layer.self_attn.o.weight.dtype == torch.bfloat16
    assert action_layer.ffn[0].weight.dtype == torch.bfloat16
    assert model.action_context[0].weight.dtype == torch.bfloat16
    assert model.world_input.weight.dtype == torch.bfloat16
    # The configured action shell survives the same recursive cast used by
    # both the trainer and PolicyServerWrapper.
    assert model.action_input.weight.dtype == torch.float32
    assert model.action_time_embedding[0].weight.dtype == torch.float32
    assert model.action_time_projection[1].weight.dtype == torch.float32
    assert model.action_output.weight.dtype == torch.float32
    assert model.state_to_planner.weight.dtype == torch.float32
    assert action_layer.norm3.weight.dtype == torch.float32
    assert action_layer.self_attn.norm_q.weight.dtype == torch.float32
    assert action_layer.cross_attn.norm_k.weight.dtype == torch.float32
    assert model.action_flow_dtype() == torch.float32

    observed: dict[str, list[torch.dtype]] = {
        "action_input_in": [],
        "action_input_out": [],
        "core_q_in": [],
        "norm3_in": [],
        "norm3_out": [],
        "cross_q_in": [],
        "action_output_in": [],
        "action_output_out": [],
    }
    handles = [
        model.action_input.register_forward_hook(
            lambda _module, args, output: (
                observed["action_input_in"].append(args[0].dtype),
                observed["action_input_out"].append(output.dtype),
            ) and None
        ),
        action_layer.self_attn.q.register_forward_pre_hook(
            lambda _module, args: observed["core_q_in"].append(args[0].dtype)
        ),
        action_layer.norm3.register_forward_hook(
            lambda _module, args, output: (
                observed["norm3_in"].append(args[0].dtype),
                observed["norm3_out"].append(output.dtype),
            ) and None
        ),
        action_layer.cross_attn.q.register_forward_pre_hook(
            lambda _module, args: observed["cross_q_in"].append(args[0].dtype)
        ),
        model.action_output.register_forward_hook(
            lambda _module, args, output: (
                observed["action_output_in"].append(args[0].dtype),
                observed["action_output_out"].append(output.dtype),
            ) and None
        ),
    ]
    batch = 2
    inputs = {
        "action_plan": torch.randn(batch, 4, 12),
        "world_plan": torch.randn(batch, 4, 12),
        "current_world": torch.randn(batch, model.current_world_tokens, 8),
        "current_state": torch.randn(batch, 1, 3),
    }
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.forward_train(
            **inputs,
            target_action=torch.randn(batch, 4, 3),
            target_world=torch.randn(batch, model.world_tokens, 8),
            action_is_pad=torch.zeros(batch, 4, dtype=torch.bool),
            future_valid=torch.ones(batch),
        )
    assert output["action_loss_raw"].dtype == torch.float32
    output["loss"].backward()
    assert model.action_input.weight.grad.dtype == torch.float32
    assert model.action_output.weight.grad.dtype == torch.float32
    assert action_layer.norm3.weight.grad.dtype == torch.float32

    model.eval()
    action, future_world = model.sample(**inputs, seed=7)
    for handle in handles:
        handle.remove()
    assert future_world is None
    assert action.dtype == torch.float32
    assert set(observed["action_input_in"]) == {torch.float32}
    assert set(observed["action_input_out"]) == {torch.float32}
    assert set(observed["core_q_in"]) == {torch.bfloat16}
    assert set(observed["norm3_in"]) == {torch.float32}
    assert set(observed["norm3_out"]) == {torch.float32}
    assert set(observed["cross_q_in"]) == {torch.bfloat16}
    assert set(observed["action_output_in"]) == {torch.float32}
    assert set(observed["action_output_out"]) == {torch.float32}


def test_inherited_precision_keeps_norm3_in_core_dtype() -> None:
    model = _tiny_model("base").to(dtype=torch.bfloat16).eval()
    action_layer = model.layers[0].action
    observed: dict[str, list[torch.dtype]] = {
        "norm3_in": [],
        "norm3_out": [],
        "cross_q_in": [],
    }
    handles = [
        action_layer.norm3.register_forward_hook(
            lambda _module, args, output: (
                observed["norm3_in"].append(args[0].dtype),
                observed["norm3_out"].append(output.dtype),
            ) and None
        ),
        action_layer.cross_attn.q.register_forward_pre_hook(
            lambda _module, args: observed["cross_q_in"].append(args[0].dtype)
        ),
    ]
    batch = 2
    action, future_world = model.sample(
        action_plan=torch.randn(batch, 4, 12),
        world_plan=torch.randn(batch, 4, 12),
        current_world=torch.randn(batch, model.current_world_tokens, 8),
        current_state=torch.randn(batch, 1, 3),
        seed=11,
    )
    for handle in handles:
        handle.remove()

    assert future_world is None
    assert action.dtype == torch.bfloat16
    assert set(observed["norm3_in"]) == {torch.bfloat16}
    assert set(observed["norm3_out"]) == {torch.bfloat16}
    assert set(observed["cross_q_in"]) == {torch.bfloat16}


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is required for the fp32-shell regression",
)
def test_fp32_action_shell_cuda_base_inference_dtype_boundary() -> None:
    device = torch.device("cuda")
    model = _tiny_model(
        "base",
        action_precision_mode="fp32_shell",
        action_prediction_type="velocity",
        action_velocity_target="noise_minus_clean",
    ).to(device=device, dtype=torch.bfloat16).eval()
    batch = 2
    with torch.inference_mode():
        action, future_world = model.sample(
            action_plan=torch.randn(batch, 4, 12, device=device),
            world_plan=torch.randn(batch, 4, 12, device=device),
            current_world=torch.randn(
                batch,
                model.current_world_tokens,
                8,
                device=device,
            ),
            current_state=torch.randn(batch, 1, 3, device=device),
            seed=13,
        )

    assert future_world is None
    assert action.dtype == torch.float32
    assert action.is_cuda
    assert torch.isfinite(action).all()


def test_action_precision_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="action_precision_mode"):
        _tiny_model("base", action_precision_mode="fp16_shell")


def test_tiny_train_and_sample_paths_share_shapes_state_and_dtype() -> None:
    model = _tiny_model(
        "base",
        action_prediction_type="velocity",
        action_velocity_target="clean_minus_noise",
        repeated_diffusion_steps=2,
        world_loss_weight=0.1,
    )
    batch = 2
    inputs = {
        "action_plan": torch.randn(batch, 4, 12),
        "world_plan": torch.randn(batch, 4, 12),
        "current_world": torch.randn(batch, model.current_world_tokens, 8),
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
    assert output["action_objective"].requires_grad
    assert output["world_objective"].requires_grad
    torch.testing.assert_close(
        output["loss"],
        output["action_objective"] + output["world_objective"],
    )
    torch.testing.assert_close(
        output["action_objective"].detach(),
        output["action_loss_raw"],
    )
    torch.testing.assert_close(
        output["world_objective"].detach(),
        0.1 * output["world_loss_raw"],
    )
    output["loss"].backward()

    model.eval()
    action, future_world = model.sample(**inputs, seed=7)
    assert action.shape == (batch, 4, 3)
    assert future_world is None
    assert action.dtype == model.action_input.weight.dtype


def test_layerwise_plans_map_one_to_one_without_state_conditioning() -> None:
    model = _tiny_model(
        "base",
        layerwise_planner_coupling=True,
        state_dim=0,
    )
    assert model._legacy_world_condition_on_state is False
    assert model.state_to_planner is None
    assert model.checkpoint_contract_version == "layerwise_query_only_world_v2"
    batch, queries = 2, 4
    action_plans = [
        torch.randn(batch, queries, 12)
        for _ in range(model.num_layers)
    ]
    world_plans = [
        torch.randn(batch, queries, 12)
        for _ in range(model.num_layers)
    ]
    action_layers, world_layers, _ = model._prepare_conditions(
        action_plan=action_plans,
        world_plan=world_plans,
        current_world=torch.randn(batch, model.current_world_tokens, 8),
        current_state=None,
    )
    assert len(action_layers) == len(world_layers) == model.num_layers
    assert all(plan.shape[1] == queries for plan in action_layers)
    assert all(plan.shape[1] == queries for plan in world_layers)
    assert isinstance(model.action_context, torch.nn.ModuleList)
    assert isinstance(model.world_context, torch.nn.ModuleList)
    assert model.action_context[0] is not model.action_context[1]
    assert model.world_context[0] is not model.world_context[1]

    output = model.forward_train(
        action_plan=action_plans,
        world_plan=world_plans,
        current_world=torch.randn(batch, model.current_world_tokens, 8),
        current_state=None,
        target_action=torch.randn(batch, 4, 3),
        target_world=torch.randn(batch, model.world_tokens, 8),
        action_is_pad=torch.zeros(batch, 4, dtype=torch.bool),
        future_valid=torch.ones(batch),
    )
    output["loss"].backward()
    assert all(projector[0].weight.grad is not None for projector in model.action_context)
    assert all(projector[0].weight.grad is not None for projector in model.world_context)
    assert not any("state_to_planner" in key for key in model.state_dict())


def test_layerwise_state_conditions_action_but_not_world() -> None:
    model = _tiny_model(
        "base",
        layerwise_planner_coupling=True,
        state_dim=3,
    )
    assert model._legacy_world_condition_on_state is False
    assert model.state_to_planner is not None
    batch, queries = 2, 4
    action_plans = [
        torch.randn(batch, queries, 12)
        for _ in range(model.num_layers)
    ]
    world_plans = [
        torch.randn(batch, queries, 12)
        for _ in range(model.num_layers)
    ]
    action_layers, world_layers, _ = model._prepare_conditions(
        action_plan=action_plans,
        world_plan=world_plans,
        current_world=torch.randn(batch, model.current_world_tokens, 8),
        current_state=torch.randn(batch, 1, 3),
    )
    assert all(plan.shape[1] == queries + 1 for plan in action_layers)
    assert all(plan.shape[1] == queries for plan in world_layers)


def test_legacy_causal_checkpoint_defaults_remain_velocity_repeat_one() -> None:
    model = _tiny_model("base")
    assert model.action_prediction_type == "velocity"
    assert model.action_velocity_target == "noise_minus_clean"
    assert model.repeated_diffusion_steps == 1
    assert model._legacy_world_condition_on_state is True
    assert (
        model.checkpoint_contract_version
        == "legacy_shared_context_state_world_v1"
    )
    assert not any(
        "action_dino" in key
        for key in model.state_dict()
    )
    _tiny_model("base").load_state_dict(
        model.state_dict(),
        strict=True,
    )

    batch, queries = 2, 4
    action_layers, world_layers, _ = model._prepare_conditions(
        action_plan=torch.randn(batch, queries, 12),
        world_plan=torch.randn(batch, queries, 12),
        current_world=torch.randn(batch, model.current_world_tokens, 8),
        current_state=torch.randn(batch, 1, 3),
    )
    assert all(plan.shape[1] == queries + 1 for plan in action_layers)
    assert all(plan.shape[1] == queries + 1 for plan in world_layers)


def test_multires_current_prefix_is_shared_by_world_and_action_and_cached() -> None:
    model = _tiny_model(
        "base",
        layerwise_planner_coupling=True,
        current_world_grid=(4, 6),
        action_prediction_type="velocity",
        action_velocity_target="clean_minus_noise",
        world_loss_weight=0.1,
    )
    assert model.multires_world_input is True
    assert model.current_world_tokens == 24
    assert model.world_tokens == 6
    assert (
        model.checkpoint_contract_version
        == "layerwise_query_only_multires_world_v3"
    )
    assert not any("action_dino" in key for key in model.state_dict())
    physical_tokens = (
        model.current_world_tokens
        + model.world_tokens
        + model.action_horizon
    )
    assert model.physical_attention_mask.shape == (
        physical_tokens,
        physical_tokens,
    )
    current_end = model.current_world_tokens
    future_end = current_end + model.world_tokens
    assert not model.physical_attention_mask[
        :current_end,
        current_end:future_end,
    ].any()
    assert model.physical_attention_mask[
        current_end:future_end,
        :future_end,
    ].all()
    assert model.physical_attention_mask[
        future_end:,
        :current_end,
    ].all()
    assert not model.physical_attention_mask[
        future_end:,
        current_end:future_end,
    ].any()

    # Resolution changes only sequence geometry and non-persistent masks/RoPE.
    # No new checkpoint parameters are introduced.
    same_depth_v2 = _tiny_model(
        "base",
        layerwise_planner_coupling=True,
    )
    model.load_state_dict(same_depth_v2.state_dict(), strict=True)

    batch = 2
    action_plan = [
        torch.randn(batch, 4, 12)
        for _ in range(model.num_layers)
    ]
    world_plan = [
        torch.randn(batch, 4, 12)
        for _ in range(model.num_layers)
    ]
    current_world = torch.randn(
        batch,
        model.current_world_tokens,
        8,
        requires_grad=True,
    )
    current_state = torch.randn(batch, 1, 3)
    common = {
        "action_plan": action_plan,
        "world_plan": world_plan,
        "current_world": current_world,
        "current_state": current_state,
    }
    output = model.forward_train(
        **common,
        target_action=torch.randn(batch, 4, 3),
        target_world=torch.randn(batch, model.world_tokens, 8),
        action_is_pad=torch.zeros(batch, 4, dtype=torch.bool),
        future_valid=torch.ones(batch),
    )
    world_current_grad = torch.autograd.grad(
        output["world_objective"],
        current_world,
        retain_graph=True,
    )[0]
    action_current_grad = torch.autograd.grad(
        output["action_objective"],
        current_world,
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(world_current_grad) > 0
    assert torch.count_nonzero(action_current_grad) > 0
    output["loss"].backward()

    model.eval()
    cache = model._prepare_action_base_cache(**common)
    assert cache.world_keys[0].shape == (
        batch,
        model.current_world_tokens,
        model.attention_inner_dim,
    )
    noisy_action = torch.randn(batch, 4, 3)
    action_time = torch.full((batch,), 500.0)
    cached_prediction = model._predict_action_base(
        **common,
        noisy_action=noisy_action,
        action_time=action_time,
        cache=cache,
    )
    uncached_prediction = model._predict_action_base(
        **common,
        noisy_action=noisy_action,
        action_time=action_time,
    )
    torch.testing.assert_close(cached_prediction, uncached_prediction)

    noisy_world = torch.randn(batch, model.world_tokens, 8)
    prediction, world = model._predict(
        **common,
        noisy_action=noisy_action,
        noisy_world=noisy_world,
        action_time=action_time,
        world_time=action_time,
        action_is_pad=None,
    )
    assert prediction.shape == (batch, model.action_horizon, model.action_dim)
    assert world.shape == (batch, model.world_tokens, model.world_dim)


def test_multires_current_and_future_shapes_are_independently_validated() -> None:
    model = _tiny_model(
        "base",
        current_world_grid=(4, 6),
    )
    common = {
        "action_plan": torch.randn(1, 4, 12),
        "world_plan": torch.randn(1, 4, 12),
        "current_state": torch.randn(1, 1, 3),
    }
    with pytest.raises(ValueError, match="current DINO must have shape"):
        model.sample(
            **common,
            current_world=torch.randn(1, model.current_world_tokens - 1, 8),
            num_inference_steps=1,
        )
    with pytest.raises(ValueError, match="target future DINO must have shape"):
        model.forward_train(
            **common,
            current_world=torch.randn(1, model.current_world_tokens, 8),
            target_action=torch.randn(1, 4, 3),
            target_world=torch.randn(1, model.current_world_tokens, 8),
            action_is_pad=None,
            future_valid=torch.ones(1),
        )


def test_jit_x_prediction_converts_clean_action_to_scheduler_velocity() -> None:
    model = _tiny_model(
        "base",
        action_prediction_type="jit_x",
        jit_t_eps=0.05,
    )
    clean = torch.tensor([[[0.25, -0.50, 0.75]]])
    noise = torch.tensor([[[-0.75, 0.50, 0.25]]])
    sigma = 0.4
    noisy = (1.0 - sigma) * clean + sigma * noise
    timestep = torch.tensor(
        [sigma * model.train_action_scheduler.num_train_timesteps]
    )

    velocity = model._action_prediction_to_velocity(
        clean,
        noisy,
        timestep,
        scheduler=model.train_action_scheduler,
    )
    torch.testing.assert_close(velocity, noise - clean)


def test_repeated_diffusion_steps_repeat_conditions_with_independent_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model(
        "joint",
        action_prediction_type="velocity",
        action_velocity_target="clean_minus_noise",
        repeated_diffusion_steps=2,
        world_loss_weight=0.1,
    )
    batch = 2
    action_plan = torch.randn(batch, 4, 12)
    world_plan = torch.randn(batch, 4, 12)
    current_world = torch.randn(batch, model.current_world_tokens, 8)
    current_state = torch.randn(batch, 1, 3)
    target_action = torch.randn(batch, 4, 3)
    target_world = torch.randn(batch, model.world_tokens, 8)
    captured = {}

    def fixed_action_time(batch_size, *, device, dtype):
        assert batch_size == 2 * batch
        return torch.full(
            (batch_size,),
            500.0,
            device=device,
            dtype=dtype,
        )

    def fixed_world_time(batch_size, *, device, dtype):
        assert batch_size == 2 * batch
        return torch.full(
            (batch_size,),
            700.0,
            device=device,
            dtype=dtype,
        )

    repeated_target_action = target_action.repeat(2, 1, 1)

    def predict(**kwargs):
        captured.update(kwargs)
        # At sigma=0.5, recover the exact clean-minus-noise velocity without
        # reaching into RNG state.
        exact_velocity = (
            repeated_target_action.to(
                device=kwargs["noisy_action"].device,
                dtype=kwargs["noisy_action"].dtype,
            )
            - kwargs["noisy_action"]
        ) / 0.5
        return (
            exact_velocity,
            torch.zeros_like(kwargs["noisy_world"]),
        )

    monkeypatch.setattr(
        model.train_action_scheduler,
        "sample_training_t",
        fixed_action_time,
    )
    monkeypatch.setattr(
        model.train_world_scheduler,
        "sample_training_t",
        fixed_world_time,
    )
    monkeypatch.setattr(model, "_predict", predict)

    output = model.forward_train(
        action_plan=action_plan,
        world_plan=world_plan,
        current_world=current_world,
        target_action=target_action,
        target_world=target_world,
        action_is_pad=torch.zeros(batch, 4, dtype=torch.bool),
        future_valid=torch.ones(batch),
        current_state=current_state,
    )

    assert captured["noisy_action"].shape[0] == 2 * batch
    torch.testing.assert_close(
        captured["action_plan"],
        action_plan.repeat(2, 1, 1),
    )
    torch.testing.assert_close(
        captured["world_plan"],
        world_plan.repeat(2, 1, 1),
    )
    assert not torch.equal(
        captured["noisy_action"][:batch],
        captured["noisy_action"][batch:],
    )
    torch.testing.assert_close(
        output["action_loss_raw"],
        torch.zeros_like(output["action_loss_raw"]),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert model.world_loss_weight == pytest.approx(0.1)


def test_clean_minus_noise_sampling_flips_velocity_for_sigma_euler_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model(
        "base",
        action_prediction_type="velocity",
        action_velocity_target="clean_minus_noise",
    ).eval()
    batch = 1
    clean_action = torch.randn(batch, model.action_horizon, model.action_dim)

    def predict_denoising_velocity(**kwargs):
        clean = clean_action.to(
            device=kwargs["noisy_action"].device,
            dtype=kwargs["noisy_action"].dtype,
        )
        sigma = kwargs["action_time"].to(clean.dtype) / float(
            model.infer_action_scheduler.num_train_timesteps
        )
        return (clean - kwargs["noisy_action"]) / sigma[:, None, None]

    monkeypatch.setattr(
        model,
        "_predict_action_base",
        predict_denoising_velocity,
    )
    action, future_world = model.sample(
        action_plan=torch.randn(batch, 4, 12),
        world_plan=torch.randn(batch, 4, 12),
        current_world=torch.randn(batch, model.current_world_tokens, 8),
        current_state=torch.randn(batch, 1, 3),
        seed=7,
        num_inference_steps=1,
    )

    assert future_world is None
    torch.testing.assert_close(action, clean_action)


def test_base_sample_skips_future_world_prediction_and_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model("base").eval()
    batch = 1
    inputs = {
        "action_plan": torch.randn(batch, 4, 12),
        "world_plan": torch.randn(batch, 4, 12),
        "current_world": torch.randn(batch, model.current_world_tokens, 8),
        "current_state": torch.randn(batch, 1, 3),
    }
    action_predict_calls = 0
    world_attention_calls = 0
    original_action_predict = model._predict_action_base

    def counted_action_predict(**kwargs):
        nonlocal action_predict_calls
        action_predict_calls += 1
        return original_action_predict(**kwargs)

    for layer in model.layers:
        original_world_attention = layer.world.attention_io

        def counted_world_attention(
            *args,
            _original=original_world_attention,
            **kwargs,
        ):
            nonlocal world_attention_calls
            world_attention_calls += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(
            layer.world,
            "attention_io",
            counted_world_attention,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("base inference must not execute the future-world path")

    monkeypatch.setattr(model, "_predict_action_base", counted_action_predict)
    monkeypatch.setattr(model, "_predict", forbidden)
    monkeypatch.setattr(
        model.infer_world_scheduler,
        "inference_schedule",
        forbidden,
    )
    monkeypatch.setattr(model.world_output, "forward", forbidden)

    action, future_world = model.sample(
        **inputs,
        seed=7,
        num_inference_steps=3,
    )
    assert action.shape == (batch, 4, 3)
    assert future_world is None
    assert action_predict_calls == 3
    # Current-world features are invariant across action denoising and are
    # therefore evaluated once per physical layer, not once per denoising step.
    assert world_attention_calls == model.num_layers


@pytest.mark.parametrize("layerwise_coupling", [False, True])
def test_base_action_only_sample_matches_full_masked_world_path(
    layerwise_coupling: bool,
) -> None:
    model = _tiny_model(
        "base",
        layerwise_planner_coupling=layerwise_coupling,
    ).eval()
    batch = 1
    action_plan = torch.randn(batch, 4, 12)
    world_plan = torch.randn(batch, 4, 12)
    if layerwise_coupling:
        action_plan = [
            torch.randn(batch, 4, 12)
            for _ in range(model.num_layers)
        ]
        world_plan = [
            torch.randn(batch, 4, 12)
            for _ in range(model.num_layers)
        ]
    inputs = {
        "action_plan": action_plan,
        "world_plan": world_plan,
        "current_world": torch.randn(batch, model.current_world_tokens, 8),
        "current_state": torch.randn(batch, 1, 3),
    }
    inference_steps = 3
    seed = 7
    action_only, future_world = model.sample(
        **inputs,
        seed=seed,
        num_inference_steps=inference_steps,
    )
    assert future_world is None

    device, dtype = model._device_dtype()
    action_generator = torch.Generator(device="cpu")
    world_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(seed)
    world_generator.manual_seed(seed)
    action = torch.randn(
        batch,
        model.action_horizon,
        model.action_dim,
        generator=action_generator,
    ).to(device=device, dtype=dtype)
    noisy_world = torch.randn(
        inputs["current_world"].shape,
        generator=world_generator,
    ).to(device=device, dtype=dtype)
    action_times, action_deltas = model.infer_action_scheduler.inference_schedule(
        inference_steps,
        device=device,
        dtype=dtype,
    )
    world_times, world_deltas = model.infer_world_scheduler.inference_schedule(
        inference_steps,
        device=device,
        dtype=dtype,
    )
    for action_time, action_delta, world_time, world_delta in zip(
        action_times,
        action_deltas,
        world_times,
        world_deltas,
    ):
        action_prediction, world_prediction = model._predict(
            **inputs,
            noisy_action=action,
            noisy_world=noisy_world,
            action_time=action_time.expand(batch),
            world_time=world_time.expand(batch),
            action_is_pad=None,
        )
        action = model.infer_action_scheduler.step(
            action_prediction,
            action_delta,
            action,
        ).to(dtype)
        noisy_world = model.infer_world_scheduler.step(
            world_prediction,
            world_delta,
            noisy_world,
        ).to(dtype)

    torch.testing.assert_close(action_only, action, rtol=1.0e-5, atol=1.0e-6)


def test_sample_runtime_override_is_joint_steps_not_action_plus_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_model("joint").eval()
    batch = 1
    inputs = {
        "action_plan": torch.randn(batch, 4, 12),
        "world_plan": torch.randn(batch, 4, 12),
        "current_world": torch.randn(batch, model.current_world_tokens, 8),
        "current_state": torch.randn(batch, 1, 3),
    }
    predict_calls = 0
    original_predict = model._predict

    def counted_predict(**kwargs):
        nonlocal predict_calls
        predict_calls += 1
        return original_predict(**kwargs)

    monkeypatch.setattr(model, "_predict", counted_predict)
    model.sample(**inputs, seed=7, num_inference_steps=3)

    # Each joint call advances both action and world once.
    assert predict_calls == 3


def test_rtc_prefix_weights_match_lerobot_schedules() -> None:
    linear = CausalDINOActionMoT.rtc_prefix_weights(
        inference_delay=1,
        execution_horizon=4,
        total_horizon=6,
        schedule="linear",
    )
    torch.testing.assert_close(
        linear,
        torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0, 0.0]),
    )
    exp = CausalDINOActionMoT.rtc_prefix_weights(
        inference_delay=0,
        execution_horizon=4,
        total_horizon=6,
        schedule="exp",
    )
    assert torch.all(exp[:4] > 0)
    assert torch.all(exp[:-1] >= exp[1:])
    assert torch.equal(exp[4:], torch.zeros(2))


def test_rtc_guidance_moves_scheduler_step_toward_previous_tail() -> None:
    model = _tiny_model("base").eval()
    noisy = torch.ones(1, model.action_horizon, model.action_dim)
    target = torch.zeros_like(noisy)
    weights = torch.zeros(1, model.action_horizon, 1)
    weights[:, :2] = 1.0
    action_time = torch.full(
        (1,),
        0.5 * model.infer_action_scheduler.num_train_timesteps,
    )

    guided, auxiliary = model._rtc_guided_action_velocity(
        noisy_action=noisy,
        action_time=action_time,
        target=target,
        weights=weights,
        max_guidance_weight=10.0,
        velocity_fn=lambda value: (value * 0.0, None),
    )
    stepped = model.infer_action_scheduler.step(
        guided,
        torch.tensor(-0.1),
        noisy,
    )

    assert auxiliary is None
    assert torch.linalg.vector_norm(stepped[:, :2]) < torch.linalg.vector_norm(
        noisy[:, :2]
    )
    torch.testing.assert_close(stepped[:, 2:], noisy[:, 2:])


@pytest.mark.parametrize("interaction_mode", ["base", "joint"])
def test_rtc_sample_accepts_batched_prefix_lengths(
    interaction_mode: str,
) -> None:
    model = _tiny_model(interaction_mode).eval()
    batch = 2
    sample_inputs = {
        "action_plan": torch.randn(batch, 4, 12),
        "world_plan": torch.randn(batch, 4, 12),
        "current_world": torch.randn(batch, model.current_world_tokens, 8),
        "current_state": torch.randn(batch, 1, 3),
    }
    baseline, _ = model.sample(
        **sample_inputs,
        seed=7,
        num_inference_steps=1,
    )
    action, future_world = model.sample(
        **sample_inputs,
        seed=7,
        num_inference_steps=1,
        prev_action_chunk_normalized=torch.randn(batch, 4, 3),
        rtc_prefix_lengths=torch.tensor([2, 0]),
        inference_delay=0,
        execution_horizon=2,
        prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
    )
    assert action.shape == (batch, 4, 3)
    assert (future_world is None) == (interaction_mode == "base")
    torch.testing.assert_close(action[1], baseline[1])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_sample_rejects_nonpositive_runtime_steps() -> None:
    model = _tiny_model("joint").eval()
    with pytest.raises(ValueError, match="must be positive"):
        model.sample(
            action_plan=torch.randn(1, 4, 12),
            world_plan=torch.randn(1, 4, 12),
            current_world=torch.randn(1, model.current_world_tokens, 8),
            current_state=torch.randn(1, 1, 3),
            num_inference_steps=0,
        )


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
    ("filename", "run_id", "text_enabled", "fp32_shell", "action_eval"),
    [
        (
            "released_rynn50k/rynn_base_h25_50k.yaml",
            "starvla_rynnbrain11_robodojo_dino_mot_base_h25_50k_v1",
            False,
            False,
            True,
        ),
        (
            "released_rynn50k/rynn_base_text_h25_mem_50k.yaml",
            "starvla_rynnbrain11_robodojo_dino_mot_base_text_h25_eventmem_ntp_nohist_fp32_50k",
            True,
            True,
            False,
        ),
        (
            "released_rynn50k/rynn_base_text_h25_mem_bf16_50k.yaml",
            "starvla_rynnbrain11_robodojo_dino_mot_base_text_h25_eventmem_ntp_nohist_bf16_50k",
            True,
            False,
            False,
        ),
    ],
)
def test_training_configs_pin_the_physical_contract(
    filename: str,
    run_id: str,
    text_enabled: bool,
    fp32_shell: bool,
    action_eval: bool,
) -> None:
    path = TRAIN_CONFIG_DIR / filename
    config_text = path.read_text(encoding="utf-8")
    config = OmegaConf.load(path)
    physical = config.framework.world_action_mot

    assert config.run_id == run_id
    assert "fastwam" not in config_text.lower()
    assert physical.architecture == "causal_dino_mot"
    assert physical.interaction_mode == "base"
    assert physical.world_attention_mask_mode == "first_frame_causal"
    assert (physical.world_hidden_size, physical.action_hidden_size) == (512, 1024)
    assert (physical.world_ffn_dim, physical.action_ffn_dim) == (2048, 4096)
    expected_layers = 30
    assert (
        physical.num_layers,
        physical.num_attention_heads,
        physical.attention_head_dim,
    ) == (expected_layers, 24, 128)
    assert physical.layerwise_planner_coupling is False
    assert "world_condition_on_state" not in physical
    assert config.framework.action_model.state_dim == 14
    assert config.framework.action_model.action_horizon == 25
    assert config.framework.planner.num_action_queries == 25
    assert config.datasets.vla_data.include_state is True
    assert config.framework.qwenvl.truncate_vlm_layers == 0
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
    assert physical.num_inference_timesteps == 20
    assert physical.action_prediction_type == "velocity"
    assert physical.action_velocity_target == "noise_minus_clean"
    assert physical.repeated_diffusion_steps == 1
    assert physical.action_loss_weight == pytest.approx(1.0)
    assert physical.world_loss_weight == pytest.approx(1.0)
    assert physical.text_loss_weight == pytest.approx(0.005 if text_enabled else 0.0)
    assert config.trainer.logging_frequency == 200
    assert bool(config.trainer.action_eval_enabled) is action_eval
    assert config.framework.dino.embed_dim == 768
    assert config.framework.dino.dino_pool == 2
    assert config.framework.dino.current_dino_pool is None
    assert config.framework.dino.get("action_dino_pool", None) is None
    assert config.datasets.vla_data.image_layout == "tri_view_composite"
    assert config.framework.qwenvl.attn_implementation == "sdpa"
    assert config.framework.qwenvl.require_attn_implementation is True
    assert str(config.framework.qwenvl.base_vlm).rstrip("/").endswith(
        "/rynnbrain1.1-2B"
    )
    assert not config.framework.qwenvl.enable_thinking
    assert bool(config.framework.planner.text_supervision.enabled) is text_enabled
    assert str(physical.get("action_precision_mode", "inherit")) == (
        "fp32_shell" if fp32_shell else "inherit"
    )


def test_shared_vlm_interface_gradient_metrics_are_weighted_and_non_mutating() -> None:
    from starVLA.training.train_starvla import VLATrainer

    vlm_source = torch.tensor(
        [1.0, 2.0, -0.5, 0.25, 2.0],
        requires_grad=True,
    )
    world_queries = vlm_source[:2] * 1.0
    action_queries = vlm_source[2:] * 1.0
    # These stand in for already-weighted physical objectives.  The world
    # objective has no ACTION-query dependency, matching the causal contract.
    action_objective = 2.0 * world_queries.sum() + 3.0 * action_queries.sum()
    world_objective = -world_queries[0] + world_queries[1]

    metrics = VLATrainer._shared_vlm_interface_gradient_metrics(
        action_objective,
        world_objective,
        (world_queries, action_queries),
    )

    # The diagnostic stops at query outputs and therefore never populates or
    # traverses into the upstream shared-VLM leaf.
    assert vlm_source.grad is None
    assert set(metrics) == {
        "train/shared_vlm_interface_grad_norm_action",
        "train/shared_vlm_interface_grad_norm_world",
    }
    assert metrics["train/shared_vlm_interface_grad_norm_action"] == pytest.approx(
        35.0**0.5
    )
    assert metrics["train/shared_vlm_interface_grad_norm_world"] == pytest.approx(
        2.0**0.5
    )

    # retain_graph=True keeps the real optimizer backward valid, and the
    # diagnostic itself never populates leaf .grad fields.
    (action_objective + world_objective).backward()
    torch.testing.assert_close(
        vlm_source.grad,
        torch.tensor([1.0, 3.0, 3.0, 3.0, 3.0]),
    )

    logged = VLATrainer._build_native_loss_metrics(
        {
            "mot_action_loss_raw": torch.tensor(0.2),
            "mot_world_loss_raw": torch.tensor(0.4),
            "mot_action_loss_weighted": torch.tensor(0.2),
            "mot_world_loss_weighted": torch.tensor(0.04),
            "mot_world_loss_weight": torch.tensor(0.1),
            "mot_text_loss_raw": torch.tensor(0.0),
            "mot_text_loss_weighted": torch.tensor(0.0),
            "mot_text_loss_weight": torch.tensor(0.005),
            "mot_text_sample_count": torch.tensor(0.0),
        },
        torch.tensor(0.24),
    )
    assert logged["train/action_loss"] == pytest.approx(0.2)
    assert logged["train/world_loss"] == pytest.approx(0.04)
    assert logged["train/text_loss"] == pytest.approx(0.0)
    assert logged["train/action_loss_weighted"] == pytest.approx(0.2)
    assert logged["train/world_loss_weighted"] == pytest.approx(0.04)
    assert logged["train/world_loss_weight"] == pytest.approx(0.1)
    assert logged["train/text_loss_weight"] == pytest.approx(0.005)


def test_shared_vlm_gradient_probe_reuses_real_backward_with_one_aux_grad(
    monkeypatch,
) -> None:
    from starVLA.training.train_starvla import VLATrainer

    vlm_source = torch.tensor(
        [1.0, 2.0, -0.5, 0.25, 2.0],
        requires_grad=True,
    )
    world_queries = vlm_source[:2] * 1.0
    action_queries = vlm_source[2:] * 1.0
    action_objective = 2.0 * world_queries.sum() + 3.0 * action_queries.sum()
    world_objective = -world_queries[0] + world_queries[1]

    original_grad = torch.autograd.grad
    calls = 0

    def counted_grad(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    probe = VLATrainer._prepare_shared_vlm_interface_gradient_probe(
        action_objective,
        world_objective,
        (world_queries, action_queries),
    )
    (action_objective + world_objective).backward()
    metrics = VLATrainer._finalize_shared_vlm_interface_gradient_probe(
        probe,
    )

    assert calls == 1
    assert metrics["train/shared_vlm_interface_grad_norm_action"] == pytest.approx(
        35.0**0.5
    )
    assert metrics["train/shared_vlm_interface_grad_norm_world"] == pytest.approx(
        2.0**0.5
    )
    torch.testing.assert_close(
        vlm_source.grad,
        torch.tensor([1.0, 3.0, 3.0, 3.0, 3.0]),
    )
    assert world_queries.grad is None
    assert action_queries.grad is None


def test_legacy_action_eval_does_not_override_framework_inference_steps(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from starVLA.training.train_starvla import VLATrainer

    class _EvalModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.kwargs = None

        def predict_action(self, **kwargs):
            self.kwargs = kwargs
            return {"normalized_actions": torch.zeros(1, 1, 1).numpy()}

    model = _EvalModel().train()
    examples = [{"action": torch.zeros(1, 1).numpy()}]
    trainer = VLATrainer.__new__(VLATrainer)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(
        is_main_process=False,
        unwrap_model=lambda wrapped: wrapped,
    )
    trainer._get_next_batch = lambda: examples
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    VLATrainer.eval_action_model(trainer, {})

    assert model.kwargs == {"examples": examples, "use_ddim": True}
    assert model.training


def test_physical_objectives_allow_diagnostics_before_the_real_backward() -> None:
    from starVLA.training.train_starvla import VLATrainer

    model = _tiny_model(
        "base",
        action_prediction_type="velocity",
        action_velocity_target="clean_minus_noise",
        world_loss_weight=0.1,
    )
    batch = 2
    world_plan_source = torch.randn(batch, 4, 12, requires_grad=True)
    action_plan_source = torch.randn(batch, 4, 12, requires_grad=True)
    world_plan = world_plan_source * 1.0
    action_plan = action_plan_source * 1.0
    output = model.forward_train(
        action_plan=action_plan,
        world_plan=world_plan,
        current_world=torch.randn(batch, model.current_world_tokens, 8),
        current_state=torch.randn(batch, 1, 3),
        target_action=torch.randn(batch, 4, 3),
        target_world=torch.randn(batch, model.world_tokens, 8),
        action_is_pad=torch.zeros(batch, 4, dtype=torch.bool),
        future_valid=torch.ones(batch),
    )

    metrics = VLATrainer._shared_vlm_interface_gradient_metrics(
        output["action_objective"],
        output["world_objective"],
        (world_plan, action_plan),
    )
    assert set(metrics) == {
        "train/shared_vlm_interface_grad_norm_action",
        "train/shared_vlm_interface_grad_norm_world",
    }
    assert all(parameter.grad is None for parameter in model.parameters())
    assert world_plan_source.grad is None
    assert action_plan_source.grad is None

    probe = VLATrainer._prepare_shared_vlm_interface_gradient_probe(
        output["action_objective"],
        output["world_objective"],
        (world_plan, action_plan),
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    output["loss"].backward()
    optimized_metrics = VLATrainer._finalize_shared_vlm_interface_gradient_probe(
        probe,
    )
    for key, expected in metrics.items():
        assert optimized_metrics[key] == pytest.approx(expected, rel=1.0e-5, abs=1.0e-7)
    assert world_plan.grad is None
    assert action_plan.grad is None
    assert world_plan_source.grad is not None
    assert action_plan_source.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters())
