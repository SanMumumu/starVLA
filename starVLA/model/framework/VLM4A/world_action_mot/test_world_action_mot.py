import torch

from .model import (
    ConditionCrossAttentionBlock,
    MoTExpertBlock,
    WorldActionMoT,
    _masked_action_mse,
)


def _model(state_dim=0, **mot_overrides):
    mot_config = {
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "attention_pattern": "alternating_condition_joint",
        "action_mlp_ratio": 2,
        "world_mlp_ratio": 2,
        "dropout": 0,
        "max_world_tokens": 8,
        "time_frequency_dim": 16,
        "enable_gradient_checkpointing": False,
        "num_inference_timesteps": 2,
        "action_prediction_type": "jit_x",
        "jit_t_eps": 0.05,
        "flow_time_sampling": "gr00t",
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
    }
    mot_config.update(mot_overrides)
    return WorldActionMoT(
        planner_dim=24,
        world_dim=12,
        action_config={
            "action_horizon": 4,
            "action_dim": 3,
            "state_dim": state_dim,
        },
        mot_config=mot_config,
    )


def test_experts_own_independent_qkv():
    block = MoTExpertBlock(32, 4, 2, 2, 0)
    assert block.action_attention.q_proj is not block.world_attention.q_proj
    assert block.action_attention.k_proj is not block.world_attention.k_proj
    assert block.action_attention.v_proj is not block.world_attention.v_proj
    action_parameter_ids = {id(p) for p in block.action_attention.parameters()}
    world_parameter_ids = {id(p) for p in block.world_attention.parameters()}
    assert action_parameter_ids.isdisjoint(world_parameter_ids)


def test_fastwam_attention_pattern_alternates_cross_and_joint_blocks():
    model = _model(num_layers=4)
    assert isinstance(model.layers[0], ConditionCrossAttentionBlock)
    assert isinstance(model.layers[1], MoTExpertBlock)
    assert isinstance(model.layers[2], ConditionCrossAttentionBlock)
    assert isinstance(model.layers[3], MoTExpertBlock)
    cross = model.layers[0]
    assert cross.action_attention.k_proj is not cross.world_attention.k_proj
    assert cross.action_attention.v_proj is not cross.world_attention.v_proj


def test_full_attention_only_masks_padded_action_keys():
    model = _model()
    result = model._tokens(
        action_plan=torch.randn(1, 4, 24),
        world_plan=torch.randn(1, 3, 24),
        current_world=torch.randn(1, 6, 12),
        noisy_action=torch.randn(1, 4, 3),
        noisy_world=torch.randn(1, 6, 12),
        action_time=torch.tensor([0.5]),
        world_time=torch.tensor([0.5]),
        action_is_pad=torch.tensor([[False, False, True, True]]),
    )
    (
        action_hidden,
        world_hidden,
        action_memory,
        world_memory,
        bias,
        *_,
    ) = result
    assert action_hidden.shape[1] == 4
    assert world_hidden.shape[1] == 6
    assert action_memory.shape[1] == 4
    assert world_memory.shape[1] == 3 + 6
    assert bias.shape == (
        1,
        1,
        1,
        action_hidden.shape[1] + world_hidden.shape[1],
    )
    assert torch.isfinite(bias[0, 0, 0, action_hidden.shape[1] :]).all()
    assert torch.isfinite(bias[0, 0, 0, :2]).all()
    assert torch.isneginf(bias[0, 0, 0, 2:4]).all()


def test_conditions_are_read_only_cross_attention_memories():
    model = _model(state_dim=3)
    action_plan = torch.randn(2, 4, 24)
    world_plan = torch.randn(2, 3, 24)
    current_world = torch.randn(2, 6, 12)
    noisy_action = torch.randn(2, 4, 3)
    noisy_world = torch.randn(2, 6, 12)
    current_state = torch.randn(2, 1, 3)
    result = model._tokens(
        action_plan=action_plan,
        world_plan=world_plan,
        current_world=current_world,
        noisy_action=noisy_action,
        noisy_world=noisy_world,
        action_time=torch.tensor([0.2, 0.7]),
        world_time=torch.tensor([0.2, 0.7]),
        action_is_pad=None,
        current_state=current_state,
    )
    action_hidden, world_hidden, action_memory, world_memory, bias, *_ = result
    assert action_hidden.shape[1] == 4
    assert world_hidden.shape[1] == 6
    assert action_memory.shape[1] == 1 + 4
    assert world_memory.shape[1] == 3 + 6
    assert torch.isfinite(bias).all()
    assert bias.shape[-1] == action_hidden.shape[1] + world_hidden.shape[1]
    action_memory.sum().backward()
    assert model.state_encoder.input_projection.weight.grad is not None

    result_at_other_time = model._tokens(
        action_plan=action_plan,
        world_plan=world_plan,
        current_world=current_world,
        noisy_action=noisy_action,
        noisy_world=noisy_world,
        action_time=torch.tensor([0.8, 0.1]),
        world_time=torch.tensor([0.8, 0.1]),
        action_is_pad=None,
        current_state=current_state,
    )
    assert torch.equal(action_memory.detach(), result_at_other_time[2].detach())
    assert torch.equal(world_memory.detach(), result_at_other_time[3].detach())

    result_with_other_conditions = model._tokens(
        action_plan=action_plan + 5.0,
        world_plan=world_plan - 4.0,
        current_world=current_world + 3.0,
        noisy_action=noisy_action,
        noisy_world=noisy_world,
        action_time=torch.tensor([0.2, 0.7]),
        world_time=torch.tensor([0.2, 0.7]),
        action_is_pad=None,
        current_state=current_state - 2.0,
    )
    assert torch.equal(action_hidden.detach(), result_with_other_conditions[0].detach())
    assert torch.equal(world_hidden.detach(), result_with_other_conditions[1].detach())
    assert not torch.equal(
        action_memory.detach(), result_with_other_conditions[2].detach()
    )
    assert not torch.equal(
        world_memory.detach(), result_with_other_conditions[3].detach()
    )


def test_fastwam_per_sample_padding_reduction():
    squared_error = torch.tensor(
        [
            [[1.0], [3.0]],
            [[10.0], [999.0]],
        ]
    )
    action_is_pad = torch.tensor(
        [
            [False, False],
            [False, True],
        ]
    )
    # mean([mean([1, 3]), mean([10])]) = mean([2, 10]) = 6.
    assert torch.equal(
        _masked_action_mse(squared_error, action_is_pad),
        torch.tensor(6.0),
    )


def test_jit_x_action_parameterization_recovers_flow_velocity():
    model = _model()
    clean = torch.tensor([[[2.0, -1.0, 0.5]]])
    noise = torch.tensor([[[-2.0, 3.0, -0.5]]])
    time = torch.tensor([0.4])
    noisy = (1.0 - time[:, None, None]) * noise + time[:, None, None] * clean
    velocity = model._action_to_velocity(clean, noisy, time)
    assert torch.allclose(velocity, clean - noise)


def test_bfloat16_multistep_sampling_preserves_model_dtype():
    model = _model(state_dim=3).to(dtype=torch.bfloat16).eval()
    seen_dtypes = []
    seen_times = []

    def predict(**kwargs):
        seen_dtypes.append(
            (
                kwargs["action_plan"].dtype,
                kwargs["world_plan"].dtype,
                kwargs["current_world"].dtype,
                kwargs["noisy_action"].dtype,
                kwargs["noisy_world"].dtype,
                kwargs["current_state"].dtype,
            )
        )
        seen_times.append(
            (
                kwargs["action_time"].dtype,
                torch.equal(kwargs["action_time"], kwargs["world_time"]),
            )
        )
        return (
            torch.zeros_like(kwargs["noisy_action"]),
            torch.zeros_like(kwargs["noisy_world"]),
        )

    model._predict = predict
    action, world = model.sample(
        action_plan=torch.randn(2, 4, 24),
        world_plan=torch.randn(2, 3, 24),
        current_world=torch.randn(2, 6, 12),
        current_state=torch.randn(2, 1, 3),
        seed=11,
    )

    expected = (torch.bfloat16,) * 6
    assert seen_dtypes == [expected] * model.inference_steps
    assert seen_times == [(torch.float32, True)] * model.inference_steps
    assert action.dtype == torch.bfloat16
    assert world.dtype == torch.bfloat16


def test_bfloat16_training_matches_sampling_input_dtype_without_autocast():
    model = _model(state_dim=3).to(dtype=torch.bfloat16).train()
    captured = {}

    def sample_time(batch_size, *, device):
        assert batch_size == 2
        return torch.tensor([0.2, 0.7], device=device, dtype=torch.float32)

    def predict(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros_like(kwargs["noisy_action"]),
            torch.zeros_like(kwargs["noisy_world"]),
        )

    model._sample_flow_time = sample_time
    model._predict = predict
    output = model.forward_train(
        # Deliberately provide fp32 tensors: both public train and sample
        # boundaries must align them to the bf16 physical model.
        action_plan=torch.randn(2, 4, 24),
        world_plan=torch.randn(2, 3, 24),
        current_world=torch.randn(2, 6, 12),
        target_action=torch.randn(2, 4, 3),
        target_world=torch.randn(2, 6, 12),
        action_is_pad=torch.zeros(2, 4, dtype=torch.bool),
        future_valid=torch.ones(2),
        current_state=torch.randn(2, 1, 3),
    )

    for key in (
        "action_plan",
        "world_plan",
        "current_world",
        "noisy_action",
        "noisy_world",
        "current_state",
    ):
        assert captured[key].dtype == torch.bfloat16
    # The shared clock remains fp32 in both paths and is converted only where
    # it participates in physical-latent arithmetic.
    assert captured["action_time"].dtype == torch.float32
    assert torch.equal(captured["action_time"], captured["world_time"])
    assert torch.isfinite(output["loss"])


def test_gr00t_time_sampling_respects_noise_ceiling():
    model = _model()
    time = model._sample_flow_time(1024, device=torch.device("cpu"))
    assert time.shape == (1024,)
    assert bool((time >= 0).all())
    assert bool((time <= model.noise_s).all())


def test_joint_training_uses_one_shared_physical_time():
    model = _model()
    shared_time = torch.tensor([0.2, 0.7])
    captured = {}

    def sample_time(batch_size, *, device):
        assert batch_size == 2
        return shared_time.to(device)

    def predict(**kwargs):
        captured["action_time"] = kwargs["action_time"]
        captured["world_time"] = kwargs["world_time"]
        return (
            torch.zeros_like(kwargs["noisy_action"]),
            torch.zeros_like(kwargs["noisy_world"]),
        )

    model._sample_flow_time = sample_time
    model._predict = predict
    model.forward_train(
        action_plan=torch.randn(2, 4, 24),
        world_plan=torch.randn(2, 3, 24),
        current_world=torch.randn(2, 6, 12),
        target_action=torch.randn(2, 4, 3),
        target_world=torch.randn(2, 6, 12),
        action_is_pad=None,
        future_valid=torch.ones(2),
    )
    assert torch.equal(captured["action_time"], shared_time)
    assert torch.equal(captured["world_time"], shared_time)


def test_repeated_diffusion_reuses_conditions_with_independent_flow_samples():
    model = _model(state_dim=3, repeated_diffusion_steps=2)
    sampled_batches = []
    captured = {}

    def sample_time(batch_size, *, device):
        sampled_batches.append(batch_size)
        return torch.tensor([0.1, 0.2, 0.6, 0.8], device=device)

    def predict(**kwargs):
        captured.update(kwargs)
        return (
            torch.zeros_like(kwargs["noisy_action"]),
            torch.zeros_like(kwargs["noisy_world"]),
        )

    action_plan = torch.randn(2, 4, 24)
    world_plan = torch.randn(2, 3, 24)
    current_state = torch.randn(2, 1, 3)
    model._sample_flow_time = sample_time
    model._predict = predict
    model.forward_train(
        action_plan=action_plan,
        world_plan=world_plan,
        current_world=torch.randn(2, 6, 12),
        target_action=torch.randn(2, 4, 3),
        target_world=torch.randn(2, 6, 12),
        action_is_pad=torch.zeros(2, 4, dtype=torch.bool),
        future_valid=torch.ones(2),
        current_state=current_state,
    )
    assert sampled_batches == [4]
    assert torch.equal(captured["action_plan"], action_plan.repeat(2, 1, 1))
    assert torch.equal(captured["world_plan"], world_plan.repeat(2, 1, 1))
    assert torch.equal(captured["current_state"], current_state.repeat(2, 1, 1))
    assert torch.equal(captured["action_time"], captured["world_time"])
    assert torch.unique(captured["action_time"]).numel() == 4


def test_joint_flow_trains_and_samples_both_streams():
    torch.manual_seed(7)
    model = _model(state_dim=3)
    model.gradient_checkpointing = True
    batch = 2
    current_state = torch.randn(batch, 1, 3)
    output = model.forward_train(
        action_plan=torch.randn(batch, 4, 24),
        world_plan=torch.randn(batch, 3, 24),
        current_world=torch.randn(batch, 6, 12),
        target_action=torch.randn(batch, 4, 3),
        target_world=torch.randn(batch, 6, 12),
        action_is_pad=torch.tensor([[False] * 4, [False, False, True, True]]),
        future_valid=torch.tensor([1.0, 0.0]),
        current_state=current_state,
    )
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert model.layers[0].action_attention.q_proj.weight.grad is not None
    assert model.layers[0].world_attention.q_proj.weight.grad is not None
    assert model.layers[0].action_attention.k_proj.weight.grad is not None
    assert model.layers[1].action_attention.q_proj.weight.grad is not None
    assert model.layers[1].world_attention.q_proj.weight.grad is not None
    assert model.action_plan_projection.weight.grad is not None
    assert model.world_plan_projection.weight.grad is not None
    assert model.current_world_projection.weight.grad is not None
    assert model.state_encoder.input_projection.weight.grad is not None

    model.eval()
    action, world = model.sample(
        action_plan=torch.randn(batch, 4, 24),
        world_plan=torch.randn(batch, 3, 24),
        current_world=torch.randn(batch, 6, 12),
        current_state=current_state,
        seed=11,
    )
    assert action.shape == (batch, 4, 3)
    assert world.shape == (batch, 6, 12)
