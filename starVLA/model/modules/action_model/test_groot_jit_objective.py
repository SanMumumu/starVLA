"""Focused checks for QwenGR00T action-output parameterizations."""

import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    _masked_action_mse,
    action_prediction_to_velocity,
)


def _minimal_head_config(prediction_type=None, action_model_type="DiT-B"):
    embedding_dim = 1024 if action_model_type == "DiT-LAWAM" else 768
    action_model = {
        "action_model_type": action_model_type,
        "hidden_size": 16,
        "state_dim": 0,
        "action_dim": 2,
        "action_horizon": 2,
        "num_inference_timesteps": 2,
        "num_target_vision_tokens": 1,
        "add_pos_embed": False,
        "max_seq_len": 4,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "diffusion_model_cfg": {
            "cross_attention_dim": 8,
            "dropout": 0.0,
            "final_dropout": False,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 1,
            "output_dim": embedding_dim,
            "positional_embeddings": None,
        },
    }
    if prediction_type is not None:
        action_model.update(prediction_type=prediction_type, jit_t_eps=0.05)
    return OmegaConf.create({"framework": {"action_model": action_model}})


def test_lawam_dit_shape_is_consistent():
    head = FlowmatchingActionHead(_minimal_head_config(action_model_type="DiT-LAWAM"))

    assert head.input_embedding_dim == 1024
    assert head.model.inner_dim == 1024
    assert head.model.config.num_attention_heads == 16
    assert head.model.config.attention_head_dim == 64
    assert head.action_encoder.layer1.out_features == 1024
    assert head.action_decoder.layer1.in_features == 1024


def test_velocity_parameterization_is_identity():
    prediction = torch.randn(2, 4, 7)
    noisy = torch.randn_like(prediction)

    actual = action_prediction_to_velocity(
        prediction,
        noisy,
        0.9,
        prediction_type="velocity",
        t_eps=0.05,
    )

    assert actual is prediction


def test_jit_x_parameterization_matches_reference():
    clean = torch.tensor([[[1.0, -0.5], [0.25, 0.75]]])
    noise = torch.tensor([[[-1.0, 0.5], [0.5, -0.25]]])
    t = torch.tensor([[[0.8]]])
    noisy = t * clean + (1 - t) * noise
    x_prediction = clean + 0.1

    target_velocity = action_prediction_to_velocity(
        clean,
        noisy,
        t,
        prediction_type="jit_x",
        t_eps=0.05,
    )
    pred_velocity = action_prediction_to_velocity(
        x_prediction,
        noisy,
        t,
        prediction_type="jit_x",
        t_eps=0.05,
    )

    denominator = (1 - t).clamp_min(0.05)
    torch.testing.assert_close(target_velocity, (clean - noisy) / denominator)
    torch.testing.assert_close(pred_velocity, (x_prediction - noisy) / denominator)
    torch.testing.assert_close(
        ((pred_velocity - target_velocity) ** 2).mean(),
        (((x_prediction - clean) / denominator) ** 2).mean(),
    )


def test_jit_t_eps_caps_late_time_weight():
    noisy = torch.zeros(1, 1, 1)
    prediction = torch.ones_like(noisy)

    actual = action_prediction_to_velocity(
        prediction,
        noisy,
        0.999,
        prediction_type="jit_x",
        t_eps=0.05,
    )

    torch.testing.assert_close(actual, torch.full_like(actual, 20.0))


def test_prediction_type_does_not_change_checkpoint_structure():
    legacy_head = FlowmatchingActionHead(_minimal_head_config())
    jit_head = FlowmatchingActionHead(_minimal_head_config("jit_x"))

    assert legacy_head.prediction_type == "velocity"
    assert jit_head.prediction_type == "jit_x"
    legacy_shapes = {key: tuple(value.shape) for key, value in legacy_head.state_dict().items()}
    jit_shapes = {key: tuple(value.shape) for key, value in jit_head.state_dict().items()}
    assert legacy_shapes == jit_shapes
    jit_head.load_state_dict(legacy_head.state_dict(), strict=True)
    legacy_head.load_state_dict(jit_head.state_dict(), strict=True)

    class FixedBeta:
        @staticmethod
        def sample(shape):
            return torch.full(tuple(shape), 0.25)

    legacy_head.beta_dist = FixedBeta()
    legacy_time = legacy_head.sample_time(2, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(legacy_time, torch.full((2,), (0.999 - 0.25) / 0.999))

    jit_head.beta_dist = FixedBeta()
    jit_head.flow_time_sampling = "gr00t"
    gr00t_time = jit_head.sample_time(2, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(gr00t_time, torch.full((2,), (1.0 - 0.25) * 0.999))

    vl_embs = torch.randn(1, 3, 8)
    actions = torch.randn(1, 2, 2)
    loss = jit_head(vl_embs, actions)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(jit_head.action_decoder.layer2.weight.grad).all()

    predicted_actions = jit_head.predict_action(vl_embs)
    assert predicted_actions.shape == actions.shape
    assert torch.isfinite(predicted_actions).all()


def test_all_padded_actions_contribute_zero_loss():
    head = FlowmatchingActionHead(_minimal_head_config("jit_x"))
    vl_embs = torch.randn(1, 3, 8)
    actions = torch.randn(1, 2, 2)

    loss = head(vl_embs, actions, action_is_pad=torch.ones(1, 2, dtype=torch.bool))

    assert loss.item() == 0.0
    loss.backward()
    assert head.action_decoder.layer2.weight.grad is not None


def test_padding_reduction_matches_fastwam_per_sample_mean():
    squared_error = torch.tensor([[[1.0], [3.0], [100.0]], [[10.0], [100.0], [100.0]]])
    action_is_pad = torch.tensor([[False, False, True], [False, True, True]])

    loss = _masked_action_mse(squared_error, action_is_pad)

    # FastWAM: mean([mean([1, 3]), mean([10])]) == 6. A global valid-step
    # reduction would incorrectly return 14/3.
    torch.testing.assert_close(loss, torch.tensor(6.0))


def test_correlated_noise_never_silently_falls_back_to_iid():
    head = FlowmatchingActionHead(_minimal_head_config())
    head.use_correlated_noise = True

    try:
        head._sample_initial_noise(1, torch.device("cpu"), torch.float32)
    except RuntimeError as exc:
        assert "no action-correlation Cholesky" in str(exc)
    else:
        raise AssertionError("correlated-noise sampling unexpectedly fell back to IID")

    head.set_action_correlation(torch.eye(4))
    noise = head._sample_initial_noise(3, torch.device("cpu"), torch.float32)
    assert noise.shape == (3, 2, 2)
    assert torch.isfinite(noise).all()


if __name__ == "__main__":
    test_lawam_dit_shape_is_consistent()
    test_velocity_parameterization_is_identity()
    test_jit_x_parameterization_matches_reference()
    test_jit_t_eps_caps_late_time_weight()
    test_prediction_type_does_not_change_checkpoint_structure()
    test_all_padded_actions_contribute_zero_loss()
    test_padding_reduction_matches_fastwam_per_sample_mean()
    test_correlated_noise_never_silently_falls_back_to_iid()
