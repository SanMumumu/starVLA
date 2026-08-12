from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image


EVAL_DIR = Path(__file__).resolve().parent


def _write_causal_mot_shape_checkpoint(
    checkpoint: Path,
    config: dict,
) -> None:
    """Write a metadata-only state_dict with the config's physical shapes."""

    framework = config["framework"]
    physical = framework["world_action_mot"]
    action = framework["action_model"]
    dino = framework["dino"]
    world_hidden = int(physical["world_hidden_size"])
    action_hidden = int(physical["action_hidden_size"])
    world_ffn = int(physical["world_ffn_dim"])
    action_ffn = int(physical["action_ffn_dim"])
    layers = int(physical["num_layers"])
    inner = (
        int(physical["num_attention_heads"])
        * int(physical["attention_head_dim"])
    )
    layerwise = bool(physical.get("layerwise_planner_coupling", False))

    def meta(*shape: int) -> torch.Tensor:
        return torch.empty(shape, device="meta")

    state = {
        "action_model.world_input.weight": meta(
            world_hidden,
            int(dino["embed_dim"]),
        ),
        "action_model.action_input.weight": meta(
            action_hidden,
            int(action["action_dim"]),
        ),
        "action_plan_queries.embedding": meta(
            1,
            int(framework["planner"]["num_action_queries"]),
            int(framework["qwenvl"]["vl_hidden_dim"]),
        ),
        "world_plan_queries.embedding": meta(
            1,
            int(framework["planner"]["num_world_queries"]),
            int(framework["qwenvl"]["vl_hidden_dim"]),
        ),
        "action_model.layers.0.world.self_attn.q.weight": meta(
            inner,
            world_hidden,
        ),
        "action_model.layers.0.world.ffn.0.weight": meta(
            world_ffn,
            world_hidden,
        ),
        "action_model.layers.0.action.ffn.0.weight": meta(
            action_ffn,
            action_hidden,
        ),
    }
    if int(action.get("state_dim", 0) or 0) > 0:
        state["action_model.state_to_planner.weight"] = meta(
            int(framework["qwenvl"]["vl_hidden_dim"]),
            int(action["state_dim"]),
        )
    if layerwise:
        state["action_model.world_context.0.0.weight"] = meta(
            world_hidden,
            int(framework["qwenvl"]["vl_hidden_dim"]),
        )
        state["action_model.action_context.0.0.weight"] = meta(
            action_hidden,
            int(framework["qwenvl"]["vl_hidden_dim"]),
        )
    else:
        state["action_model.world_context.0.weight"] = meta(
            world_hidden,
            int(framework["qwenvl"]["vl_hidden_dim"]),
        )
        state["action_model.action_context.0.weight"] = meta(
            action_hidden,
            int(framework["qwenvl"]["vl_hidden_dim"]),
        )
    for layer in range(1, layers):
        state[f"action_model.layers.{layer}.world.modulation"] = meta(
            1,
            6,
            world_hidden,
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint)


def _load_adapter(monkeypatch):
    xpolicy = types.ModuleType("XPolicyLab")
    model_template = types.ModuleType("XPolicyLab.model_template")
    process_data = types.ModuleType("XPolicyLab.utils.process_data")
    utils = types.ModuleType("XPolicyLab.utils")

    class ModelTemplate:
        pass

    model_template.ModelTemplate = ModelTemplate
    process_data.get_robot_action_dim_info = lambda _name: {
        "arm_dim": [6, 6],
        "ee_dim": [1, 1],
    }
    process_data.pack_robot_state = lambda *_args, **_kwargs: np.zeros(14, dtype=np.float32)
    process_data.unpack_robot_state = lambda action, *_args, **_kwargs: action

    monkeypatch.setitem(sys.modules, "XPolicyLab", xpolicy)
    monkeypatch.setitem(sys.modules, "XPolicyLab.model_template", model_template)
    monkeypatch.setitem(sys.modules, "XPolicyLab.utils", utils)
    monkeypatch.setitem(sys.modules, "XPolicyLab.utils.process_data", process_data)

    path = EVAL_DIR / "robodojo_model.py"
    spec = importlib.util.spec_from_file_location("_test_robodojo_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_simulator_launcher():
    path = EVAL_DIR / "launch_robodojo_client.py"
    spec = importlib.util.spec_from_file_location(
        "_test_launch_robodojo_client",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simulator_batch_override_does_not_read_stale_shared_deploy(
    monkeypatch,
) -> None:
    module = _load_simulator_launcher()
    namespace = {"_eval_batch_from_deploy": lambda _policy_name: False}
    monkeypatch.setenv("ROBODOJO_EVAL_BATCH", "true")
    module._install_eval_batch_override(namespace)
    assert namespace["_eval_batch_from_deploy"]("starVLA") is True

    monkeypatch.setenv("ROBODOJO_EVAL_BATCH", "false")
    module._install_eval_batch_override(namespace)
    assert namespace["_eval_batch_from_deploy"]("starVLA") is False

    monkeypatch.setenv("ROBODOJO_EVAL_BATCH", "six")
    with pytest.raises(ValueError, match="must be a boolean"):
        module._install_eval_batch_override(namespace)

    monkeypatch.setenv("ROBODOJO_EVAL_BATCH", "true")
    with pytest.raises(RuntimeError, match="no longer exposes"):
        module._install_eval_batch_override({})


def test_default_replan_is_12_and_restarts_each_new_chunk(monkeypatch) -> None:
    deploy = yaml.safe_load((EVAL_DIR / "deploy.yml").read_text(encoding="utf-8"))
    assert deploy["replan_interval"] == 12
    assert deploy["text_replan_chunks"] == 4
    assert deploy["log_planner_text"] is False
    assert deploy["num_ddim_steps"] == 10
    assert deploy["use_ddim"] is True
    assert deploy["rtc_enabled"] is False
    assert deploy["rtc_execution_horizon"] == 4
    assert deploy["rtc_inference_delay"] == 1
    assert deploy["rtc_max_guidance_weight"] == pytest.approx(20.0)
    assert deploy["rtc_prefix_attention_schedule"] == "linear"
    assert deploy["rtc_debug_max_replans"] == 8
    assert "include_state" not in deploy
    assert "expected_state_dim" not in deploy

    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.replan_interval = 12
    model.action_chunk_size = 16
    model.action_dim = 1
    model.step_by_env = {}
    model.action_chunks_by_env = {}
    calls = []

    def infer_chunk(env_idx: int) -> np.ndarray:
        calls.append(env_idx)
        base = 100 * (len(calls) - 1)
        return np.arange(base, base + 16, dtype=np.float32).reshape(16, 1)

    model._infer_chunk = infer_chunk
    actions = [float(model._next_action(7)[0]) for _ in range(26)]
    assert actions == [
        *[float(value) for value in range(12)],
        *[float(value) for value in range(100, 112)],
        200.0,
        201.0,
    ]
    assert calls == [7, 7, 7]


def test_adapter_omits_state_for_query_only_checkpoints(monkeypatch) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model._build_composite = lambda _images: Image.fromarray(
        np.zeros((384, 320, 3), dtype=np.uint8)
    )
    model.image_size = (320, 384)
    model.default_instruction = "stack bowls"
    model.action_type = "joint"
    model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    model.state_dim = 14
    model.expects_state = False

    def forbidden_state(*_args, **_kwargs):
        raise AssertionError("stateless checkpoint must not pack proprioception")

    monkeypatch.setattr(module, "pack_robot_state", forbidden_state)
    camera = np.zeros((32, 32, 3), dtype=np.uint8)
    converted = model._convert_obs(
        {
            "task_instruction": "stack bowls",
            "vision": {
                "cam_head": camera,
                "cam_left_wrist": camera,
                "cam_right_wrist": camera,
            },
        }
    )
    assert set(converted) == {"lang", "image"}


def test_client_rejects_a_different_server_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_adapter(monkeypatch)
    expected = tmp_path / "qwen/checkpoints/steps_50000.pt"
    different = tmp_path / "rynn/checkpoints/steps_50000.pt"
    module._validate_checkpoint_selection(
        expected,
        {"ckpt_path": str(expected)},
    )
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        module._validate_checkpoint_selection(
            expected,
            {"ckpt_path": str(different)},
        )


def test_client_validates_loaded_legacy_and_layerwise_mot_recipes(
    monkeypatch,
) -> None:
    module = _load_adapter(monkeypatch)
    common = {
        "framework_name": "QwenWorldActionMoT",
        "mot_interaction_mode": "joint",
        "mot_action_prediction_type": "velocity",
        "planner_query_mask_contract": "positional_suffix_v2",
    }
    legacy = {
        **common,
        "mot_contract_version": "legacy_shared_context_state_world_v1",
        "mot_layerwise_planner_coupling": False,
        "mot_num_inference_timesteps": 20,
        "mot_action_velocity_target": "noise_minus_clean",
    }
    module._validate_mot_runtime_contract(legacy)

    jit_x = {
        **legacy,
        "mot_action_prediction_type": "jit_x",
    }
    module._validate_mot_runtime_contract(jit_x)

    layerwise = {
        **common,
        "mot_contract_version": "layerwise_query_only_world_v2",
        "mot_layerwise_planner_coupling": True,
        "mot_num_inference_timesteps": 10,
        "mot_action_velocity_target": "clean_minus_noise",
    }
    module._validate_mot_runtime_contract(layerwise)

    multires = {
        **layerwise,
        "mot_contract_version": (
            "layerwise_query_only_multires_world_v3"
        ),
        "mot_multires_world_input": True,
        "mot_current_dino_tokens": 480,
        "mot_future_dino_tokens": 120,
    }
    module._validate_mot_runtime_contract(multires)

    scheduler_native = {
        **multires,
        "mot_action_velocity_target": "noise_minus_clean",
    }
    module._validate_mot_runtime_contract(scheduler_native)

    invalid_target = {
        **multires,
        "mot_action_velocity_target": "unsupported",
    }
    with pytest.raises(RuntimeError, match="invalid action velocity target"):
        module._validate_mot_runtime_contract(invalid_target)

    stale = {
        **legacy,
        "planner_query_mask_contract": None,
    }
    with pytest.raises(RuntimeError, match="stale planner-mask code"):
        module._validate_mot_runtime_contract(stale)


def test_batch_replan_sends_one_batched_inference_request(monkeypatch) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.replan_interval = 12
    model.action_chunk_size = 16
    model.action_dim = 1
    model.action_type = "joint"
    model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    model.step_by_env = {}
    model.action_chunks_by_env = {}
    model._latest_env_idx_list = [0, 1, 2, 3]
    calls = []

    def infer_chunks(
        env_idx_list: list[int],
        *,
        refresh_text_envs=None,
    ) -> dict[int, np.ndarray]:
        assert refresh_text_envs == []
        calls.append(tuple(env_idx_list))
        return {
            env_idx: (
                np.arange(16, dtype=np.float32).reshape(16, 1)
                + 100 * env_idx
                + 1000 * (len(calls) - 1)
            )
            for env_idx in env_idx_list
        }

    model._infer_chunks = infer_chunks
    for _ in range(13):
        actions = model.get_action_batch()

    assert calls == [(0, 1, 2, 3), (0, 1, 2, 3)]
    assert [float(env_actions[0][0]) for env_actions in actions] == [
        1000.0,
        1100.0,
        1200.0,
        1300.0,
    ]


def test_infer_chunks_batches_examples_in_one_server_call(monkeypatch) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.obs_by_env = {
        env_idx: {"lang": f"task-{env_idx}"}
        for env_idx in range(4)
    }
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.unnorm_key = "new_embodiment"
    model.action_chunk_size = 16
    model.action_dim = 14
    model.planner_text_by_env = {}
    payloads = []

    class Client:
        def predict_action(self, payload):
            payloads.append(payload)
            return {
                "ok": True,
                "data": {
                    "actions": np.zeros((4, 16, 14), dtype=np.float32),
                },
            }

    model.client = Client()
    chunks = model._infer_chunks([0, 1, 2, 3])

    assert len(payloads) == 1
    assert "cached_planner_texts" not in payloads[0]
    assert all(
        "planner_history_images" not in example
        and "finished_task_list" not in example
        for example in payloads[0]["examples"]
    )
    assert [example["lang"] for example in payloads[0]["examples"]] == [
        "task-0",
        "task-1",
        "task-2",
        "task-3",
    ]
    assert set(chunks) == {0, 1, 2, 3}


def test_infer_rtc_sends_the_previous_normalized_h16_tail(
    monkeypatch,
    capsys,
) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.obs_by_env = {0: {"lang": "task-0"}, 1: {"lang": "task-1"}}
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.unnorm_key = "new_embodiment"
    model.action_chunk_size = 16
    model.action_dim = 1
    model.replan_interval = 12
    model.rtc_enabled = True
    model.rtc_execution_horizon = 4
    model.rtc_inference_delay = 1
    model.rtc_prefix_attention_schedule = "linear"
    model.rtc_max_guidance_weight = 20.0
    model.rtc_debug_max_replans = 8
    model.rtc_debug_replans_by_env = {}
    model.planner_text_by_env = {}
    old_chunk = np.arange(16, dtype=np.float32).reshape(16, 1)
    model.normalized_action_chunks_by_env = {0: old_chunk.copy()}
    payloads = []

    class Client:
        def predict_action(self, payload):
            payloads.append(payload)
            return {
                "ok": True,
                "data": {
                    "actions": np.zeros((2, 16, 1), dtype=np.float32),
                    "normalized_actions": np.full(
                        (2, 16, 1),
                        7.0,
                        dtype=np.float32,
                    ),
                },
            }

    model.client = Client()
    model._infer_chunks([0, 1])

    payload = payloads[0]
    np.testing.assert_array_equal(payload["rtc_prefix_lengths"], [4, 0])
    np.testing.assert_array_equal(
        payload["prev_action_chunk_normalized"][0, :4, 0],
        [12.0, 13.0, 14.0, 15.0],
    )
    np.testing.assert_array_equal(
        payload["prev_action_chunk_normalized"][1],
        np.zeros((16, 1), dtype=np.float32),
    )
    assert payload["inference_delay"] == 1
    assert payload["execution_horizon"] == 4
    assert payload["prefix_attention_schedule"] == "linear"
    assert payload["max_guidance_weight"] == pytest.approx(20.0)
    rtc_log = capsys.readouterr().out
    assert "[starVLA][RoboDojo][RTC] env=0 replan=1" in rtc_log
    assert "planned_step_rmse=1.000000" in rtc_log
    assert "new_switch_rmse=4.000000" in rtc_log
    assert "rtc_target_rmse=5.000000" in rtc_log
    assert "prefix_rmse=6.595453" in rtc_log
    np.testing.assert_array_equal(
        model.normalized_action_chunks_by_env[0],
        np.full((16, 1), 7.0, dtype=np.float32),
    )


def test_text_planner_refreshes_once_every_four_action_replans(
    monkeypatch,
) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.replan_interval = 12
    model.text_planning_enabled = True
    model.text_replan_chunks = 4
    model.action_chunk_size = 16
    model.action_dim = 1
    model.action_type = "joint"
    model.robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
    model.step_by_env = {}
    model.action_chunks_by_env = {}
    model.planner_text_by_env = {}
    model._latest_env_idx_list = [0, 1]
    calls = []

    def infer_chunks(
        env_idx_list: list[int],
        *,
        refresh_text_envs=None,
    ) -> dict[int, np.ndarray]:
        refresh = tuple(refresh_text_envs or ())
        calls.append((tuple(env_idx_list), refresh))
        for env_idx in refresh:
            model.planner_text_by_env[env_idx] = f"plan-{len(calls)}-{env_idx}"
        return {
            env_idx: np.arange(16, dtype=np.float32).reshape(16, 1)
            for env_idx in env_idx_list
        }

    model._infer_chunks = infer_chunks
    for _ in range(49):
        model.get_action_batch()

    assert calls == [
        ((0, 1), (0, 1)),
        ((0, 1), ()),
        ((0, 1), ()),
        ((0, 1), ()),
        ((0, 1), (0, 1)),
    ]


def test_text_inference_payload_marks_refresh_rows_with_none(monkeypatch) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.obs_by_env = {
        env_idx: {"lang": f"task-{env_idx}"}
        for env_idx in range(3)
    }
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.unnorm_key = "new_embodiment"
    model.action_chunk_size = 16
    model.action_dim = 14
    model.text_planning_enabled = True
    model.planner_text_by_env = {
        0: "cached-0",
        1: "stale-1",
        2: "cached-2",
    }
    payloads = []

    class Client:
        def predict_action(self, payload):
            payloads.append(payload)
            return {
                "ok": True,
                "data": {
                    "actions": np.zeros((3, 16, 14), dtype=np.float32),
                    "planner_text": ["cached-0", "fresh-1", "cached-2"],
                    "planner_text_refreshed": [False, True, False],
                },
            }

    model.client = Client()
    model._infer_chunks([0, 1, 2], refresh_text_envs=[1])

    assert payloads[0]["cached_planner_texts"] == [
        "cached-0",
        None,
        "cached-2",
    ]
    assert model.planner_text_by_env == {
        0: "cached-0",
        1: "fresh-1",
        2: "cached-2",
    }


def test_event_memory_inference_updates_delta_then_keeps_state(monkeypatch) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.obs_by_env = {0: {"lang": "put the cup away"}}
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.unnorm_key = "new_embodiment"
    model.action_chunk_size = 25
    model.action_dim = 14
    model.rtc_enabled = False
    model.text_planning_enabled = True
    model.event_memory_enabled = True
    model.text_history_enabled = False
    model.event_semantic_fields = {
        "memory": "semantic_memory",
        "cached_subtask": "cached_current_subtask",
        "decision": "semantic_decision",
        "memory_add": "memory_add",
        "cache_valid": "semantic_cache_valid",
    }
    model.event_empty_memory = "None."
    model.event_empty_cached_subtask = "None."
    model.event_semantic_offset = -10
    model.event_replan_interval = 10
    model.semantic_memory_by_env = {}
    model.cached_current_subtask_by_env = {}
    model.planner_text_by_env = {}
    model.log_planner_text = False
    payloads = []

    class Client:
        def predict_action(self, payload):
            payloads.append(payload)
            if len(payloads) == 1:
                semantic = {
                    "semantic_decision": ["UPDATE"],
                    "semantic_memory_add": ["None."],
                    "semantic_current_subtask": ["Open the drawer."],
                    "planner_text": [
                        "<UPDATE>\nMemory Add: None.\nCurrent Subtask: Open the drawer."
                    ],
                }
            else:
                semantic = {
                    "semantic_decision": ["KEEP"],
                    "semantic_memory_add": [None],
                    "semantic_current_subtask": [None],
                    "planner_text": ["<KEEP>"],
                }
            return {
                "ok": True,
                "data": {
                    "actions": np.zeros((1, 25, 14), dtype=np.float32),
                    **semantic,
                },
            }

    model.client = Client()
    model._infer_chunks([0])
    first = payloads[0]["examples"][0]
    assert first["semantic_memory"] == "None."
    assert first["cached_current_subtask"] == "None."
    assert first["semantic_cache_valid"] is False
    assert "cached_planner_texts" not in payloads[0]
    assert model.semantic_memory_by_env[0] == "None."
    assert model.cached_current_subtask_by_env[0] == "Open the drawer."

    model._infer_chunks([0])
    second = payloads[1]["examples"][0]
    assert second["semantic_memory"] == "None."
    assert second["cached_current_subtask"] == "Open the drawer."
    assert second["semantic_cache_valid"] is True
    assert model.cached_current_subtask_by_env[0] == "Open the drawer."


def test_history_inference_updates_memory_but_reuses_the_original_prompt_state(
    monkeypatch,
) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.obs_by_env = {
        0: {
            "lang": "place all objects",
            "image": [np.full((8, 8, 3), 48, dtype=np.uint8)],
        }
    }
    model.step_by_env = {0: 48}
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.unnorm_key = "new_embodiment"
    model.action_chunk_size = 16
    model.action_dim = 14
    model.text_planning_enabled = True
    model.text_history_enabled = True
    model.text_history_frame_offsets = (-36, -24, -12)
    model.text_history_image_field = "planner_history_images"
    model.text_history_image_size = (4, 4)
    model.finished_task_list_field = "finished_task_list"
    model.empty_finished_task_list = "None"
    model.planner_text_by_env = {0: "stale"}
    model.finished_task_list_by_env = {
        0: "Place the figurine on the stand."
    }
    model.planner_input_finished_task_list_by_env = {}
    model.planner_observation_history_by_env = {0: {}}
    for step in (0, 12, 24, 36, 48):
        model._record_planner_observation(
            0,
            step,
            np.full((8, 8, 3), step, dtype=np.uint8),
        )
    payloads = []

    class Client:
        def predict_action(self, payload):
            payloads.append(payload)
            refreshed = len(payloads) == 1
            return {
                "ok": True,
                "data": {
                    "actions": np.zeros((1, 16, 14), dtype=np.float32),
                    "planner_text": [
                        (
                            "Current subtask: Place the clock.\n"
                            "Finished Task List: Place the figurine on the stand. "
                            "Place the mouse on the mouse pad."
                        )
                    ],
                    "planner_text_refreshed": [refreshed],
                    "planner_finished_task_list": [
                        "Place the figurine on the stand. "
                        "Place the mouse on the mouse pad."
                    ],
                },
            }

    model.client = Client()
    model._infer_chunks([0], refresh_text_envs=[0])

    first_example = payloads[0]["examples"][0]
    assert first_example["finished_task_list"] == (
        "Place the figurine on the stand."
    )
    history = first_example["planner_history_images"]
    assert history.shape == (3, 4, 4, 3)
    assert [int(image[0, 0, 0]) for image in history] == [12, 24, 36]
    assert model.finished_task_list_by_env[0] == (
        "Place the figurine on the stand. Place the mouse on the mouse pad."
    )
    assert model.planner_input_finished_task_list_by_env[0] == (
        "Place the figurine on the stand."
    )

    # While the planner text is cached, WORLD/ACTION queries must reuse the
    # same prompt memory that produced it, not pair the response with a new
    # causal prefix unseen during training.
    model._infer_chunks([0], refresh_text_envs=[])
    assert payloads[1]["examples"][0]["finished_task_list"] == (
        "Place the figurine on the stand."
    )
    assert payloads[1]["cached_planner_texts"][0].startswith(
        "Current subtask:"
    )


def test_history_only_mem_request_sends_frames_without_any_text_plan(
    monkeypatch,
) -> None:
    module = _load_adapter(monkeypatch)
    model = module.Model.__new__(module.Model)
    model.obs_by_env = {
        0: {
            "lang": "stack the bowls",
            "image": [np.full((4, 4, 3), 50, dtype=np.uint8)],
        }
    }
    model.step_by_env = {0: 100}
    model.use_ddim = True
    model.num_ddim_steps = 10
    model.unnorm_key = "new_embodiment"
    model.action_chunk_size = 25
    model.action_dim = 14
    model.rtc_enabled = False
    model.text_planning_enabled = False
    model.text_history_enabled = True
    model.text_history_frame_offsets = (-100, -80, -60, -40, -20)
    model.text_history_image_field = "planner_history_images"
    model.text_history_image_size = (4, 4)
    model.planner_observation_history_by_env = {0: {}}
    for step in (0, 20, 40, 60, 80, 100):
        model._record_planner_observation(
            0,
            step,
            np.full((4, 4, 3), step, dtype=np.uint8),
        )
    payloads = []

    class Client:
        def predict_action(self, payload):
            payloads.append(payload)
            return {
                "ok": True,
                "data": {
                    "actions": np.zeros((1, 25, 14), dtype=np.float32),
                },
            }

    model.client = Client()
    result = model._infer_chunks([0])

    assert result[0].shape == (25, 14)
    assert "cached_planner_texts" not in payloads[0]
    example = payloads[0]["examples"][0]
    assert "finished_task_list" not in example
    assert example["planner_history_images"].shape == (5, 4, 4, 3)


def test_eval_launcher_keeps_native_runtime_contract() -> None:
    eval_launcher = (EVAL_DIR / "eval_robodojo.sh").read_text(encoding="utf-8")
    assert 'ROBODOJO_REPLAN_STEPS:-12' in eval_launcher
    assert '--num_envs "${NUM_ENVS}"' in eval_launcher
    assert 'eval_batch="${EVAL_BATCH}"' in eval_launcher
    assert 'ROBODOJO_EVAL_BATCH="${EVAL_BATCH}"' in eval_launcher
    assert 'log_planner_text="${LOG_PLANNER_TEXT}"' in eval_launcher
    assert 'expected_checkpoint_path="${CHECKPOINT_PATH}"' in eval_launcher
    assert "client.raw.log" in eval_launcher
    assert "ROBODOJO_CONCISE_LOGS" in eval_launcher
    assert '${RUN_NAME}_${OUTPUT_RUN_ID}' in eval_launcher
    assert 'additional_info="eval"' in eval_launcher
    assert 'ROBODOJO_RUN_ID="${ROBODOJO_NATIVE_RUN_ID:-run}"' in eval_launcher
    assert "run_metadata.json" in eval_launcher


def test_eval_launchers_keep_optional_rtc_and_visualization_contracts() -> None:
    eval_launcher = (EVAL_DIR / "eval_robodojo.sh").read_text(
        encoding="utf-8"
    )
    visualization_launcher = (
        EVAL_DIR / "run_aidi_robodojo_visualize.sh"
    ).read_text(encoding="utf-8")
    assert "ROBODOJO_RTC_ENABLED" in eval_launcher
    assert "ROBODOJO_RTC_EXECUTION_HORIZON" in eval_launcher
    assert "ROBODOJO_RTC_INFERENCE_DELAY" in eval_launcher
    assert "ROBODOJO_RTC_PREFIX_ATTENTION_SCHEDULE" in eval_launcher
    assert "ROBODOJO_RTC_MAX_GUIDANCE_WEIGHT" in eval_launcher
    assert "ROBODOJO_RTC_DEBUG_MAX_REPLANS" in eval_launcher
    assert 'rtc_enabled="${RTC_ENABLED}"' in eval_launcher
    assert "ROBODOJO_EVAL_MODE=visualize" in visualization_launcher
    assert "ROBODOJO_TRIALS" in visualization_launcher
    assert "ROBODOJO_RUN_NAME" in visualization_launcher


def test_eval_launchers_support_25_action_chunk_with_replan_16() -> None:
    eval_launcher = (EVAL_DIR / "eval_robodojo.sh").read_text(encoding="utf-8")
    assert "ROBODOJO_EXPECTED_ACTION_CHUNK_SIZE:-16" in eval_launcher
    assert "10#${EXPECTED_ACTION_CHUNK_SIZE}" in eval_launcher
    assert 'expected_action_chunk_size="${EXPECTED_ACTION_CHUNK_SIZE}"' in eval_launcher


def test_text_h25_event_memory_recipe_preserves_fp32_physical_contract() -> None:
    repo_root = EVAL_DIR.parents[2]
    train_dir = EVAL_DIR.parent / "train_files/released_rynn50k"
    control = yaml.safe_load(
        (train_dir / "rynn_base_h25_50k.yaml").read_text(encoding="utf-8")
    )
    control["framework"]["world_action_mot"][
        "action_precision_mode"
    ] = "fp32_shell"
    variant = yaml.safe_load(
        (train_dir / "rynn_base_text_h25_mem_50k.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert int(variant["framework"]["action_model"]["action_horizon"]) == 25
    assert int(variant["framework"]["planner"]["num_action_queries"]) == 25
    history = variant["datasets"]["vla_data"]["text_annotations"]["history"]
    assert history == {"enabled": False}
    assert "mem_vision_encoder" not in variant["framework"]["qwenvl"]
    text = variant["framework"]["planner"]["text_supervision"]
    assert text["mode"] == "event_driven_memory_ntp"
    assert text["keep_token"] == "<KEEP>"
    assert text["update_token"] == "<UPDATE>"
    assert text["scheduled_sampling"]["points"] == [
        [0, 0.0],
        [30000, 0.0],
        [40000, 0.2],
        [50000, 0.5],
    ]
    event = variant["datasets"]["vla_data"]["text_annotations"][
        "event_memory"
    ]
    assert event["semantic_offset"] == -10
    assert event["replan_interval"] == 10
    assert event["sampler"]["per_device_batch_size"] == 6
    assert [
        event["sampler"]["update_per_batch"],
        event["sampler"]["hard_keep_per_batch"],
        event["sampler"]["random_keep_per_batch"],
    ] == [2, 2, 2]

    physical_ignored = {
        "run_id",
        "reproduction_profile",
        "text_supervision",
        "text_loss_weight",
        "data_mix",
        "text_annotations",
        "action_eval_enabled",
    }
    assert physical_ignored  # documents the verifier's intentional delta set
    assert variant["framework"]["world_action_mot"]["action_precision_mode"] == (
        control["framework"]["world_action_mot"]["action_precision_mode"]
    ) == "fp32_shell"
    assert variant["datasets"]["vla_data"]["per_device_batch_size"] == 12
    assert variant["datasets"]["vla_data"]["world_model"] == control[
        "datasets"
    ]["vla_data"]["world_model"]
    runbook = (
        repo_root / "执行脚本/Robodojo/released_rynn50k/run.sh"
    ).read_text(encoding="utf-8")
    assert "RoboDojo 8-server + 8-client vector-env eval" in runbook
    assert "job_server_rynnbrain.yaml" in runbook
    assert "job_client_robodojo.yaml" in runbook
    assert "base_text_h25_eventmem_ntp_nohist_fp32_50k" in runbook
    assert "base_text_h25_eventmem_ntp_nohist_bf16_50k" in runbook
    assert "job_base_text_bf16_50k.yaml" in runbook
    assert "ROBODOJO_TEXT_REPLAN_CHUNKS=1" in runbook
    assert "run_robodojo_policy_servers_8.sh" in runbook
    assert "run_aidi_robodojo_fast_full.sh" in runbook


def test_short_result_tree_uses_metadata_instead_of_checkpoint_in_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module_name = "_test_summarize_robodojo_table1"
    path = EVAL_DIR / "summarize_robodojo_table1.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    checkpoint = tmp_path / "qwen_joint/checkpoints/steps_80000_pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    eval_root = checkpoint.parents[1] / "robodojo_eval_results"
    job_root = (
        eval_root
        / "full_official_fast_20260727_120000-1"
        / "jobs/job001_stack_bowls"
    )
    result = (
        job_root
        / "native/eval_result/RoboDojo/stack_bowls"
        / "starVLA/arx_x5/0_eval/run/_result.json"
    )
    result.parent.mkdir(parents=True)
    result.write_text(
        json.dumps(
            {
                "details": {
                    "0": {"success": True, "score": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )
    (job_root / "run_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "ckpt_name": "qwen_joint",
                "seed": 0,
            }
        ),
        encoding="utf-8",
    )

    selected = module.scan_candidates(
        eval_root,
        checkpoint=checkpoint,
        ckpt_name="qwen_joint",
        seed=0,
    )
    assert selected["stack_bowls"].path == result
    assert "steps_80000" not in str(result.relative_to(eval_root))
    assert "ckpt_name=" not in str(result)


def test_checkpoint_preflight_reports_h25_recipe_and_legacy_defaults(
    tmp_path: Path,
) -> None:
    verifier_path = EVAL_DIR / "verify_robodojo_checkpoint_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_test_verify_robodojo_checkpoint_contract",
        verifier_path,
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    repo_root = str(EVAL_DIR.parents[2])
    sys.path.insert(0, repo_root)
    try:
        spec.loader.exec_module(verifier)
    finally:
        sys.path.remove(repo_root)
    verify = verifier.verify

    source = (
        EVAL_DIR.parent
        / "train_files/released_rynn50k/rynn_base_h25_50k.yaml"
    )
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    run_dir = tmp_path / "mot"
    checkpoint = run_dir / "checkpoints/steps_1_pytorch_model.pt"
    run_dir.mkdir(parents=True)
    zeros = [0.0] * 14
    modality = {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": [1.0] * 14,
        "q01": zeros,
        "q99": zeros,
        "mask": [True] * 14,
    }
    stats = {
        "new_embodiment": {
            "state": modality,
            "action": modality,
        }
    }
    stats_path = run_dir / "dataset_statistics.json"
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    config_path = run_dir / "config.full.yaml"
    serialized_config = yaml.safe_dump(config)
    (run_dir / "config.yaml").write_text(
        serialized_config,
        encoding="utf-8",
    )
    config_path.write_text(serialized_config, encoding="utf-8")
    _write_causal_mot_shape_checkpoint(checkpoint, config)
    summary = verify(str(checkpoint))
    assert summary["include_state"] is True
    assert summary["mot_weight_contract"]["state_dim"] == 14
    assert summary["mot_action_recipe"] == {
        "prediction_type": "velocity",
        "velocity_target": "noise_minus_clean",
        "loss": "velocity_mse",
        "repeated_diffusion_steps": 1,
        "jit_t_eps": 0.05,
        "world_loss_weight": 1.0,
        "saved_num_inference_steps": 20,
    }
    assert (
        summary["mot_weight_contract"]["version"]
        == "legacy_shared_context_state_world_v1"
    )
    mismatched_construction = yaml.safe_load(yaml.safe_dump(config))
    mismatched_construction["framework"]["world_action_mot"]["num_layers"] = 28
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(mismatched_construction),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="num_layers vs checkpoint"):
        verify(str(checkpoint))
    (run_dir / "config.yaml").write_text(
        serialized_config,
        encoding="utf-8",
    )

    h25_config = yaml.safe_load(
        (
            EVAL_DIR.parent
            / "train_files/released_rynn50k/rynn_base_h25_50k.yaml"
        ).read_text(encoding="utf-8")
    )
    h25_serialized = yaml.safe_dump(h25_config)
    (run_dir / "config.yaml").write_text(h25_serialized, encoding="utf-8")
    config_path.write_text(h25_serialized, encoding="utf-8")
    _write_causal_mot_shape_checkpoint(checkpoint, h25_config)
    h25_summary = verify(str(checkpoint))
    assert h25_summary["action_chunk"] == [25, 14]
    assert h25_summary["mot_weight_contract"]["num_action_queries"] == 25

    text_mem_config = yaml.safe_load(
        (
            EVAL_DIR.parent
            / "train_files/released_rynn50k/rynn_base_text_h25_mem_50k.yaml"
        ).read_text(encoding="utf-8")
    )
    text_mem_serialized = yaml.safe_dump(text_mem_config)
    (run_dir / "config.yaml").write_text(
        text_mem_serialized,
        encoding="utf-8",
    )
    config_path.write_text(text_mem_serialized, encoding="utf-8")
    _write_causal_mot_shape_checkpoint(checkpoint, text_mem_config)
    text_mem_summary = verify(str(checkpoint))
    assert text_mem_summary["action_chunk"] == [25, 14]
    assert text_mem_summary["mot_weight_contract"]["num_action_queries"] == 25
    assert text_mem_summary["text_planning_enabled"] is True
    assert text_mem_summary["event_memory_enabled"] is True
    assert text_mem_summary["event_memory_contract"] == {
        "semantic_offset": -10,
        "replan_interval": 10,
        "keep_token": "<KEEP>",
        "update_token": "<UPDATE>",
        "max_new_tokens": 96,
        "rgb_history": False,
    }

    text_mem_bf16_config = yaml.safe_load(
        (
            EVAL_DIR.parent
            / "train_files/released_rynn50k/rynn_base_text_h25_mem_bf16_50k.yaml"
        ).read_text(encoding="utf-8")
    )
    text_mem_bf16_serialized = yaml.safe_dump(text_mem_bf16_config)
    (run_dir / "config.yaml").write_text(
        text_mem_bf16_serialized,
        encoding="utf-8",
    )
    config_path.write_text(text_mem_bf16_serialized, encoding="utf-8")
    _write_causal_mot_shape_checkpoint(checkpoint, text_mem_bf16_config)
    text_mem_bf16_summary = verify(str(checkpoint))
    assert text_mem_bf16_summary["action_chunk"] == [25, 14]
    assert text_mem_bf16_summary["text_planning_enabled"] is True
    assert text_mem_bf16_summary["event_memory_enabled"] is True

    legacy = yaml.safe_load(yaml.safe_dump(config))
    physical = legacy["framework"]["world_action_mot"]
    physical.pop("action_prediction_type")
    physical.pop("action_velocity_target")
    physical.pop("jit_t_eps", None)
    physical.pop("repeated_diffusion_steps")
    legacy_serialized = yaml.safe_dump(legacy)
    (run_dir / "config.yaml").write_text(
        legacy_serialized,
        encoding="utf-8",
    )
    config_path.write_text(legacy_serialized, encoding="utf-8")
    _write_causal_mot_shape_checkpoint(checkpoint, legacy)
    legacy_summary = verify(str(checkpoint))
    assert legacy_summary["mot_action_recipe"]["prediction_type"] == "velocity"
    assert (
        legacy_summary["mot_action_recipe"]["velocity_target"]
        == "noise_minus_clean"
    )
    assert legacy_summary["mot_action_recipe"]["loss"] == "velocity_mse"
    assert legacy_summary["mot_action_recipe"]["repeated_diffusion_steps"] == 1

    semantic_mismatch = yaml.safe_load(yaml.safe_dump(legacy))
    semantic_mismatch["framework"]["world_action_mot"][
        "action_velocity_target"
    ] = "clean_minus_noise"
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(semantic_mismatch),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inference semantics"):
        verify(str(checkpoint))


def test_checkpoint_preflight_accepts_the_three_released_50k_contracts(
    tmp_path: Path,
) -> None:
    verifier_path = EVAL_DIR / "verify_robodojo_checkpoint_contract.py"
    spec = importlib.util.spec_from_file_location(
        "_test_verify_released_robodojo_50k",
        verifier_path,
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)

    cases = (
        (
            "starvla_qwen3_robodojo_dino_mot_joint_text_50k",
            "rynn_base_text_h25_mem_50k.yaml",
            True,
        ),
        (
            "starvla_qwen3_robodojo_dino_mot_joint_50k",
            "rynn_base_h25_50k.yaml",
            False,
        ),
        (
            "starvla_rynnbrain11_robodojo_dino_mot_joint_50k",
            "rynn_base_h25_50k.yaml",
            False,
        ),
    )
    zeros = [0.0] * 14
    modality = {
        "min": zeros,
        "max": zeros,
        "mean": zeros,
        "std": [1.0] * 14,
        "q01": zeros,
        "q99": zeros,
        "mask": [True] * 14,
    }
    statistics = {
        "new_embodiment": {
            "state": modality,
            "action": modality,
        }
    }

    for run_name, source_name, expected_text in cases:
        config = yaml.safe_load(
            (
                EVAL_DIR.parent
                / "train_files/released_rynn50k"
                / source_name
            ).read_text(encoding="utf-8")
        )
        config["run_id"] = run_name
        if "qwen3" in run_name:
            config["framework"]["qwenvl"]["base_vlm"] = (
                "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/"
                "CKPTS/Qwen3-VL-2B-Instruct/"
            )
        # These fixtures model the released pre-layerwise checkpoints, whose
        # saved architecture consumed normalized 14-D proprioception.
        config["framework"]["action_model"]["state_dim"] = 14
        config["framework"]["action_model"]["action_horizon"] = 16
        config["framework"]["planner"]["num_action_queries"] = 16
        config["datasets"]["vla_data"]["include_state"] = True
        config["datasets"]["vla_data"]["action_horizon"] = 16
        physical = config["framework"]["world_action_mot"]
        physical["interaction_mode"] = "joint"
        physical.update(
            {
                "world_hidden_size": 512,
                "action_hidden_size": 1024,
                "world_ffn_dim": 2048,
                "action_ffn_dim": 4096,
                "num_layers": 30,
                "num_attention_heads": 24,
                "attention_head_dim": 128,
                # The released sampler was configured with 20, while the
                # RoboDojo client intentionally evaluates it with 10 steps.
                "num_inference_timesteps": 20,
            }
        )
        physical.pop("layerwise_planner_coupling", None)
        physical.pop("action_prediction_type", None)
        physical.pop("action_velocity_target", None)
        physical.pop("repeated_diffusion_steps", None)
        physical.pop("jit_t_eps", None)
        config["framework"]["qwenvl"].pop("truncate_vlm_layers", None)
        # Released 50k checkpoints predate the dense-current/sparse-future
        # contract. Their saved configs use pooled 120-token current and
        # future grids, even though this fixture starts from today's YAML.
        config["framework"]["dino"].pop("current_dino_pool", None)

        run_dir = tmp_path / run_name
        checkpoint = (
            run_dir / "checkpoints/steps_50000_pytorch_model.pt"
        )
        serialized = yaml.safe_dump(config)
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(
            serialized,
            encoding="utf-8",
        )
        (run_dir / "config.full.yaml").write_text(
            serialized,
            encoding="utf-8",
        )
        (run_dir / "dataset_statistics.json").write_text(
            json.dumps(statistics),
            encoding="utf-8",
        )
        _write_causal_mot_shape_checkpoint(checkpoint, config)

        summary = verifier.verify(str(checkpoint))
        assert (
            summary["mot_weight_contract"]["version"]
            == "legacy_shared_context_state_world_v1"
        )
        assert summary["mot_weight_contract"]["num_layers"] == 30
        assert summary["text_planning_enabled"] is expected_text
        assert summary["released_50k_contract"]["run"] == run_name
        assert summary["released_50k_contract"]["text"] is expected_text
        assert summary["mot_action_recipe"]["prediction_type"] == "velocity"
        assert (
            summary["mot_action_recipe"]["velocity_target"]
            == "noise_minus_clean"
        )
        assert summary["mot_action_recipe"]["loss"] == "velocity_mse"
        assert summary["mot_action_recipe"]["repeated_diffusion_steps"] == 1
