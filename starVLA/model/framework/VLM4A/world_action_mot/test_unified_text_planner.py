from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.QwenWorldActionMoT import (
    QwenWorldActionMoT,
    _truncate_text_backbone,
)


class _Batch(dict):
    def to(self, device):
        for key, value in tuple(self.items()):
            if torch.is_tensor(value):
                self[key] = value.to(device)
        return self


def _left_pad(rows: list[list[int]], pad: int = 0) -> _Batch:
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, width - len(row) :] = torch.tensor(row)
        attention[index, width - len(row) :] = 1
    return _Batch(input_ids=input_ids, attention_mask=attention)


class _TeacherProcessor:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(
            padding_side="right",
            pad_token_id=0,
            eos_token_id=99,
        )
        self.calls = []

    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        self.calls.append((messages, add_generation_prompt))
        if add_generation_prompt:
            return _left_pad([[10, 11, 12] for _ in messages])
        rows = []
        for conversation in messages:
            response = conversation[1]["content"][0]["text"]
            # Real Qwen templates may place a newline token after <|im_end|>.
            rows.append([10, 11, 12] + ([20] if response else []) + [99, 77])
        return _left_pad(rows)


def _bind(harness, name: str):
    descriptor = QwenWorldActionMoT.__dict__[name]
    method = getattr(QwenWorldActionMoT, name)
    setattr(
        harness,
        name,
        method if isinstance(descriptor, staticmethod) else MethodType(method, harness),
    )


def test_first_28_vlm_layers_are_kept_and_configs_are_aligned() -> None:
    text_config = SimpleNamespace(num_hidden_layers=32)
    language_model = torch.nn.Module()
    language_model.layers = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2) for _ in range(32)]
    )
    language_model.config = SimpleNamespace(num_hidden_layers=32)
    multimodal = torch.nn.Module()
    multimodal.language_model = language_model
    multimodal.config = SimpleNamespace(num_hidden_layers=32)
    full_model = torch.nn.Module()
    full_model.model = multimodal
    full_model.config = SimpleNamespace(
        num_hidden_layers=32,
        text_config=text_config,
    )

    original = _truncate_text_backbone(full_model, 28)

    assert original == 32
    assert len(language_model.layers) == 28
    assert full_model.config.num_hidden_layers == 28
    assert full_model.config.text_config.num_hidden_layers == 28
    assert multimodal.config.num_hidden_layers == 28
    assert language_model.config.num_hidden_layers == 28


def test_extract_plans_preserves_one_query_tensor_per_vlm_layer() -> None:
    harness = SimpleNamespace(num_world_queries=2, num_action_queries=3)
    _bind(harness, "_extract_plans")
    world_mask = torch.tensor([[False, True, True, False, False, False]])
    action_mask = torch.tensor([[False, False, False, True, True, True]])
    hidden_layers = [
        torch.full((1, 6, 4), float(layer))
        for layer in range(28)
    ]

    action, world = harness._extract_plans(
        hidden_layers,
        world_mask,
        action_mask,
        batch_size=1,
    )

    assert len(action) == len(world) == 28
    assert action[17].shape == (1, 3, 4)
    assert world[17].shape == (1, 2, 4)
    torch.testing.assert_close(action[17], torch.full_like(action[17], 17.0))
    torch.testing.assert_close(world[17], torch.full_like(world[17], 17.0))


def test_query_suffix_uses_exact_positions_and_extends_qwen35_sequence_fields() -> None:
    inputs = _Batch(
        input_ids=torch.tensor([[10, 11, 12], [0, 20, 21]]),
        attention_mask=torch.tensor([[1, 1, 1], [0, 1, 1]]),
        mm_token_type_ids=torch.tensor([[1, 1, 0], [0, 2, 0]]),
        token_type_ids=torch.tensor([[3, 3, 0], [0, 4, 0]]),
        position_ids=torch.arange(3).repeat(2, 1),
        cache_position=torch.arange(3),
    )

    world_mask, action_mask = QwenWorldActionMoT._append_query_suffix(
        inputs,
        world_token_id=50,
        action_token_id=51,
        n_world=2,
        n_action=3,
    )

    assert inputs["input_ids"].tolist() == [
        [10, 11, 12, 50, 50, 51, 51, 51],
        [0, 20, 21, 50, 50, 51, 51, 51],
    ]
    assert world_mask.sum(dim=1).tolist() == [2, 2]
    assert action_mask.sum(dim=1).tolist() == [3, 3]
    assert not bool((world_mask & action_mask).any())
    assert world_mask[:, 3:5].all()
    assert action_mask[:, 5:].all()
    assert inputs["mm_token_type_ids"].tolist() == [
        [1, 1, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
    ]
    assert inputs["token_type_ids"].tolist() == [
        [3, 3, 0, 0, 0, 0, 0, 0],
        [0, 4, 0, 0, 0, 0, 0, 0],
    ]
    assert "position_ids" not in inputs
    assert "cache_position" not in inputs


def test_extract_plans_rejects_query_mask_cardinality_before_reshape() -> None:
    harness = SimpleNamespace(num_world_queries=16, num_action_queries=16)
    _bind(harness, "_extract_plans")
    hidden = torch.zeros(1, 182, 2048)
    world_mask = torch.zeros(1, 182, dtype=torch.bool)
    action_mask = torch.ones(1, 182, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"action=\[182\] expected=16"):
        harness._extract_plans(
            hidden,
            world_mask,
            action_mask,
            batch_size=1,
        )


def test_teacher_forcing_is_one_context_text_eos_world_action_pass() -> None:
    harness = SimpleNamespace(
        text_planning_enabled=True,
        text_loss_weight=0.25,
        text_supervision={
            "enabled": True,
            "subtask_field": "subtask_text",
            "completed_subtask_field": "completed_subtask_text",
            "prompt_template": "{instruction}\\nPlan the current subtask.",
            "response_template": "Subtask: {subtask_text}",
        },
        qwen_vl_interface=SimpleNamespace(
            processor=_TeacherProcessor(),
            model=SimpleNamespace(
                generation_config=SimpleNamespace(eos_token_id=[99])
            ),
        ),
        world_placeholder_id=50,
        action_placeholder_id=51,
        num_world_queries=2,
        num_action_queries=2,
        device=torch.device("cpu"),
        config=OmegaConf.create(
            {"framework": {"qwenvl": {"enable_thinking": False}}}
        ),
    )
    harness._planner_current_views = lambda _example: []
    harness._prepare_mem_vision_inputs = lambda inputs, _examples: inputs
    harness._activate_mem_vision_inputs = lambda _inputs: nullcontext()
    for name in (
        "_planner_user_message",
        "_text_prompt",
        "_text_target",
        "_token_id_set",
        "_validate_prompt_prefix",
        "_truncate_assistant_at_eos",
        "_assistant_labels",
        "_append_query_suffix",
        "_extract_plans",
        "_teacher_forced_planner_hidden",
    ):
        _bind(harness, name)

    def run_backbone(_self, inputs, world_mask, action_mask):
        _self.seen_input_ids = inputs["input_ids"].clone()
        _self.seen_world_mask = world_mask.clone()
        _self.seen_action_mask = action_mask.clone()
        return inputs["input_ids"].float().unsqueeze(-1)

    def text_loss(_self, hidden, labels):
        _self.seen_labels = labels.clone()
        return hidden.new_tensor(3.0)

    harness._run_planner_backbone = MethodType(run_backbone, harness)
    harness._next_token_loss = MethodType(text_loss, harness)
    examples = [
        {
            "lang": "stack the bowls",
            "subtask_text": "grasp the blue bowl",
            "completed_subtask_text": "approached the bowls",
        },
        {
            "lang": "close the drawer",
            "subtask_text": "push the drawer closed",
            "completed_subtask_text": "grasped the handle",
        },
    ]

    action, world, text_loss, count = harness._teacher_forced_planner_hidden(
        examples
    )

    assert count == 2
    assert float(text_loss) == 3.0
    assert world.squeeze(-1).tolist() == [[50.0, 50.0], [50.0, 50.0]]
    assert action.squeeze(-1).tolist() == [[51.0, 51.0], [51.0, 51.0]]
    # Full chat template supplied the assistant EOS before the query suffix.
    assert harness.seen_input_ids[0, -5:].tolist() == [99, 50, 50, 51, 51]
    assert harness.seen_input_ids[1, -5:].tolist() == [99, 50, 50, 51, 51]
    # Every row supervises its response token and the terminating EOS.
    assert harness.seen_labels[0][harness.seen_labels[0] != -100].tolist() == [
        20,
        99,
    ]
    assert harness.seen_labels[1][harness.seen_labels[1] != -100].tolist() == [
        20,
        99,
    ]
    assert not harness.seen_labels[:, -4:].ne(-100).any()
    # One physical backbone pass, with WORLD positions strictly before ACTION.
    assert harness.seen_world_mask.nonzero()[:, 1].max() < (
        harness.seen_action_mask.nonzero()[:, 1].min()
    )


def test_text_target_rejects_any_missing_annotation() -> None:
    harness = SimpleNamespace(
        text_planning_enabled=True,
        text_supervision={
            "subtask_field": "subtask_text",
            "completed_subtask_field": "completed_subtask_text",
            "response_template": (
                "Current subtask: {subtask_text}\n"
                "Completed subtask: {completed_subtask_text}"
            ),
        },
    )
    _bind(harness, "_text_target")

    with pytest.raises(ValueError, match="non-empty text annotations"):
        harness._text_target(
            {
                "subtask_text": "grasp the bowl",
                "completed_subtask_text": "",
                "dataset_name": "RoboDojo_lerobot_v21_language_v2",
                "trajectory_id": 7,
                "base_index": 42,
            }
        )


def test_event_keep_removes_eos_before_physical_queries() -> None:
    harness = SimpleNamespace(
        keep_token_id=42,
        update_token_id=43,
        qwen_vl_interface=SimpleNamespace(
            processor=SimpleNamespace(
                tokenizer=SimpleNamespace(pad_token_id=0, eos_token_id=99)
            )
        ),
    )
    for name in (
        "_token_id_set",
        "_event_decision_positions",
        "_event_remove_keep_eos",
    ):
        _bind(harness, name)
    inputs = _left_pad(
        [
            [10, 11, 42, 99],
            [10, 11, 43, 20, 99],
        ]
    )

    trimmed = harness._event_remove_keep_eos(
        inputs,
        torch.tensor([2, 2]),
        ["KEEP", "UPDATE"],
    )

    assert trimmed["input_ids"][0][trimmed["attention_mask"][0].bool()].tolist() == [
        10,
        11,
        42,
    ]
    assert trimmed["input_ids"][1][trimmed["attention_mask"][1].bool()].tolist() == [
        10,
        11,
        43,
        20,
        99,
    ]


def test_event_schedule_and_update_parser_follow_training_contract() -> None:
    harness = SimpleNamespace(
        event_schedule_points=[
            (0, 0.0),
            (30000, 0.0),
            (40000, 0.2),
            (50000, 0.5),
        ]
    )
    _bind(harness, "_event_sampling_probability")
    assert harness._event_sampling_probability(29999) == 0.0
    assert harness._event_sampling_probability(35000) == pytest.approx(0.1)
    assert harness._event_sampling_probability(45000) == pytest.approx(0.35)
    assert harness._event_sampling_probability(50000) == pytest.approx(0.5)

    memory_add, subtask = QwenWorldActionMoT._parse_event_update_body(
        "Memory Add: Opened the drawer.\n"
        "Current Subtask: Pick up the cup."
    )
    assert memory_add == "Opened the drawer."
    assert subtask == "Pick up the cup."
    with pytest.raises(ValueError, match="Memory Add"):
        QwenWorldActionMoT._parse_event_update_body("Pick up the cup")


def test_event_semantic_objective_splits_decision_and_update_body_ce() -> None:
    decisions = ["UPDATE", "KEEP", "KEEP", "UPDATE", "KEEP", "KEEP"]
    ids = torch.tensor(
        [
            [10, 11, 43, 20, 99] if decision == "UPDATE"
            else [10, 11, 42, 99, 99]
            for decision in decisions
        ],
        dtype=torch.long,
    )
    inputs = _Batch(input_ids=ids, attention_mask=torch.ones_like(ids))
    output_head = torch.nn.Linear(8, 128, bias=False)
    harness = SimpleNamespace(
        event_data_config={
            "sampler": {"per_device_batch_size": 6, "update_per_batch": 2}
        },
        keep_token_id=42,
        update_token_id=43,
        device=torch.device("cpu"),
        qwen_vl_interface=SimpleNamespace(
            model=SimpleNamespace(get_output_embeddings=lambda: output_head)
        ),
    )
    harness._event_decision = lambda example: example["decision"]
    harness._event_ground_truth_response = lambda example: example["decision"]
    harness._event_chat_inputs = lambda _examples, _responses: (
        inputs,
        torch.tensor([2] * 6),
    )
    harness._event_decision_positions = lambda *_args: [2] * 6
    harness._run_text_backbone = lambda _inputs: torch.randn(6, 5, 8)
    _bind(harness, "_next_token_loss")
    _bind(harness, "_event_semantic_objective")

    result = harness._event_semantic_objective(
        [{"decision": decision} for decision in decisions]
    )

    assert torch.isfinite(result["loss"])
    assert float(result["sample_count"]) == 6.0
    assert float(result["update_count"]) == 2.0
    assert float(result["decision_loss"]) > 0.0
    assert float(result["body_loss"]) > 0.0


def test_event_initial_cache_forces_update_over_constrained_keep_logit() -> None:
    class _DecisionHead(torch.nn.Module):
        def forward(self, hidden):
            logits = hidden.new_zeros((hidden.shape[0], 64))
            logits[:, 42] = 10.0
            logits[:, 43] = -10.0
            return logits

    harness = SimpleNamespace(
        keep_token_id=42,
        update_token_id=43,
        event_semantic_fields={"cache_valid": "semantic_cache_valid"},
        qwen_vl_interface=SimpleNamespace(
            model=SimpleNamespace(get_output_embeddings=lambda: _DecisionHead())
        ),
    )
    harness._event_prompt_inputs = lambda _examples: _Batch(
        input_ids=torch.ones(2, 3, dtype=torch.long),
        attention_mask=torch.ones(2, 3, dtype=torch.long),
    )
    harness._run_text_backbone = lambda _inputs: torch.zeros(2, 3, 8)
    harness._event_generate_update_bodies = lambda examples: [
        "Memory Add: None.\nCurrent Subtask: Start." for _ in examples
    ]
    _bind(harness, "_event_generate_states")

    decisions, bodies = harness._event_generate_states(
        [
            {"semantic_cache_valid": False},
            {"semantic_cache_valid": True},
        ]
    )

    assert decisions == ["UPDATE", "KEEP"]
    assert bodies == ["Memory Add: None.\nCurrent Subtask: Start.", ""]


def test_event_scheduled_condition_uses_prediction_without_gt_fallback() -> None:
    harness = SimpleNamespace(
        device=torch.device("cpu"),
        keep_token="<KEEP>",
        update_token="<UPDATE>",
        event_schedule_points=[(0, 1.0)],
    )
    harness._event_decision = lambda _example: "UPDATE"
    harness._event_ground_truth_response = lambda _example: (
        "<UPDATE>\nMemory Add: GT.\nCurrent Subtask: GT."
    )
    harness._event_generate_states = lambda _examples: (
        ["KEEP"] * len(_examples),
        [""] * len(_examples),
    )
    harness._event_generated_response = lambda decision, _body: f"<{decision}>"

    def physical(_examples, decisions, responses):
        harness.seen_decisions = decisions
        harness.seen_responses = responses
        return torch.zeros(len(_examples), 1, 2), torch.zeros(len(_examples), 1, 2)

    harness._event_physical_hidden_from_responses = physical
    _bind(harness, "_event_sampling_probability")
    _bind(harness, "_event_training_physical_hidden")

    _action, _world, metrics = harness._event_training_physical_hidden(
        [{}, {}, {}], global_step=1
    )

    assert harness.seen_decisions == ["KEEP", "KEEP", "KEEP"]
    assert harness.seen_responses == ["<KEEP>", "<KEEP>", "<KEEP>"]
    assert float(metrics["scheduled_count"]) == 3.0


def test_maintained_robodojo_text_and_mem_recipes() -> None:
    train_dir = REPO_ROOT / "examples/RoboDojo/train_files/released_rynn50k"
    job_dir = REPO_ROOT / "执行脚本/Robodojo/released_rynn50k"

    event = OmegaConf.load(train_dir / "rynn_base_text_h25_mem_50k.yaml")
    text = event.framework.planner.text_supervision
    annotations = event.datasets.vla_data.text_annotations
    assert bool(text.enabled)
    assert str(text.mode) == "event_driven_memory_ntp"
    assert str(text.keep_token) == "<KEEP>"
    assert str(text.update_token) == "<UPDATE>"
    assert not bool(text.history.enabled)
    assert "mem_vision_encoder" not in event.framework.qwenvl
    assert float(event.framework.world_action_mot.text_loss_weight) == 0.005
    assert str(event.framework.world_action_mot.interaction_mode) == "base"
    assert str(event.framework.world_action_mot.action_precision_mode) == "fp32_shell"
    assert bool(annotations.enabled)
    assert bool(annotations.event_memory.enabled)
    assert int(annotations.event_memory.semantic_offset) == -10
    assert int(annotations.event_memory.replan_interval) == 10
    assert event.datasets.vla_data.data_mix == "robodojo_v21_language"
    assert int(event.framework.action_model.action_horizon) == 25
    assert int(event.framework.planner.num_action_queries) == 25
    assert int(event.datasets.vla_data.per_device_batch_size) == 12
    assert int(event.trainer.expected_global_batch_size) == 768
    assert int(event.trainer.max_train_steps) == 50000

    event_bf16 = OmegaConf.load(
        train_dir / "rynn_base_text_h25_mem_bf16_50k.yaml"
    )
    assert str(event_bf16.framework.world_action_mot.action_precision_mode) == (
        "inherit"
    )
    assert str(event_bf16.run_id).endswith(
        "base_text_h25_eventmem_ntp_nohist_bf16_50k"
    )
    fp32_payload = OmegaConf.to_container(event, resolve=True)
    bf16_payload = OmegaConf.to_container(event_bf16, resolve=True)
    fp32_payload["run_id"] = bf16_payload["run_id"]
    fp32_payload["framework"]["reproduction_profile"] = bf16_payload[
        "framework"
    ]["reproduction_profile"]
    fp32_payload["framework"]["world_action_mot"][
        "action_precision_mode"
    ] = "inherit"
    assert fp32_payload == bf16_payload

    event_job = OmegaConf.to_container(
        OmegaConf.load(job_dir / "job_base_text_50k.yaml"),
        resolve=False,
    )
    assert int(event_job["REQUIRED"]["WORKER_MIN_NUM"]) == 8
    assert int(event_job["REQUIRED"]["WORKER_MAX_NUM"]) == 8
    assert int(event_job["REQUIRED"]["GPU_PER_WORKER"]) == 8
    assert event_job["OPTIONAL"]["DOCKER_IMAGE"] == (
        "docker.hobot.cc/imagesys/starvla-rynn-ppu:v1.2"
    )
    assert str(event_job["REQUIRED"]["RUN_SCRIPTS"]).endswith(
        "released_rynn50k/launch.sh base_text_h25_mem"
    )

    event_bf16_job = OmegaConf.to_container(
        OmegaConf.load(job_dir / "job_base_text_bf16_50k.yaml"),
        resolve=False,
    )
    assert str(event_bf16_job["REQUIRED"]["RUN_SCRIPTS"]).endswith(
        "released_rynn50k/launch.sh base_text_h25_mem_bf16"
    )

    history = OmegaConf.load(train_dir / "rynn_base_history_h25_mem_50k.yaml")
    assert not bool(history.framework.planner.text_supervision.enabled)
    assert float(history.framework.world_action_mot.text_loss_weight) == 0.0
    assert bool(history.framework.qwenvl.mem_vision_encoder.enabled)
    assert bool(history.framework.planner.text_supervision.history.enabled)
    assert list(
        history.datasets.vla_data.text_annotations.history.frame_offsets
    ) == [-100, -80, -60, -40, -20]
    assert int(
        history.datasets.vla_data.text_annotations.history.memory_offset
    ) == -100
    assert int(history.framework.action_model.action_horizon) == 25

    control = OmegaConf.load(train_dir / "rynn_base_h25_50k.yaml")
    for field in (
        "architecture",
        "interaction_mode",
        "world_attention_mask_mode",
        "world_hidden_size",
        "action_hidden_size",
        "world_ffn_dim",
        "action_ffn_dim",
        "num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "action_prediction_type",
        "action_velocity_target",
        "jit_t_eps",
        "repeated_diffusion_steps",
        "action_loss_weight",
        "world_loss_weight",
    ):
        assert history.framework.world_action_mot[field] == (
            control.framework.world_action_mot[field]
        )

    runbook = (job_dir / "run.sh").read_text(encoding="utf-8")
    history_submit = "job_base_history_50k.yaml"
    event_submit = "job_base_text_50k.yaml"
    event_bf16_submit = "job_base_text_bf16_50k.yaml"
    assert history_submit in runbook
    assert event_submit in runbook
    assert event_bf16_submit in runbook
    assert (
        runbook.index(history_submit)
        < runbook.index(event_submit)
        < runbook.index(event_bf16_submit)
    )


def test_disabled_text_planning_keeps_the_original_planner_path() -> None:
    harness = SimpleNamespace(text_planning_enabled=False)
    harness._planner_hidden = lambda _examples: (
        torch.ones(2, 2, 3),
        torch.full((2, 2, 3), 2.0),
    )
    _bind(harness, "_teacher_forced_planner_hidden")

    action, world, text_loss, count = harness._teacher_forced_planner_hidden(
        [{}, {}]
    )

    assert action.unique().tolist() == [1.0]
    assert world.unique().tolist() == [2.0]
    assert float(text_loss) == 0.0
    assert count == 0


def test_disabled_history_keeps_the_original_single_image_message() -> None:
    harness = SimpleNamespace(text_history_enabled=False)
    harness._planner_current_views = lambda _example: ["current-image"]
    _bind(harness, "_planner_user_message")

    message = harness._planner_user_message({}, "do the task")

    assert message == {
        "role": "user",
        "content": [
            {"type": "image", "image": "current-image"},
            {"type": "text", "text": "do the task"},
        ],
    }


def test_history_prompt_orders_images_and_exposes_only_previous_memory() -> None:
    harness = SimpleNamespace(
        text_history_enabled=True,
        text_history={
            "finished_task_list_field": "finished_task_list",
        },
        text_supervision={
            "prompt_template": (
                "Task: {instruction}\n"
                "Previous Finished Task List: {finished_task_list}"
            )
        },
    )
    harness._planner_history_views = lambda _example: ["history-1", "history-2"]
    harness._planner_current_views = lambda _example: ["current"]
    _bind(harness, "_planner_user_message")
    _bind(harness, "_text_prompt")

    example = {
        "lang": "put away the objects",
        "finished_task_list": "Place the figurine on the stand.",
        # Current labels must never leak into the user prompt.
        "subtask_text": "place the clock",
        "completed_subtask_text": "all current labels",
    }
    prompt = harness._text_prompt(example)
    message = harness._planner_user_message(example, prompt)

    assert prompt == (
        "Task: put away the objects\n"
        "Previous Finished Task List: Place the figurine on the stand."
    )
    assert [part["type"] for part in message["content"]] == [
        "text",
        "image",
        "text",
        "image",
        "text",
        "image",
        "text",
    ]
    assert [
        part["image"]
        for part in message["content"]
        if part["type"] == "image"
    ] == ["history-1", "history-2", "current"]
    assert "all current labels" not in message["content"][-1]["text"]


def test_planner_history_downsampling_matches_deployment_inter_area() -> None:
    harness = SimpleNamespace(
        text_history_enabled=True,
        text_history={
            "image_field": "planner_history_images",
            "history_image_size": [4, 4],
        },
        text_history_frame_offsets=(-12,),
    )
    _bind(harness, "_view_list")
    _bind(harness, "_resize_planner_views")
    _bind(harness, "_planner_history_views")
    source = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    [actual] = harness._planner_history_views(
        {"planner_history_images": [Image.fromarray(source)]}
    )
    expected = cv2.resize(source, (4, 4), interpolation=cv2.INTER_AREA)

    assert actual.size == (4, 4)
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_finished_task_list_is_monotonic_across_planner_updates() -> None:
    harness = SimpleNamespace(
        text_history_enabled=True,
        text_history={
            "finished_task_list_field": "finished_task_list",
            "finished_task_list_prefix": "Finished Task List:",
            "empty_finished_task_list": "None",
        },
        text_planning_enabled=True,
        text_supervision={
            "subtask_field": "subtask_text",
            "completed_subtask_field": "completed_subtask_text",
            "response_template": (
                "Current subtask: {subtask_text}\n"
                "Finished Task List: {completed_subtask_text}"
            ),
        },
    )
    for name in (
        "_finished_task_list_from_planner_text",
        "_finished_task_items",
        "_merge_finished_task_lists",
        "_reconcile_generated_planner_text",
        "_text_target",
    ):
        _bind(harness, name)

    reconciled = harness._reconcile_generated_planner_text(
        {
            "finished_task_list": (
                "Place the figurine on the stand. "
                "Place the alarm clock on the drawer."
            )
        },
        (
            "Current subtask: Push the keyboard into the frame.\n"
            "Finished Task List: Place the mouse on the mouse pad. "
            "Place the figurine on the stand."
        ),
    )

    assert reconciled == (
        "Current subtask: Push the keyboard into the frame.\n"
        "Finished Task List: Place the figurine on the stand. "
        "Place the alarm clock on the drawer. "
        "Place the mouse on the mouse pad."
    )
    retained = harness._reconcile_generated_planner_text(
        {"finished_task_list": "Place the alarm clock on the drawer."},
        "Current subtask: Place the mouse.\nFinished Task List: None",
    )
    assert retained.endswith(
        "Finished Task List: Place the alarm clock on the drawer."
    )
    target = harness._text_target(
        {
            "lang": "place all objects",
            "subtask_text": "Place C",
            # The raw target reordered A/B; state-update supervision preserves
            # the memory order and only appends C.
            "completed_subtask_text": "Place B. Place A. Place C.",
            "finished_task_list": "Place A. Place B.",
        }
    )
    assert target.endswith("Finished Task List: Place A. Place B. Place C.")


def test_selected_next_token_loss_matches_causal_shift() -> None:
    head = torch.nn.Linear(3, 4, bias=False)
    harness = SimpleNamespace(
        qwen_vl_interface=SimpleNamespace(
            model=SimpleNamespace(get_output_embeddings=lambda: head)
        )
    )
    _bind(harness, "_next_token_loss")
    hidden = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 2, 1, -100]])

    actual = harness._next_token_loss(hidden, labels)
    expected = F.cross_entropy(head(hidden[0, :2]).float(), torch.tensor([2, 1]))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert hidden.grad is not None
    assert hidden.grad[0, 2:].abs().sum() == 0


class _GenerationTokenizer:
    padding_side = "left"
    pad_token_id = 0
    eos_token_id = 99

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token) for token in token_ids if token not in {0, 99})


def test_generated_text_is_trimmed_at_eos_before_world_action_queries() -> None:
    tokenizer = _GenerationTokenizer()
    harness = SimpleNamespace(
        qwen_vl_interface=SimpleNamespace(
            processor=SimpleNamespace(tokenizer=tokenizer),
            model=SimpleNamespace(
                generation_config=SimpleNamespace(eos_token_id=[99])
            ),
        )
    )
    _bind(harness, "_token_id_set")
    _bind(harness, "_generated_sequence_inputs")
    inputs = _Batch(
        input_ids=torch.tensor([[0, 10, 11], [12, 13, 14]]),
        attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        mm_token_type_ids=torch.tensor([[0, 1, 0], [2, 2, 0]]),
        token_type_ids=torch.tensor([[0, 3, 0], [4, 4, 0]]),
        position_ids=torch.zeros(2, 3, dtype=torch.long),
    )
    generated = torch.tensor(
        [
            [0, 10, 11, 20, 99, 0],
            [12, 13, 14, 21, 22, 99],
        ]
    )

    rebuilt, text = harness._generated_sequence_inputs(inputs, generated)

    assert text == ["20", "21 22"]
    assert rebuilt["input_ids"].tolist() == [
        [0, 0, 10, 11, 20, 99],
        [12, 13, 14, 21, 22, 99],
    ]
    assert rebuilt["attention_mask"].tolist() == [
        [0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ]
    assert rebuilt["mm_token_type_ids"].tolist() == [
        [0, 0, 1, 0, 0, 0],
        [2, 2, 0, 0, 0, 0],
    ]
    assert rebuilt["token_type_ids"].tolist() == [
        [0, 0, 3, 0, 0, 0],
        [4, 4, 0, 0, 0, 0],
    ]
    assert "position_ids" not in rebuilt


def test_generation_token_limit_still_inserts_eos_boundary() -> None:
    tokenizer = _GenerationTokenizer()
    harness = SimpleNamespace(
        qwen_vl_interface=SimpleNamespace(
            processor=SimpleNamespace(tokenizer=tokenizer),
            model=SimpleNamespace(
                generation_config=SimpleNamespace(eos_token_id=[99])
            ),
        )
    )
    _bind(harness, "_token_id_set")
    _bind(harness, "_generated_sequence_inputs")
    inputs = _Batch(
        input_ids=torch.tensor([[10, 11]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
    )

    rebuilt, text = harness._generated_sequence_inputs(
        inputs, torch.tensor([[10, 11, 20, 21]])
    )

    assert rebuilt["input_ids"].tolist() == [[10, 11, 20, 21, 99]]
    assert text == ["20 21"]


def test_inference_generates_to_eos_then_appends_world_and_action() -> None:
    class Processor:
        def __init__(self):
            self.tokenizer = _GenerationTokenizer()

        def apply_chat_template(self, messages, **_kwargs):
            assert len(messages) == 1
            return _Batch(
                input_ids=torch.tensor([[10, 11, 12]]),
                attention_mask=torch.ones(1, 3, dtype=torch.long),
            )

    class Model:
        generation_config = SimpleNamespace(eos_token_id=[99])

        @staticmethod
        def generate(**_kwargs):
            return torch.tensor([[10, 11, 12, 20, 21, 99]])

    harness = SimpleNamespace(
        config=OmegaConf.create(
            {"framework": {"qwenvl": {"enable_thinking": False}}}
        ),
        text_supervision={
            "prompt_template": "{instruction}",
            "max_new_tokens": 8,
            "do_sample": False,
        },
        qwen_vl_interface=SimpleNamespace(processor=Processor(), model=Model()),
        world_placeholder_id=50,
        action_placeholder_id=51,
        num_world_queries=2,
        num_action_queries=2,
        device=torch.device("cpu"),
    )
    harness._planner_current_views = lambda _example: []
    harness._prepare_mem_vision_inputs = lambda inputs, _examples: inputs
    for name in (
        "_planner_user_message",
        "_text_prompt",
        "_token_id_set",
        "_generated_sequence_inputs",
        "_generate_planner_sequence",
        "_append_query_suffix",
        "_extract_plans",
        "_generated_planner_hidden",
    ):
        _bind(harness, name)
    harness._activate_mem_vision_inputs = lambda _inputs: nullcontext()

    def run_backbone(_self, inputs, world_mask, action_mask):
        _self.seen_input_ids = inputs["input_ids"].clone()
        return inputs["input_ids"].float().unsqueeze(-1)

    harness._run_planner_backbone = MethodType(run_backbone, harness)
    action, world, text = harness._generated_planner_hidden(
        [{"lang": "pick up the cup"}]
    )

    assert text == ["20 21"]
    assert harness.seen_input_ids.tolist() == [
        [10, 11, 12, 20, 21, 99, 50, 50, 51, 51]
    ]
    assert world.squeeze(-1).tolist() == [[50.0, 50.0]]
    assert action.squeeze(-1).tolist() == [[51.0, 51.0]]


def test_cached_text_uses_training_sequence_with_current_image_and_eos() -> None:
    processor = _TeacherProcessor()
    harness = SimpleNamespace(
        config=OmegaConf.create(
            {"framework": {"qwenvl": {"enable_thinking": False}}}
        ),
        text_supervision={
            "prompt_template": "{instruction}\\nPlan the current subtask.",
        },
        qwen_vl_interface=SimpleNamespace(
            processor=processor,
            model=SimpleNamespace(
                generation_config=SimpleNamespace(eos_token_id=[99])
            ),
        ),
        world_placeholder_id=50,
        action_placeholder_id=51,
        num_world_queries=2,
        num_action_queries=2,
        device=torch.device("cpu"),
    )
    harness._planner_current_views = lambda _example: []
    harness._prepare_mem_vision_inputs = lambda inputs, _examples: inputs
    for name in (
        "_planner_user_message",
        "_text_prompt",
        "_token_id_set",
        "_validate_prompt_prefix",
        "_truncate_assistant_at_eos",
        "_append_query_suffix",
        "_extract_plans",
        "_planner_hidden_from_text",
    ):
        _bind(harness, name)

    def run_backbone(_self, inputs, world_mask, action_mask):
        _self.seen_input_ids = inputs["input_ids"].clone()
        return inputs["input_ids"].float().unsqueeze(-1)

    harness._run_planner_backbone = MethodType(run_backbone, harness)
    action, world = harness._planner_hidden_from_text(
        [{"lang": "stack the bowls"}],
        ["Current subtask: grasp bowl\nCompleted subtask: approach bowl"],
    )

    assert harness.seen_input_ids.tolist() == [
        [10, 11, 12, 20, 99, 50, 50, 51, 51]
    ]
    assert world.squeeze(-1).tolist() == [[50.0, 50.0]]
    assert action.squeeze(-1).tolist() == [[51.0, 51.0]]
    full_conversation, add_generation_prompt = processor.calls[0]
    assert add_generation_prompt is False
    assert (
        full_conversation[0][1]["content"][0]["text"]
        == "Current subtask: grasp bowl\nCompleted subtask: approach bowl"
    )


def test_mixed_batch_generates_only_cache_misses_then_replans_every_row() -> None:
    harness = SimpleNamespace()
    generated_batches = []
    planned_batches = []

    def generate(_self, examples):
        generated_batches.append([example["env"] for example in examples])
        return object(), [f"generated-{example['env']}" for example in examples]

    def plan(_self, examples, planner_text):
        planned_batches.append(
            ([example["env"] for example in examples], list(planner_text))
        )
        batch = len(examples)
        return torch.ones(batch, 2, 3), torch.full((batch, 2, 3), 2.0)

    def generate_all(_self, _examples):
        raise AssertionError("mixed cache batch must not regenerate every row")

    harness._generate_planner_sequence = MethodType(generate, harness)
    harness._planner_hidden_from_text = MethodType(plan, harness)
    harness._generated_planner_hidden = MethodType(generate_all, harness)
    _bind(harness, "_cached_or_generated_planner_hidden")

    action, world, text, refreshed = harness._cached_or_generated_planner_hidden(
        [{"env": 0}, {"env": 1}, {"env": 2}, {"env": 3}],
        ["cached-0", None, "cached-2", ""],
    )

    assert generated_batches == [[1, 3]]
    assert planned_batches == [
        (
            [0, 1, 2, 3],
            ["cached-0", "generated-1", "cached-2", "generated-3"],
        )
    ]
    assert text == ["cached-0", "generated-1", "cached-2", "generated-3"]
    assert refreshed == [False, True, False, True]
    assert action.shape == world.shape == (4, 2, 3)

    _action, _world, all_generated, all_refreshed = (
        harness._cached_or_generated_planner_hidden(
            [{"env": 4}, {"env": 5}],
            [None, None],
        )
    )
    assert generated_batches[-1] == [4, 5]
    assert planned_batches[-1] == (
        [4, 5],
        ["generated-4", "generated-5"],
    )
    assert all_generated == ["generated-4", "generated-5"]
    assert all_refreshed == [True, True]


def test_predict_action_returns_text_and_per_row_refresh_flags() -> None:
    class ActionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.rtc_guidance_supported = True

        def sample(self, **kwargs):
            self.sample_kwargs = kwargs
            batch = kwargs["action_plan"].shape[0]
            return torch.zeros(batch, 4, 3), None

    action_model = ActionModel()
    harness = SimpleNamespace(
        text_planning_enabled=True,
        action_model=action_model,
        device=torch.device("cpu"),
    )
    seen_cache = []

    def cached_planner(_self, examples, cached):
        seen_cache.append(cached)
        batch = len(examples)
        return (
            torch.ones(batch, 2, 5),
            torch.full((batch, 2, 5), 2.0),
            ["cached-0", "fresh-1"],
            [False, True],
        )

    harness._cached_or_generated_planner_hidden = MethodType(
        cached_planner,
        harness,
    )
    harness._encode_dino = lambda batch_views: torch.zeros(len(batch_views), 3, 7)
    harness._current_views = lambda _example: []
    harness._stack_state = lambda examples, dtype: torch.zeros(
        len(examples), 1, 3, dtype=dtype
    )
    _bind(harness, "predict_action")

    output = harness.predict_action(
        [{}, {}],
        cached_planner_texts=["cached-0", None],
        num_ddim_steps=6,
        prev_action_chunk_normalized=torch.zeros(2, 4, 3).numpy(),
        rtc_prefix_lengths=[4, 0],
        inference_delay=0,
        execution_horizon=4,
        prefix_attention_schedule="exp",
        max_guidance_weight=10.0,
    )

    assert seen_cache == [["cached-0", None]]
    assert output["planner_text"] == ["cached-0", "fresh-1"]
    assert output["planner_text_refreshed"] == [False, True]
    assert output["normalized_actions"].shape == (2, 4, 3)
    assert action_model.sample_kwargs["num_inference_steps"] == 6
    assert action_model.sample_kwargs["rtc_prefix_lengths"] == [4, 0]
    assert action_model.sample_kwargs["execution_horizon"] == 4
    assert action_model.sample_kwargs["prefix_attention_schedule"] == "exp"
    assert action_model.sample_kwargs["max_guidance_weight"] == 10.0
