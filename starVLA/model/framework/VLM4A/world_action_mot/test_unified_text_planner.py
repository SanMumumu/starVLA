from __future__ import annotations

from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.model.framework.VLM4A.QwenWorldActionMoT import QwenWorldActionMoT


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


def test_teacher_forcing_is_one_context_text_eos_world_action_pass() -> None:
    harness = SimpleNamespace(
        text_planning_enabled=True,
        text_loss_weight=0.25,
        text_supervision={
            "enabled": True,
            "allow_missing_annotations": True,
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
    harness._current_views = lambda _example: []
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
            "completed_subtask_text": "",
            "text_annotation_available": True,
        },
        {
            "lang": "close the drawer",
            "text_annotation_available": False,
        },
    ]

    action, world, text_loss, count = harness._teacher_forced_planner_hidden(
        examples
    )

    assert count == 1
    assert float(text_loss) == 3.0
    assert world.squeeze(-1).tolist() == [[50.0, 50.0], [50.0, 50.0]]
    assert action.squeeze(-1).tolist() == [[51.0, 51.0], [51.0, 51.0]]
    # Full chat template supplied the assistant EOS before the query suffix.
    assert harness.seen_input_ids[0, -5:].tolist() == [99, 50, 50, 51, 51]
    assert harness.seen_input_ids[1, -5:].tolist() == [99, 50, 50, 51, 51]
    # Annotated row supervises response token + EOS only; missing row has no NTP.
    assert harness.seen_labels[0][harness.seen_labels[0] != -100].tolist() == [
        20,
        99,
    ]
    assert not harness.seen_labels[1].ne(-100).any()
    assert not harness.seen_labels[:, -4:].ne(-100).any()
    # One physical backbone pass, with WORLD positions strictly before ACTION.
    assert harness.seen_world_mask.nonzero()[:, 1].max() < (
        harness.seen_action_mask.nonzero()[:, 1].min()
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
    harness._current_views = lambda _example: []
    for name in (
        "_planner_user_message",
        "_text_prompt",
        "_token_id_set",
        "_generated_sequence_inputs",
        "_append_query_suffix",
        "_extract_plans",
        "_generated_planner_hidden",
    ):
        _bind(harness, name)

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
