from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
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

    class FakeModel:
        def __init__(self) -> None:
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
