"""PR2: Qwen2.5 text-only interface for JointFlow.

复用:
- transformers AutoModelForCausalLM / AutoTokenizer

说明:
该 wrapper 直接实例化 text-only Qwen2/Qwen2.5，使用 inputs_embeds + 4D
block mask，不经过 starVLA/model/modules/vlm/__init__.py。
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn


def _resolve_torch_dtype(value):
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    value = str(value).strip().lower()
    if value in {"auto", "none", "null"}:
        return "auto" if value == "auto" else None
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "torch.float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
    }
    if value not in aliases:
        raise ValueError(f"Unsupported Qwen torch_dtype={value!r}.")
    return aliases[value]


######### // code // ##########
# 中文注释：Qwen2 text-only wrapper，负责 text token embedding 和 joint sequence forward。
# embed_text 输入 instructions(List[str])，输出 text_embeds [B,L,D]、valid_lens [B]、pad_mask [B,L]。
# forward 输入 joint embeddings [B,T,D]、4D additive mask [B,1,T,T]、position_ids [B,T]；
# 输出 hidden [B,T,D]，供 action/visual flow head 使用 query token 位置 hidden。
class _QWen2_Text_Interface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        qwen_cfg = config.framework.get("qwenvl", {})
        self.max_text_length = int(qwen_cfg.get("max_text_length", 256))
        self.model_id = qwen_cfg.get("base_vlm", "playground/Pretrained_models/Qwen2.5-0.5B")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        attn_implementation = qwen_cfg.get("attn_implementation", "eager")
        if attn_implementation == "flash_attention_2":
            raise ValueError("QwenJointFlow requires eager/sdpa attention; flash_attention_2 cannot express block masks.")
        self.fp32_forward = bool(qwen_cfg.get("fp32_forward", True))
        requested_dtype = _resolve_torch_dtype(qwen_cfg.get("torch_dtype", "float32"))

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            attn_implementation=attn_implementation,
            torch_dtype=requested_dtype,
            trust_remote_code=bool(qwen_cfg.get("trust_remote_code", False)),
        )
        if self.fp32_forward:
            model = model.to(dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=bool(qwen_cfg.get("trust_remote_code", False)),
        )
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.backbone = model.model
        self.tokenizer = tokenizer
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.attn_implementation = attn_implementation
        self.requested_torch_dtype = str(requested_dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _format_instruction(self, instruction: str) -> str:
        data_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        prompt = data_cfg.get("CoT_prompt", None) if data_cfg is not None and hasattr(data_cfg, "get") else None
        if prompt:
            return prompt.replace("{instruction}", instruction)
        return instruction

    def embed_text(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        instructions = [self._format_instruction(text) for text in instructions]

        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device, dtype=torch.bool)
        valid_lens = attention_mask.long().sum(dim=1)
        embeds = self.backbone.embed_tokens(input_ids)
        return embeds, valid_lens, attention_mask

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask_4d: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = self.dtype
        if self.fp32_forward:
            target_dtype = torch.float32
        inputs_embeds = inputs_embeds.to(dtype=target_dtype)
        # 中文注释：mask dtype 跟随计算 dtype。fp32_forward 时仍是 fp32（行为不变）；
        # 若改用 bf16 backbone（fp32_forward=false + torch_dtype=bfloat16），mask 也用 bf16，
        # 这样 SDPA 的 q/k/v 与 attn_mask dtype 一致，才能走 bf16 的 fused(flash/mem-efficient) kernel。
        # block mask 的负值是 -1e4（不是 finfo.min），在 bf16 下也能正常表示、softmax 后≈0。
        attention_mask_4d = attention_mask_4d.to(dtype=target_dtype)
        ctx = (
            torch.autocast(device_type=inputs_embeds.device.type, enabled=False)
            if self.fp32_forward and inputs_embeds.device.type in {"cuda", "cpu"}
            else nullcontext()
        )
        with ctx:
            outputs = self.backbone(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                use_cache=False,
            )
        return outputs.last_hidden_state
######### // code // ##########
