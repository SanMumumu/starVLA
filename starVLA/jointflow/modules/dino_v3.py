"""PR1: Frozen DINOv3-S wrapper.

复用:
- torchvision transforms
- torch.hub / transformers AutoModel fallback

说明:
训练时 JointFlow 默认读取离线 DINOv3 特征；该模块主要用于预计算和
predict_action live 图像特征抽取。
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


def _apply_transform(image: Image.Image, transform):
    return transform(image)


######### // code // ##########
# 中文注释：冻结 DINOv3 ViT-S/16，输出 patch tokens。
# 输入：imgs [B*V,3,H,W]，已经按 ImageNet mean/std 标准化。
# 输出：patch_tokens [B*V,N_v,384]；224x224+patch16 时 N_v=196。
class DINOv3Backbone(nn.Module):
    def __init__(
        self,
        name: str = "dinov3_vits16",
        hf_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        repo_or_dir: str = "facebookresearch/dinov3",
        weights: str | None = None,
        loader: str = "auto",
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        trust_repo: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.hf_model_id = hf_model_id
        self.repo_or_dir = repo_or_dir
        self.weights = weights
        self.loader = loader
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_channels = int(embed_dim)
        self._hf_mode = False

        self.dino_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.body = self._load_body(trust_repo=trust_repo)
        self.body.eval()
        for param in self.body.parameters():
            param.requires_grad_(False)

    def _load_body(self, trust_repo: bool):
        errors: list[str] = []
        if self.loader in {"auto", "torchhub"}:
            try:
                kwargs = {}
                if self.weights:
                    kwargs["weights"] = self.weights
                source = "local" if Path(self.repo_or_dir).exists() else "github"
                return torch.hub.load(self.repo_or_dir, self.name, source=source, trust_repo=trust_repo, **kwargs)
            except Exception as exc:  # pragma: no cover - depends on local weights/network
                errors.append(f"torch.hub failed: {exc}")
                if self.loader == "torchhub":
                    raise RuntimeError("; ".join(errors)) from exc

        try:
            from transformers import AutoModel

            self._hf_mode = True
            return AutoModel.from_pretrained(self.hf_model_id, trust_remote_code=True)
        except Exception as exc:  # pragma: no cover - depends on installed transformers/model cache
            errors.append(f"HF AutoModel failed: {exc}")
            raise RuntimeError(
                "Unable to load DINOv3. Provide a local repo/weights or a cached HF model. " + " | ".join(errors)
            ) from exc

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        if not self._hf_mode:
            if hasattr(self.body, "forward_features"):
                out = self.body.forward_features(imgs)
                if isinstance(out, dict) and "x_norm_patchtokens" in out:
                    return out["x_norm_patchtokens"]
                if isinstance(out, dict) and "patch_tokens" in out:
                    return out["patch_tokens"]
            out = self.body(imgs)
        else:
            out = self.body(pixel_values=imgs)

        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                return out["x_norm_patchtokens"]
            if "last_hidden_state" in out:
                tokens = out["last_hidden_state"]
            else:
                raise RuntimeError(f"Unsupported DINOv3 output keys: {list(out.keys())}")
        elif hasattr(out, "feature_maps") and out.feature_maps:
            fmap = out.feature_maps[-1]
            return fmap.flatten(2).transpose(1, 2)
        elif hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        elif torch.is_tensor(out):
            tokens = out
        else:
            raise RuntimeError(f"Unsupported DINOv3 output type: {type(out)}")

        expected_patches = (self.image_size // self.patch_size) ** 2
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected DINO tokens [B,T,C], got {tuple(tokens.shape)}")
        if tokens.shape[1] > expected_patches:
            tokens = tokens[:, -expected_patches:, :]
        return tokens

    def prepare_dino_input(self, img_list: Sequence[Sequence[Image.Image]]) -> torch.Tensor:
        with ThreadPoolExecutor() as executor:
            image_tensors = torch.stack(
                [
                    torch.stack(
                        list(executor.map(lambda view: _apply_transform(view.convert("RGB"), self.dino_transform), views))
                    )
                    for views in img_list
                ]
            )

        bsz, num_view, channels, height, width = image_tensors.shape
        image_tensors = image_tensors.view(bsz * num_view, channels, height, width)
        return image_tensors.to(next(self.parameters()).device)
######### // code // ##########

