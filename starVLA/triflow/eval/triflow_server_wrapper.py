"""PR7: TriFlow policy-server wrapper (live DINO at eval; NO state).

复用（全部 import，不改源）:
- deployment.model_server.policy_wrapper.PolicyServerWrapper
  (加载 framework + 动作反归一化 + handshake metadata)
- starVLA.jointflow.modules.dino_v3.DINOv3Backbone (eval 现场提特征)
- starVLA.jointflow.data.mix_registry.register_jointflow_mixtures

为何子类化（对照 v1 jointflow_server_wrapper）:
1) 必须先 import triflow framework 触发 @FRAMEWORK_REGISTRY.register("TriFlow")，
   base 的 auto-import 只扫 starVLA/model/framework/。
2) TriFlow 不吃 state：client（v1 eval client 原样复用）会发 state，这里直接删掉。
3) run_dir 打包资产刷新：dino_v3_stats.json（live DINO 归一化）与 vocab_map.json
   （紧凑词表）——跨机部署时 config 里的 playground 路径可能不存在。
4) fp32 server（v1 血泪教训：bf16 eval 数值口径不一致）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

######### // code // ##########
# 中文注释：vocab_map 重定向必须发生在 framework 构造（父类 __init__ → from_pretrained）
# 之前，因此放在 import 之后、类定义之前是不够的——在 __init__ 里 super() 之前设 env。
import starVLA.triflow.framework.tri_flow  # noqa: F401 - 注册 TriFlow framework
from starVLA.jointflow.data.mix_registry import register_jointflow_mixtures
from starVLA.jointflow.modules.dino_v3 import DINOv3Backbone
from deployment.model_server.policy_wrapper import PolicyServerWrapper

logger = logging.getLogger(__name__)


class TriFlowPolicyServerWrapper(PolicyServerWrapper):
    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
    ) -> None:
        register_jointflow_mixtures()
        # 中文注释：absolute() 不跟符号链接（resolve() 会把链接展开到别处，run_dir 就算错了）
        run_dir = Path(ckpt_path).absolute().parents[1]
        bundled_vocab = run_dir / "vocab_map.json"
        if bundled_vocab.exists() and not os.environ.get("TRIFLOW_VOCAB_MAP", "").strip():
            # 中文注释：优先用 run_dir 打包的词表（与训练逐字节一致），framework
            # __init__ 会读 TRIFLOW_VOCAB_MAP 覆盖 config 路径。
            os.environ["TRIFLOW_VOCAB_MAP"] = str(bundled_vocab)
            logger.info("TriFlow eval: using bundled vocab_map %s", bundled_vocab)
        if use_bf16:
            logger.warning("TriFlow eval: bf16 requested but TriFlow is fp32-only; forcing fp32 (v1 lesson).")
            use_bf16 = False
        super().__init__(ckpt_path=ckpt_path, device=device, use_bf16=use_bf16, unnorm_key=unnorm_key)
        self._run_dir = run_dir
        self._maybe_rebuild_live_dino()
        self._maybe_refresh_dino_stats()

    # ------------------------------------------------------------------
    # live DINO rebuild（env 覆盖 gated/离线权重场景；v1 同款，env 前缀 TRIFLOW_）
    # ------------------------------------------------------------------
    def _maybe_rebuild_live_dino(self) -> None:
        repo = os.environ.get("TRIFLOW_DINO_REPO_OR_DIR", "").strip()
        weights = os.environ.get("TRIFLOW_DINO_WEIGHTS", "").strip()
        loader = os.environ.get("TRIFLOW_DINO_LOADER", "").strip()
        if not (repo or weights or loader):
            return  # framework 在首次 predict 时 lazy 加载
        fw = self._framework
        dino_cfg = fw.config.framework.dino
        logger.info(
            "TriFlow eval: rebuilding live DINO (repo=%s weights=%s loader=%s)",
            repo or "<cfg>", weights or "<cfg>", loader or "<cfg>",
        )
        fw.dino = DINOv3Backbone(
            name=dino_cfg.get("name", "dinov3_vits16"),
            hf_model_id=dino_cfg.get("hf_model_id", "facebook/dinov3-vits16-pretrain-lvd1689m"),
            repo_or_dir=repo or dino_cfg.get("repo_or_dir", "facebookresearch/dinov3"),
            weights=(weights or dino_cfg.get("weights", None)) or None,
            loader=loader or dino_cfg.get("loader", "auto"),
            image_size=int(dino_cfg.get("image_size", 224)),
            patch_size=int(dino_cfg.get("patch_size", 16)),
            embed_dim=int(dino_cfg.get("embed_dim", 384)),
        ).to(fw.device)

    def _maybe_refresh_dino_stats(self) -> None:
        bundled = self._run_dir / "dino_v3_stats.json"
        if not bundled.exists():
            return
        try:
            self._framework._load_dino_stats(str(bundled))
            logger.info("TriFlow eval: refreshed DINO stats from %s", bundled)
        except Exception as exc:  # pragma: no cover
            logger.warning("TriFlow eval: failed to refresh DINO stats from %s: %s", bundled, exc)

    @staticmethod
    def _to_pil_images(imgs) -> List[Image.Image]:
        # 中文注释：client 经 msgpack 发 np.uint8；live DINO 需要 PIL。
        seq = imgs if isinstance(imgs, (list, tuple)) else [imgs]
        out = []
        for im in seq:
            if isinstance(im, Image.Image):
                out.append(im.convert("RGB"))
            else:
                out.append(Image.fromarray(np.asarray(im).astype(np.uint8)).convert("RGB"))
        return out

    def _prepare_examples(self, examples: List[dict]) -> List[dict]:
        # 中文注释：(1) image np→PIL；(2) 删除 state —— TriFlow 完全不吃 state，
        # v1 eval client 原样复用时它仍会发 state，这里丢弃即可。
        out = []
        for ex in examples:
            if not isinstance(ex, dict):
                out.append(ex)
                continue
            new = dict(ex)
            if new.get("image", None) is not None:
                new["image"] = self._to_pil_images(new["image"])
            new.pop("state", None)
            out.append(new)
        return out

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        examples = self._prepare_examples(examples)
        return super().predict_action(examples=examples, unnorm_key=unnorm_key, **kwargs)
######### // code // ##########
