"""PR2: TriFlow smoke 测试（验收门）.

复用:
- starVLA.jointflow.modules.attention_mask.visualize_mask (mask 出图)

验收项:
1. 6 任务 mask PNG（干净行对噪声列全暗 / 噪声行全亮 / 文本条件 pad 列暗 / joint 两噪声块互见）
2. 全 (task, branch) 假数据 forward 有限
3. 梯度覆盖断言：每次 backward 后所有可训练参数都有 grad（DDP full coverage），
   且 锚定组件梯度全零 / 在用组件梯度非零 —— 锚点表 == used_components 推导，机器验证
4. future_valid 全 False 时 loss 掩零但覆盖不变（无 early-return 分叉）
5. mask 泄漏单测：扰动噪声块输入，干净块 hidden 必须逐位不变
6. 参数量打印（预期 85–95M）+ predict_* 形状检查

用法:
  python starVLA/triflow/scripts/smoke_test.py \
      --config_yaml starVLA/triflow/configs/triflow_libero.yaml \
      --out playground/Debug_Checkpoints/triflow_smoke
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.jointflow.modules.attention_mask import visualize_mask
from starVLA.triflow.data.build_vocab import build_vocab_map
from starVLA.triflow.framework.tri_flow import TriFlowVLA
from starVLA.triflow.modules.attention_mask import build_clean_noisy_mask


FAKE_LANGS = [
    "pick up the ketchup and place it in the basket",
    "pick up the orange juice and place it in the basket",
]


######### // code // ##########
# 中文注释：构造假 batch（与 JointLiberoDataset 输出字段一致；无 state）。
def make_fake_batch(batch_size: int, views: int, n_patches: int, d_dino: int, horizon: int, action_dim: int, future_valid: bool = True):
    rng = np.random.default_rng(0)
    batch = []
    for idx in range(batch_size):
        batch.append(
            {
                "dino_0": rng.standard_normal((views, n_patches, d_dino)).astype("float32"),
                "dino_1": rng.standard_normal((views, n_patches, d_dino)).astype("float32"),
                "action": rng.uniform(-1, 1, size=(horizon, action_dim)).astype("float32"),
                "lang": FAKE_LANGS[idx % len(FAKE_LANGS)],
                "dino_view_keys": ["video.primary_image", "video.wrist_image"][:views],
                "future_valid": bool(future_valid),
                "future_valid_steps": np.int64(8 if future_valid else 0),
                "future_stride": np.int64(8),
            }
        )
    return batch


def ensure_vocab(cfg) -> None:
    """vocab_map 缺失时（如新环境），用假指令现场生成临时词表，让 smoke 自洽。"""
    path = Path(cfg.framework.text.vocab_map_path)
    if path.exists():
        return
    vocab_map = build_vocab_map(cfg.framework.text.tokenizer_path, FAKE_LANGS)
    tmp = Path(tempfile.mkdtemp()) / "vocab_map_smoke.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(vocab_map, f)
    cfg.framework.text.vocab_map_path = str(tmp)
    print(f"[smoke] vocab_map missing; built temp vocab at {tmp}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/triflow/configs/triflow_libero.yaml")
    parser.add_argument("--out", type=str, default="playground/Debug_Checkpoints/triflow_smoke")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.config_yaml)
    ensure_vocab(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = TriFlowVLA(cfg).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] trainable params = {n_params/1e6:.1f}M")
    # 中文注释：参数量随 tower 配置变化（base≈88M / large(ELF-M)≈330M / ELF-L≈650M），
    # 只做下限与离谱上限的 sanity 检查。
    assert 50e6 < n_params < 1.2e9, f"param count {n_params} out of sane range"

    # 中文注释：out_proj_* 零初始化会让 step-0 的上游梯度恰为 0（grad 存在但全零，
    # DDP 覆盖无碍；训练一步后即非零）。仅为让"在用组件梯度非零"断言可观察，
    # 测试里微扰输出投影——不影响真实训练代码。
    with torch.no_grad():
        for lin in (model.embedder.out_proj_v, model.embedder.out_proj_l, model.embedder.out_proj_a):
            lin.weight.add_(torch.randn_like(lin.weight) * 0.01)

    horizon, action_dim = model.action_horizon, model.action_dim
    batch = make_fake_batch(2, 2, model.n_patches, model.d_dino, horizon, action_dim)

    # ---- 1. mask PNG ----
    for name, spec in model.task_registry.items():
        sizes = {"v": 2 * model.n_patches, "l": int(model.codec.max_len), "a": horizon, "vf": model.n_patches}
        block_sizes = [sizes[k] for k in spec.blocks]
        lens = torch.tensor([5, 7]) if ("l" in spec.cond) else None
        text_idx = spec.blocks.index("l") if ("l" in spec.cond) else None
        mask = build_clean_noisy_mask(
            block_sizes, list(spec.noisy_flags), batch_size=2, dtype=torch.float32,
            device=torch.device("cpu"), text_block_idx=text_idx, text_valid_lens=lens,
        )
        visualize_mask(mask, [f"{k}{'*' if f else ''}" for k, f in zip(spec.blocks, spec.noisy_flags)], out_dir / f"mask_{name}.png")
        # 规则自检：噪声行全亮（除文本 pad 列）；干净行在噪声列全暗
        m0 = mask[0, 0]
        starts = np.cumsum([0] + block_sizes)
        for qi, q_noisy in enumerate(spec.noisy_flags):
            for ki, k_noisy in enumerate(spec.noisy_flags):
                sub = m0[starts[qi]:starts[qi + 1], starts[ki]:starts[ki + 1]]
                if k_noisy and not q_noisy:
                    assert (sub != 0).all(), f"{name}: clean q sees noisy k!"
                elif not (text_idx is not None and ki == text_idx):
                    assert (sub == 0).all(), f"{name}: visible pair masked unexpectedly"
    print(f"[smoke] mask PNGs + rules OK -> {out_dir}/mask_*.png")

    # ---- 2+3. forward / 梯度覆盖断言 ----
    branches = []
    for name, spec in model.task_registry.items():
        branches.append((name, False))
        if "l" in spec.target:
            branches.append((name, True))
    comp_map = model._component_map()
    for task, dec in branches:
        model.zero_grad(set_to_none=True)
        out = model(batch, task=task, decoder_step=dec)
        loss = out[f"{task}_loss"]
        assert torch.isfinite(loss), f"{task} dec={dec}: non-finite loss"
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{task} dec={dec}: params missing grad (DDP would hang): {missing[:8]}"
        used = model.used_components(model.task_registry[task], dec)
        for comp_name, comp in comp_map.items():
            params = model._params_of(comp)
            grad_norm = sum(float(p.grad.abs().sum()) for p in params if p.grad is not None)
            if comp_name in used:
                assert grad_norm > 0, f"{task} dec={dec}: used component `{comp_name}` got zero grad"
            else:
                assert grad_norm == 0, f"{task} dec={dec}: anchored component `{comp_name}` got NONZERO grad {grad_norm}"
        print(f"[smoke] {task:8s} dec={int(dec)} loss={float(loss):.4f} coverage OK ({len(used)} used comps)")

    # ---- 4. future_valid 全 False：loss 掩零、覆盖不变 ----
    invalid = make_fake_batch(2, 2, model.n_patches, model.d_dino, horizon, action_dim, future_valid=False)
    for task in [t for t, s in model.task_registry.items() if "vf" in s.target or "vf" in s.cond]:
        model.zero_grad(set_to_none=True)
        out = model(invalid, task=task)
        out[f"{task}_loss"].backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{task} (all-invalid future): coverage broken"
    print("[smoke] all-invalid-future coverage OK")

    # ---- 5. mask 泄漏单测：扰动噪声块输入，干净块 hidden 必须逐位不变 ----
    model.eval()
    with torch.no_grad():
        spec = model.task_registry["joint"]
        b = model._examples_to_batch(batch, spec)
        clean = model._clean_latents(b, spec)
        t_map = {k: torch.ones(clean["bsz"], device=device) for k in spec.blocks}
        for k in spec.target:
            t_map[k] = torch.full((clean["bsz"],), 0.37, device=device)
        lat1 = {k: clean["x0"][k].clone() for k in spec.blocks}
        lat2 = {k: v.clone() for k, v in lat1.items()}
        for k in spec.target:
            lat2[k] = lat2[k] + torch.randn_like(lat2[k]) * 5.0
        tok1, m1, sl1 = model._assemble(spec, lat1, t_map, clean)
        tok2, m2, sl2 = model._assemble(spec, lat2, t_map, clean)
        h1, h2 = model.tower(tok1, m1), model.tower(tok2, m2)
        for k in spec.cond:
            same = torch.equal(h1[:, sl1[k]], h2[:, sl2[k]])
            assert same, f"leakage: clean block `{k}` hidden changed when noisy blocks perturbed!"
    print("[smoke] no-leakage check OK (clean hidden bit-identical)")

    # ---- 6. predict_* 形状 ----
    with torch.no_grad():
        acts = model.predict_action(batch, num_steps=4)["normalized_actions"]
        assert acts.shape == (2, horizon, action_dim), acts.shape
        lang_out = model.predict_language(batch, num_steps=4)
        assert len(lang_out["texts"]) == 2
        fut = model.predict_future(batch, task="passive", num_steps=4)["future_features"]
        assert fut.shape == (2, model.n_patches, model.d_dino), fut.shape
        joint = model.predict_joint(batch, num_steps=4)
        assert joint["normalized_actions"].shape == (2, horizon, action_dim)
    print(f"[smoke] predict_action {acts.shape} / predict_language {lang_out['texts'][0]!r} / "
          f"predict_future {fut.shape} / predict_joint OK")
    print("[smoke] ALL PASSED")
######### // code // ##########


if __name__ == "__main__":
    main()
