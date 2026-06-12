"""CED PR-A 单测：C1 visual head 扩展数值等价 + C2 null action 往返校验。

运行（starVLA env）：
  python starVLA/jointflow/scripts/test_ced_units.py --config_yaml starVLA/jointflow/configs/jointflow_libero.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.jointflow.modules.visual_dino_flow_head import VisualFlowMatchingHead


######### // code // ##########
# 中文注释：C1 验收——
# 1) weights=None 与旧实现数值一致（同 seed 下 plain / return_pred / weights=ones 三者 loss 相等）；
# 2) weights 线性作用（2×ones → 2×loss）；
# 3) 外部 (noise, t=0) 时不消耗 RNG、确定可配对（两次调用 pred 逐位相等）；
# 4) per_patch 与 pred 自洽：per_patch == mean((pred−(z−ε))², dim=-1)。
def test_visual_head() -> None:
    cfg = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {"vl_hidden_dim": 24},
                "visual_model": {
                    "d_dino": 16,
                    "hidden_size": 32,
                    "num_attention_heads": 2,
                    "attention_head_dim": 16,
                    "num_layers": 1,
                    "dropout": 0.0,
                    "num_inference_timesteps": 2,
                },
            }
        }
    )
    head = VisualFlowMatchingHead(cfg)
    head.eval()  # 关 dropout，保证同 seed 可复现
    bsz, n, d = 3, 9, 16
    cond = torch.randn(bsz, 5, 24)
    z = torch.randn(bsz, n, d)

    torch.manual_seed(7)
    loss_plain = head(cond, z)
    torch.manual_seed(7)
    loss_rp, pred, per_patch = head(cond, z, return_pred=True)
    torch.manual_seed(7)
    loss_ones = head(cond, z, weights=torch.ones(bsz, n))
    torch.manual_seed(7)
    loss_double = head(cond, z, weights=2.0 * torch.ones(bsz, n))

    assert torch.equal(loss_plain, loss_rp), f"plain vs return_pred: {loss_plain} vs {loss_rp}"
    assert torch.allclose(loss_plain, loss_ones), f"plain vs weights=1: {loss_plain} vs {loss_ones}"
    assert torch.allclose(2.0 * loss_plain, loss_double), "weights 非线性作用"
    assert pred.shape == (bsz, n, d) and per_patch.shape == (bsz, n)

    eps = torch.randn(bsz, n, d)
    l1, v1, p1 = head(cond, z, noise=eps, t=0.0, return_pred=True)
    l2, v2, p2 = head(cond, z, noise=eps, t=0.0, return_pred=True)
    assert torch.equal(v1, v2), "外部 (noise,t=0) 两次调用应逐位相等（可配对相减）"
    recon = ((v1.float() - (z - eps).float()) ** 2).mean(dim=-1)
    assert torch.allclose(recon, p1, atol=1e-6), "per_patch 与 pred/velocity 不自洽"
    # t=0 时 velocity 目标 = z − ε，loss 应等于 per_patch.mean()
    assert torch.allclose(l1, p1.mean(), atol=1e-6)
    print("[test_ced_units] C1 visual head: PASS")
######### // code // ##########


######### // code // ##########
# 中文注释：C2 验收——真实 LIBERO mixture 上：
# 1) compute_null_action_table 维序与 _pack_sample 拼接一致；
# 2) make_null_action 输出反归一化后：min_max 维 ≈ raw 0；无归一化维 == batch 首步。
def test_null_action(config_yaml: str) -> None:
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform
    from starVLA.jointflow.data.joint_dataset import build_joint_dataset, compute_null_action_table
    from starVLA.jointflow.framework.qwen_joint_flow import QwenJointFlowVLA as QwenJointFlow

    cfg = OmegaConf.load(config_yaml)
    cfg.datasets.vla_data.num_workers = 0
    mixture = build_joint_dataset(cfg, mode="train")
    table = compute_null_action_table(mixture)
    assert table, "null-action table 为空"

    ds = mixture.datasets[0]
    sample = ds[0]
    action = torch.as_tensor(np.asarray(sample["action"]), dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    action[1, 0, -1] = -action[1, 0, -1] if float(action[1, 0, -1]) != 0 else 1.0  # 让两样本首步 gripper 不同
    examples = [{"dataset_name": sample["dataset_name"]}, {"dataset_name": sample["dataset_name"]}]

    class _Shim:
        pass

    shim = _Shim()
    shim._null_action_table = {}
    QwenJointFlow.set_null_action_table(shim, table)
    null = QwenJointFlow.make_null_action(shim, action, examples)
    assert null.shape == action.shape and torch.isfinite(null).all()

    # 按 modality_keys["action"] 維序拆回各 key，逐 key 校验
    normalizers = {}
    for tr in ds.transforms.transforms:
        if isinstance(tr, StateActionTransform):
            for key in tr.apply_to:
                if str(key).startswith("action.") and key in tr._normalizers:
                    normalizers[key] = tr._normalizers[key]
    offset = 0
    for key in ds.modality_keys["action"]:
        subkey = str(key).split(".", 1)[1]
        dim = int(np.asarray(ds.metadata.statistics.action[subkey].min).reshape(-1).shape[0])
        chunk = null[:, :, offset : offset + dim]
        if key in normalizers:
            raw = normalizers[key].inverse(chunk.reshape(-1, dim))
            assert raw.abs().max() < 1e-5, f"{key}: 反归一化后应为 raw 0，得到 max|raw|={raw.abs().max()}"
        else:
            expect = action[:, :1, offset : offset + dim].expand_as(chunk)
            assert torch.equal(chunk, expect), f"{key}: 无归一化维应复制首步"
            assert not torch.equal(chunk[0], chunk[1]), f"{key}: 两样本首步不同应体现在 a₀ 中"
        offset += dim
    assert offset == null.shape[-1]

    # 查不到数据集名必须 raise
    try:
        QwenJointFlow.make_null_action(shim, action, [{"dataset_name": "no_such_dataset"}] * 2)
        raise AssertionError("未知 dataset_name 应当 raise KeyError")
    except KeyError:
        pass
    print(f"[test_ced_units] C2 null action ({sorted(table)}): PASS")
######### // code // ##########


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/jointflow/configs/jointflow_libero.yaml")
    parser.add_argument("--skip_data", action="store_true", help="只跑无数据依赖的 C1 测试")
    args = parser.parse_args()

    test_visual_head()
    if not args.skip_data:
        test_null_action(args.config_yaml)
    print("[test_ced_units] ALL PASS")
