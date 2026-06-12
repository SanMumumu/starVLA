"""CED PR-B smoke：effect 任务假数据前向/反向 + 梯度全覆盖断言（DDP readiness 的单进程等价物）。

检查项：
1) 5 个 task（policy/fdm/idm/passive/effect）forward 全有限，effect 输出含全部子指标；
2) effect backward 后**所有可训练参数都有 grad**（anchor 体系 + 三前向共同保证，
   find_unused_parameters=False 的 DDP 不挂死的充分条件）；
3) 其余 4 个 task backward 同样全覆盖（anchor 表回归）；
4) effect 三个边界路径全覆盖：future 全无效早退 / Δ 段被 effect_delta_prob 跳过 / 三种 teacher。

运行（starVLA env）：
  python starVLA/jointflow/scripts/test_ced_effect_smoke.py
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.jointflow.framework.qwen_joint_flow import QwenJointFlowVLA, _make_fake_batch


######### // code // ##########
# 中文注释：构 CED smoke 用的假 batch——在 _make_fake_batch 基础上补 effect 任务必需字段。
def _make_ced_batch(model_cfg, batch_size=2, future_valid=True):
    batch = _make_fake_batch(
        batch_size=batch_size,
        views=2,
        n_tokens=int(model_cfg.framework.visual_model.n_query),
        d_dino=int(model_cfg.framework.visual_model.d_dino),
        action_horizon=int(model_cfg.framework.action_model.action_horizon),
        action_dim=int(model_cfg.framework.action_model.action_dim),
    )
    for idx, ex in enumerate(batch):
        ex["dataset_name"] = "fake_ds"
        ex["trajectory_id"] = np.int64(idx * 10)  # idx0 → traj 0 → probe 验证集；idx1 → 训练集
        ex["future_valid"] = bool(future_valid)
    return batch


def _assert_full_grad_coverage(model, tag: str):
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"[{tag}] {len(missing)} params missing grad, e.g. {missing[:6]}"
    model.zero_grad(set_to_none=True)


def main():
    cfg = OmegaConf.load("starVLA/jointflow/configs/jointflow_libero_ced.yaml")
    cfg.framework.qwenvl.base_vlm = "playground/Pretrained_models/Qwen2.5-0.5B"
    cfg.framework.dino.load_live_backbone = False
    cfg.framework.action_model.diffusion_model_cfg.num_layers = 1
    cfg.framework.visual_model.num_layers = 1

    model = QwenJointFlowVLA(cfg)
    assert model.probe_mlp is not None and model.proj_A is not None, "ced.enabled=true 应构造 probe/proj"
    n_probe = sum(p.numel() for p in model.probe_mlp.parameters()) + sum(p.numel() for p in model.proj_A.parameters())
    assert n_probe < 1_000_000, f"probe+proj 参数量应 <1M，得到 {n_probe}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()

    horizon = int(cfg.framework.action_model.action_horizon)
    dim = int(cfg.framework.action_model.action_dim)
    null_vec = np.full(dim, 0.1, dtype=np.float32)
    null_vec[-1] = np.nan  # gripper 维：复制首步
    model.set_null_action_table({"fake_ds": null_vec})

    batch = _make_ced_batch(cfg)

    # ---- 1)+3) 全任务 forward/backward + 梯度覆盖 ----
    for task in ["policy", "fdm", "idm", "passive", "effect"]:
        out = model(batch, task=task)
        loss = out[f"{task}_loss"]
        assert torch.isfinite(loss), f"{task} loss non-finite"
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"{task} 子指标 {k} non-finite"
        loss.backward()
        _assert_full_grad_coverage(model, task)
        print(f"[smoke] task={task} loss={float(loss):.4f} keys={sorted(out)}")

    # effect 子指标完整性（含 probe 验证集指标：traj 0 % 10 == 0 在验证集）
    out = model(batch, task="effect")
    expect = {"loss/act", "loss/fdm", "loss/fdm_static", "loss/fdm_dynamic", "loss/ident", "loss/align",
              "metric/probe_mse_val", "stat/delta_norm_mean", "stat/gate_on_frac", "stat/w_clamp_frac"}
    missing_keys = expect - set(out)
    assert not missing_keys, f"effect 缺子指标: {missing_keys}"
    out["effect_loss"].backward()
    _assert_full_grad_coverage(model, "effect-metrics")

    # ---- 4a) future 全无效早退路径 ----
    out = model(_make_ced_batch(cfg, future_valid=False), task="effect")
    out["effect_loss"].backward()
    _assert_full_grad_coverage(model, "effect-no-future")
    print("[smoke] effect future-invalid early-return: OK")

    # ---- 4b) Δ 段子采样跳过路径 ----
    cfg_obj = model.config.framework.losses
    old_prob = cfg_obj.effect_delta_prob
    cfg_obj.effect_delta_prob = 0.0
    out = model(batch, task="effect")
    assert "loss/ident" not in out, "Δ 段跳过时不应有 ident 指标"
    out["effect_loss"].backward()
    _assert_full_grad_coverage(model, "effect-delta-skip")
    cfg_obj.effect_delta_prob = old_prob
    print("[smoke] effect delta-subsample skip: OK")

    # ---- 4c) 三种 teacher ----
    for teacher in ["delta", "pooled_future", "delta_shuffled"]:
        model.config.framework.losses.teacher = teacher
        model.config.framework.losses.lam_a = 0.1
        out = model(batch, task="effect")
        out["effect_loss"].backward()
        _assert_full_grad_coverage(model, f"teacher={teacher}")
        print(f"[smoke] teacher={teacher} loss={float(out['effect_loss']):.4f}")

    # ---- a₀ 数值检查：NaN 维复制首步、数值维等于常数 ----
    action = torch.as_tensor(np.stack([ex["action"] for ex in batch]), dtype=torch.float32)
    null = model.make_null_action(action, batch)
    assert torch.allclose(null[:, :, :-1], torch.full_like(null[:, :, :-1], 0.1))
    assert torch.equal(null[:, :, -1], action[:, :1, -1].expand(-1, horizon))
    print("[smoke] make_null_action wiring: OK")

    print("[test_ced_effect_smoke] ALL PASS")


if __name__ == "__main__":
    main()
