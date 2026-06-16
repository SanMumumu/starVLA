#######
# 中文注释：JointFlow→QwenGR00T 迁移的 smoke 自检（不训练，只验证迁移后代码路径正确）。
# 用真实 Qwen3-VL backbone + 假 DINO/action 批：跑 policy/fdm/idm/passive/effect 五任务 forward 与
# predict_action，校验 shape（action_horizon/action_dim/d_dino/hidden/token 数）与有限性；
# 并断言 CED 关键性质：a_null≠0 且≠真实动作、Δ 非零、effect 五个子 loss 有限。
# 用法：python -m starVLA.model.framework.VLM4A.smoke_jointflow --config_yaml examples/LIBERO/train_files/starvla_jointflow_ced_libero.yaml
#######
from __future__ import annotations

import argparse

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T


def _make_fake_examples(bsz, views, n_tokens, d_dino, horizon, action_dim, ds_name="fake_ds"):
    ex = []
    for i in range(bsz):
        ex.append(
            {
                "dino_0": np.random.randn(views, n_tokens, d_dino).astype("float32"),
                "dino_1": np.random.randn(views, n_tokens, d_dino).astype("float32"),
                "action": np.random.uniform(-1, 1, size=(horizon, action_dim)).astype("float32"),
                "future_valid": np.array([1.0], dtype="float32"),
                "lang": f"fake libero instruction {i}",
                "dataset_name": ds_name,
                "trajectory_id": i,
                "dino_view_keys": ["video.primary_image", "video.wrist_image"][:views],
            }
        )
    return ex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config_yaml", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bsz", type=int, default=2)
    p.add_argument("--views", type=int, default=2)
    #######
    # 中文注释：可选覆盖 CED teacher / action query 数，用来自检不同实验臂。
    p.add_argument("--teacher", default=None, help="delta|pooled_future|delta_shuffled|future_dino|delta_dino")
    p.add_argument("--n_action_query", default=None, help="action query token 数（E3.x 旋钮）")
    p.add_argument("--flow_matching_steps", default=None, help="E1.3 multi-step FM 步数")
    p.add_argument("--use_correlated_noise", action="store_true", help="E1.3 correlated noise（注入假 Cholesky 自检）")
    #######
    args = p.parse_args()
    device = torch.device(args.device)

    cfg = OmegaConf.load(args.config_yaml)
    #######
    if args.teacher is not None:
        cfg.framework.losses.teacher = args.teacher
    if args.n_action_query is not None:
        cfg.framework.action_model.n_action_query = int(args.n_action_query)
    if args.flow_matching_steps is not None:
        cfg.framework.action_model.flow_matching_steps = int(args.flow_matching_steps)
    if args.use_correlated_noise:
        cfg.framework.action_model.use_correlated_noise = True
    #######
    print("[smoke] building Qwen_GR00T (jointflow) ...", flush=True)
    model = Qwen_GR00T(cfg).to(device).eval()
    H = model.action_horizon
    D = model.action_dim
    d_dino = model.d_dino
    n_tok = int(cfg.framework.visual_model.get("n_query", 196))
    print(f"[smoke] action_horizon={H} action_dim={D} d_dino={d_dino} n_tok={n_tok} "
          f"ced={model.ced_enabled} action_state_dim={cfg.framework.action_model.state_dim} "
          f"n_action_query={cfg.framework.action_model.get('n_action_query', H)}")

    # CED null-action 表：平动 6 维非零、gripper NaN（复制首步）。验证 a_null 不是简单全 0。
    null_vec = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1] + [np.nan] * (D - 6), dtype="float32")
    model.set_null_action_table({"fake_ds": null_vec})

    #######
    # 中文注释：E1.3 自检——注入一个合法 Cholesky（随机 PSD 协方差的下三角），让 correlated noise 路径真正执行。
    if cfg.framework.action_model.get("use_correlated_noise", False):
        flat = H * D
        A = np.random.randn(flat, flat).astype("float32")
        chol = np.linalg.cholesky(A @ A.T / flat + np.eye(flat))
        model.set_action_correlation(chol)
        print(f"[smoke] E1.3 correlated noise 已注入 Cholesky[{flat},{flat}]；flow_matching_steps="
              f"{cfg.framework.action_model.get('flow_matching_steps', 1)}")
    #######

    ex = _make_fake_examples(args.bsz, args.views, n_tok, d_dino, H, D)
    real_action = torch.from_numpy(np.stack([e["action"] for e in ex])).to(device)
    null_action = model.make_null_action(real_action, ex)
    assert null_action.shape == real_action.shape, "null action shape mismatch"
    assert not torch.allclose(null_action, torch.zeros_like(null_action)), "a_null 不应全 0"
    assert not torch.allclose(null_action, real_action), "a_null 不应等于真实动作"
    print(f"[smoke] ✓ make_null_action: 非全0、≠真实动作 (示例平动均值={null_action[...,:6].mean().item():.3f})")

    ok = True
    with torch.no_grad():
        for task in ["policy", "fdm", "idm", "passive", "effect"]:
            try:
                out = model.forward(ex, task=task)
                loss = out[f"{task}_loss"]
                finite = bool(torch.isfinite(loss).all())
                extra = ""
                if task == "effect":
                    keys = [k for k in out if k.startswith("loss/") or k.startswith("stat/")]
                    subfin = all(bool(torch.isfinite(out[k]).all()) for k in keys)
                    dnorm = float(out.get("stat/delta_norm_mean", torch.tensor(0.0)))
                    extra = f" | 子项 {keys} 有限={subfin} delta_norm={dnorm:.4f}"
                    finite = finite and subfin
                    assert dnorm > 0, "Δ 池化范数应 > 0（pos/null 两路差分非零）"
                print(f"[smoke] ✓ task={task:8s} loss={float(loss):.4f} finite={finite}{extra}")
                ok = ok and finite
            except Exception as exc:
                ok = False
                import traceback
                print(f"[smoke] ✗ task={task:8s} FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()

        # predict_action：只走 policy，输出 [B,H,D]
        try:
            pred = model.predict_action(ex)["normalized_actions"]
            shape_ok = tuple(pred.shape) == (args.bsz, H, D)
            print(f"[smoke] ✓ predict_action shape={tuple(pred.shape)} 期望={(args.bsz, H, D)} ok={shape_ok}")
            ok = ok and shape_ok
        except Exception as exc:
            ok = False
            print(f"[smoke] ✗ predict_action FAILED: {type(exc).__name__}: {exc}")

    print(f"\n[smoke] {'全部通过 ✅' if ok else '存在失败 ❌'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
