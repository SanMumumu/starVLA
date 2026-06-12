"""PR7: TriFlow 离线评测（特征空间，不起仿真）.

复用:
- starVLA.triflow.framework.tri_flow.TriFlowVLA.from_pretrained (run_dir 资产)
- starVLA.jointflow.data.joint_dataset.build_joint_dataloader (数据)

评测项（对应 MVP 评测矩阵）:
- v2l    : 整句匹配率 / token 准确率（含 EOS，pad 区不计）
- vf     : predict_future 与 GT 未来特征的 cosine / MSE（归一空间，仅 future_valid 样本）
           + 池化特征近邻检索 retrieval@1/@5（gallery = 全部评测样本的 GT 未来特征）
- policy : 归一空间动作 MSE/MAE（快速代理；真实成功率走仿真 server 链路）

输出: <run_dir>/eval_offline.json + stdout 摘要。
说明: RGB 拼图需要视频解码（JointLiberoDataset 训练路径不解视频），MVP 先输出
检索命中索引；要图请走 server + 仿真链路或后续 RAE 解码器（接口已留）。

用法:
  python starVLA/triflow/eval/eval_offline_triflow.py \
      --ckpt_path <run_dir>/checkpoints/steps_60000_pytorch_model_ema.pt --num_batches 50
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


######### // code // ##########
# 中文注释：评测主流程。逐 batch 跑三类 predict，聚合指标。
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=None, help="覆盖采样步数（默认用 config）")
    parser.add_argument("--out", type=str, default=None, help="默认 <run_dir>/eval_offline.json")
    args = parser.parse_args()

    # 中文注释：absolute() 不跟符号链接（resolve() 会把链接展开到别处，run_dir 就算错了）
    ckpt_path = Path(args.ckpt_path).absolute()
    run_dir = ckpt_path.parents[1]
    bundled_vocab = run_dir / "vocab_map.json"
    if bundled_vocab.exists() and not os.environ.get("TRIFLOW_VOCAB_MAP", "").strip():
        os.environ["TRIFLOW_VOCAB_MAP"] = str(bundled_vocab)

    from starVLA.triflow.framework.tri_flow import TriFlowVLA  # 注册 + from_pretrained

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriFlowVLA.from_pretrained(str(ckpt_path)).to(device)
    model.eval()
    bundled_stats = run_dir / "dino_v3_stats.json"
    if bundled_stats.exists():
        model._load_dino_stats(str(bundled_stats))

    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.datasets.vla_data.per_device_batch_size = int(args.batch_size)
    cfg.datasets.vla_data.num_workers = int(args.num_workers)
    from starVLA.jointflow.data.joint_dataset import build_joint_dataloader

    loader = build_joint_dataloader(cfg)

    has_v2l = "v2l" in model.task_registry
    has_passive = "passive" in model.task_registry
    has_policy = "policy" in model.task_registry

    em_hits, tok_hits, tok_total, n_text = 0, 0, 0, 0
    vf_cos, vf_mse, n_vf = [], [], 0
    act_mse, act_mae = [], []
    pooled_pred, pooled_gt = [], []

    n_batches = 0
    for batch in loader:
        if n_batches >= int(args.num_batches):
            break
        n_batches += 1
        langs = [str(ex["lang"]) for ex in batch]

        with torch.no_grad():
            # ---- v2l ----
            if has_v2l:
                out = model.predict_language(batch, num_steps=args.num_steps)
                gt_ids, gt_lens = model.codec.encode_batch(langs)
                pred_ids = torch.as_tensor(out["token_ids"])
                for i in range(len(batch)):
                    n_text += 1
                    if out["texts"][i].strip() == langs[i].strip():
                        em_hits += 1
                    L = int(gt_lens[i])
                    tok_hits += int((pred_ids[i, :L] == gt_ids[i, :L]).sum())
                    tok_total += L

            # ---- vf (passive：纯 V→V′) ----
            if has_passive:
                valid = [bool(ex.get("future_valid", True)) for ex in batch]
                out = model.predict_future(batch, task="passive", num_steps=args.num_steps)
                pred_norm = torch.as_tensor(out["normalized_features"])  # [B,P,D]
                spec = model.task_registry["passive"]
                b = model._examples_to_batch(batch, spec)
                gt_norm = model._select_future_dino(b["dino_1"], batch).float().cpu()
                for i in range(len(batch)):
                    if not valid[i]:
                        continue
                    n_vf += 1
                    p, g = pred_norm[i], gt_norm[i]
                    cos = torch.nn.functional.cosine_similarity(p.reshape(1, -1), g.reshape(1, -1)).item()
                    vf_cos.append(cos)
                    vf_mse.append(float((p - g).pow(2).mean()))
                    pooled_pred.append(p.mean(dim=0))
                    pooled_gt.append(g.mean(dim=0))

            # ---- policy（归一空间快速代理） ----
            if has_policy:
                out = model.predict_action(batch, num_steps=args.num_steps)
                pred = torch.as_tensor(out["normalized_actions"])
                gt = torch.stack([torch.as_tensor(np.asarray(ex["action"], dtype=np.float32)) for ex in batch])
                gt = gt[:, -model.action_horizon :, : model.action_dim]
                act_mse.append(float((pred - gt).pow(2).mean()))
                act_mae.append(float((pred - gt).abs().mean()))

    # ---- 近邻检索：query=预测池化特征，gallery=全部 GT 池化特征 ----
    retrieval = {}
    if pooled_pred:
        P = torch.nn.functional.normalize(torch.stack(pooled_pred), dim=-1)
        G = torch.nn.functional.normalize(torch.stack(pooled_gt), dim=-1)
        sim = P @ G.t()  # [N,N]
        ranks = sim.argsort(dim=-1, descending=True)
        truth = torch.arange(sim.shape[0]).reshape(-1, 1)
        retrieval = {
            "n": int(sim.shape[0]),
            "top1": float((ranks[:, :1] == truth).any(dim=-1).float().mean()),
            "top5": float((ranks[:, :5] == truth).any(dim=-1).float().mean()),
        }

    report = {
        "ckpt": str(ckpt_path),
        "num_batches": n_batches,
        "v2l": {
            "n": n_text,
            "exact_match": (em_hits / n_text) if n_text else None,
            "token_acc": (tok_hits / tok_total) if tok_total else None,
        },
        "vf": {
            "n": n_vf,
            "cosine_mean": float(np.mean(vf_cos)) if vf_cos else None,
            "mse_mean": float(np.mean(vf_mse)) if vf_mse else None,
            "retrieval": retrieval,
        },
        "policy_proxy": {
            "norm_action_mse": float(np.mean(act_mse)) if act_mse else None,
            "norm_action_mae": float(np.mean(act_mae)) if act_mae else None,
        },
    }
    out_path = Path(args.out) if args.out else run_dir / "eval_offline.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[eval_offline] report -> {out_path}")
######### // code // ##########


if __name__ == "__main__":
    main()
