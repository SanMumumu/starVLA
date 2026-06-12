"""PR3-PR6: 单 batch 过拟合验收脚本.

复用:
- starVLA.triflow.framework.tri_flow.TriFlowVLA.from_pretrained
- starVLA.jointflow.data.joint_dataset.build_joint_dataloader

逻辑: 加载 <run_root>/<run_id>/checkpoints/steps_<N>_pytorch_model.pt，
按训练同款确定性 dataloader（num_workers=0/seed/首 batch）重建 overfit batch，
跑对应 predict_*，对照验收门:
- policy/idm : max|pred-gt| < 0.05（归一空间动作）
- fdm/passive: 整图特征 cosine > 0.95（归一空间）
- v2l        : 整句匹配 100% + 查表单位 RMS 断言
- joint      : 动作 err < 0.05 且 vf cosine > 0.95

用法: python starVLA/triflow/scripts/check_overfit.py --task fdm [--run_id triflow_overfit_fdm] [--steps 2000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


######### // code // ##########
# 中文注释：按任务分发验收逻辑；全部在归一空间对照。
def _action_err(model, batch, task: str) -> float:
    gt = np.stack([np.asarray(ex["action"], dtype=np.float32) for ex in batch])[:, -model.action_horizon :, : model.action_dim]
    if task == "policy":
        pred = model.predict_action(batch)["normalized_actions"]
    else:
        out = model._generate(task, batch)
        pred = out["z"]["a"].detach().float().cpu().numpy()
    return float(np.abs(pred - gt).max())


def _vf_cosine(model, batch, task: str) -> float:
    out = model.predict_future(batch, task=task)
    pred = torch.as_tensor(out["normalized_features"])
    spec = model.task_registry[task]
    b = model._examples_to_batch(batch, spec)
    gt = model._select_future_dino(b["dino_1"], batch).float().cpu()
    cos = torch.nn.functional.cosine_similarity(pred.reshape(pred.shape[0], -1), gt.reshape(gt.shape[0], -1))
    return float(cos.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--run_root", type=str, default="playground/Debug_Checkpoints")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    run_id = args.run_id or f"triflow_overfit_{args.task}"
    run_dir = Path(args.run_root) / run_id
    ckpt = run_dir / "checkpoints" / f"steps_{args.steps}_pytorch_model.pt"

    from starVLA.triflow.framework.tri_flow import TriFlowVLA
    from starVLA.jointflow.data.joint_dataset import build_joint_dataloader

    model = TriFlowVLA.from_pretrained(str(ckpt))
    model = model.cuda().eval() if torch.cuda.is_available() else model.eval()

    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.datasets.vla_data.num_workers = 0
    batch = next(iter(build_joint_dataloader(cfg)))

    task = args.task
    ok = True
    with torch.no_grad():
        if task in {"policy", "idm"}:
            err = _action_err(model, batch, task)
            ok = err < 0.05
            print(f"[check_overfit] {task}: max|action err| = {err:.5f} (gate <0.05)")
        elif task in {"fdm", "passive"}:
            cos = _vf_cosine(model, batch, task)
            ok = cos > 0.95
            print(f"[check_overfit] {task}: vf cosine = {cos:.4f} (gate >0.95)")
        elif task == "v2l":
            x0 = model.embedder.embed_text_x0(torch.zeros(1, 4, dtype=torch.long, device=model.device))
            rms = float(x0.pow(2).mean(dim=-1).sqrt().mean())
            assert abs(rms - 1.0) < 1e-3, f"embed lookup RMS != 1: {rms}"
            out = model.predict_language(batch)
            langs = [str(ex["lang"]).strip() for ex in batch]
            hits = sum(int(p.strip() == g) for p, g in zip(out["texts"], langs))
            ok = hits == len(batch)
            print(f"[check_overfit] v2l: exact match {hits}/{len(batch)} (gate 100%); lookup RMS={rms:.4f}")
            for p, g in zip(out["texts"][:3], langs[:3]):
                print(f"    pred={p!r}\n    gt  ={g!r}")
        elif task == "joint":
            out = model.predict_joint(batch)
            gt = np.stack([np.asarray(ex["action"], dtype=np.float32) for ex in batch])[:, -model.action_horizon :, : model.action_dim]
            err = float(np.abs(out["normalized_actions"] - gt).max())
            pred = torch.as_tensor(out["normalized_features"])
            spec = model.task_registry["joint"]
            b = model._examples_to_batch(batch, spec)
            gt_vf = model._select_future_dino(b["dino_1"], batch).float().cpu()
            cos = float(torch.nn.functional.cosine_similarity(pred.reshape(pred.shape[0], -1), gt_vf.reshape(gt_vf.shape[0], -1)).mean())
            ok = err < 0.05 and cos > 0.95
            print(f"[check_overfit] joint: action err={err:.5f} (<0.05), vf cosine={cos:.4f} (>0.95)")
        else:
            raise ValueError(f"unknown task {task}")
    print(f"[check_overfit] {task}: {'PASSED' if ok else 'FAILED'}")
    raise SystemExit(0 if ok else 1)
######### // code // ##########


if __name__ == "__main__":
    main()
