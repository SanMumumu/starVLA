#!/usr/bin/env python
"""CED 反事实效果可视化 + 定量判读（qualitative + quantitative）。

把 CED 训练时的核心信号 Δ = v̂(fdm⁺) − v̂(fdm⁰) 直接「看出来」：
  - fdm⁺：真实动作条件下的未来预测；fdm⁰：空动作 a₀（原地不动）条件下的未来预测。
  - 在 t=0、共享噪声 ε 下，noisy=ε、velocity v=z−ε ⇒ `ε+v = E[z_H|cond]`，
    所以 ε+v_pos / ε+v_nul 就是模型「想象的未来」（真实动作 / 空动作），二者之差 Δ 把 ε 消掉，
    是动作对未来的纯 interventional 效果。用 RAEv2 stage-1 解码器把这些 DINO 特征解回 RGB。

每个样本输出一条横排面板（panel_*.png）：
  o_t | o_{t+H}真实 | RAE(z_t) | RAE(z_H真实) | RAE(想象·真实动作) | RAE(想象·空动作) |
  |真实动作−空动作|像素差 | Δ 叠加 o_t | 真实变化 叠加 o_t | Δ 热力图 | 真实变化 热力图

定量（summary.json / samples.jsonl）：动作敏感度、Δ 与真实变化的定位重合度、probe 从 Δ 回归动作的精度。
最后按「方法有意义的判定准则」（见 CRITERION.md / 末尾打印）给出 PASS/FAIL 判读。

复用 tools/debug_action_causality.py 的全部基础设施（RAE 解码器 / RGB 读取 / 模型加载 / 配置）。

用法（一条命令，离线 DINO latent，自动从 run 目录建配置 + 找 ckpt）：
  /mnt/nodestor/ws/Anaconda3/envs/starVLA/bin/python -m starVLA.jointflow.tools.viz_ced_effect \
    --run_dir /mnt/hwdata/wangsen/starVLA/starVLA/Z_ws/CED/jf_e107_ced_E3_1_effect_teacher \
    --num_samples 24 --max_visual_samples 12 --device cuda:0
输出落在 <run_dir>/ced_viz/。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# 复用参考工具的全部基础设施（import 即触发其 _ensure_starvla_importable，设置 sys.path）。
from starVLA.jointflow.tools.debug_action_causality import (  # noqa: E402
    RAEDinoDecoder,
    load_model,
    collect_examples,
    predict_visual_from_noise,
    _read_rgb_views,
    _select_view_index_from_example,
    _jointflow_norm_to_raw_dino,
    _tensor_to_pil_rgb,
    _heatmap_to_rgb,
    _metadata,
    _sanitize_name,
    _resolve_path,
    _infer_dino_token_count,
    _infer_rae_image_size,
    _load_config,
)
from starVLA.jointflow.data.joint_dataset import build_joint_dataset, compute_null_action_table  # noqa: E402
from starVLA.jointflow.data.mix_registry import register_jointflow_mixtures  # noqa: E402
import starVLA.jointflow.framework.qwen_joint_flow  # noqa: F401,E402  注册 framework


REPO = Path("/mnt/hwdata/wangsen/starVLA/starVLA")
DEFAULT_DATA = "/mnt/hwdata/wangsen/starVLA/DATA/LEBERO/libero"
DEFAULT_QWEN = "/mnt/hwdata/wangsen/starVLA/starVLA/playground/Pretrained_models/Qwen2.5-0.5B"
DEFAULT_RAE_REPO = "/mnt/hwdata/wangsen/starVLA/starVLA/Third_github/RAEv2"
DEFAULT_RAE_DECODER = "/mnt/hwdata/wangsen/starVLA/starVLA/Z_ws/debug/stage1_imagenet_dinov3s-k1"
DEFAULT_RAE_CONFIG = "/mnt/hwdata/wangsen/starVLA/starVLA/Third_github/RAEv2/configs/decoder/ViTXL"

# 建议判定阈值（启发式，可在结果里据情调整；阈值含义见 CRITERION.md）。
THRESH = {"sensitivity": 0.20, "loc_corr": 0.30, "loc_iou": 0.20, "probe_r2": 0.50}

CRITERION_MD = """# CED 方法「是否有意义」的判定准则

LIBERO 上 SR 在 goal/object/spatial 已 ~90%+ 饱和，long 受长程难度+评测方差主导，
SR 这个指标对「Δ 是否编码了正确的因果效应」分辨率很低。本工具绕开 SR，直接在
DINO 特征 / 像素空间检验 CED 的机制信号。三条**同时成立**才说明方法机制有效：

- **(S) 动作敏感度** `rel_effect = ‖想象未来(真实动作) − 想象未来(空动作)‖ / ‖想象未来(真实动作) − z_t‖`
  中位数 ≥ {sensitivity}。否则 FDM 基本无视动作（fdm⁺≈fdm⁰），Δ≈0，CED 无可蒸馏的东西。
- **(L) 因果定位** Δ 热力图与真实变化热力图的 Pearson 相关中位数 ≥ {loc_corr} 且 top-10% patch
  IoU 中位数 ≥ {loc_iou}。即「模型预测动作会改变的区域」落在「真实发生变化的区域」（被操作物体/夹爪/接触点）。
  否则 Δ 是弥散噪声，不是因果归因。
- **(I) 可辨识性** probe 从 Δ 回归动作的整体 R² ≥ {probe_r2}（复现训练期 gate）。否则 Δ 不含动作信息。

判读：
- **S∧L∧I 都过、但 SR 平**：机制有效，是 LIBERO 太简单/饱和导致 SR 测不出 → 换更难/更杂乱、
  或带分布漂移、prior 解不掉的基准（如杂乱抓取、长程组合、未见物体）才能体现 CED 增益。
- **任一不过**：Δ 没抓到因果结构，平掉的 SR 就是诚实结论，不能归因于「模拟器太简单」。
""".format(**THRESH)


# ----------------------------- 小工具 -----------------------------
def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten(); b = b.float().flatten()
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    return float((a @ b) / denom)


def _topk_iou(a: torch.Tensor, b: torch.Tensor, frac: float = 0.1) -> float:
    a = a.flatten(); b = b.flatten()
    k = max(1, int(round(frac * a.numel())))
    ia = set(torch.topk(a, k).indices.tolist())
    ib = set(torch.topk(b, k).indices.tolist())
    return len(ia & ib) / max(len(ia | ib), 1)


def _grid2d(vec_n: torch.Tensor) -> np.ndarray:
    n = vec_n.numel(); side = int(round(n ** 0.5))
    return vec_n.detach().float().cpu().numpy().reshape(side, side)


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.median(xs)) if xs else None


def _tile(img: Image.Image, label: str, size: int) -> Image.Image:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    out = Image.new("RGB", (size, size + 16), (255, 255, 255))
    out.paste(img, (0, 16))
    ImageDraw.Draw(out).text((2, 3), label, fill=(0, 0, 0))
    return out


def _hstack(tiles: list[Image.Image], pad: int = 3) -> Image.Image:
    h = max(t.height for t in tiles); w = sum(t.width for t in tiles) + pad * (len(tiles) - 1)
    out = Image.new("RGB", (w, h), (255, 255, 255)); x = 0
    for t in tiles:
        out.paste(t, (x, 0)); x += t.width + pad
    return out


def _heat(vec_n: torch.Tensor, size: int) -> Image.Image:
    return _heatmap_to_rgb(_grid2d(vec_n)).resize((size, size), Image.NEAREST)


def _overlay(base: Image.Image, vec_n: torch.Tensor, size: int, alpha: float = 0.55) -> Image.Image:
    return Image.blend(base.convert("RGB").resize((size, size)), _heat(vec_n, size), alpha)


def _pixel_diff(img_a: Image.Image, img_b: Image.Image, size: int, amp: float = 3.0) -> Image.Image:
    a = np.asarray(img_a.convert("RGB").resize((size, size)), np.float32)
    b = np.asarray(img_b.convert("RGB").resize((size, size)), np.float32)
    d = np.abs(a - b).mean(-1) / 255.0
    return _heatmap_to_rgb(np.clip(d * amp, 0.0, 1.0))


def _decode(model, rae: RAEDinoDecoder | None, feat_nd: torch.Tensor) -> Image.Image | None:
    """[N,D] 归一化 DINO token → RAE 解码 RGB（PIL）。"""
    if rae is None:
        return None
    if rae.has_latent_stats:
        raw = rae.denorm_to_decoder_space(feat_nd.to(rae.device))
    else:
        dev = next(model.parameters()).device
        raw = _jointflow_norm_to_raw_dino(model, feat_nd.to(dev)).to(rae.device)
    rgb = rae.decode_raw_tokens(raw.unsqueeze(0))[0]
    return _tensor_to_pil_rgb(rgb)


# ----------------------------- CED Δ 计算 -----------------------------
@torch.no_grad()
def ced_effect(model, examples: list[dict], *, seed: int, vis_steps: int) -> dict:
    """对一批样本算 CED 的 fdm⁺/fdm⁰、t=0 共享噪声 Δ、多步「想象未来」、池化 Δ 向量与 probe。"""
    dev = next(model.parameters()).device
    batch = model._examples_to_batch(examples, require_future_dino=True, require_action=True)

    cond_pos = model._run_path("fdm", batch)                      # 真实动作条件
    batch_nul = dict(batch)
    batch_nul["action"] = model.make_null_action(batch["action"], batch["examples"])  # a₀
    cond_nul = model._run_path("fdm", batch_nul)                  # 空动作条件

    z_h = model._select_future_dino(batch["dino_1"], batch["examples"]).to(dev, torch.float32)  # 真实未来
    z_t = model._select_future_dino(batch["dino_0"], batch["examples"]).to(dev, torch.float32)  # 当前
    valid = model._future_valid_mask(batch, cond_pos)

    g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
    eps = torch.randn(tuple(z_h.shape), generator=g, dtype=torch.float32).to(dev)  # 两支共享噪声

    was = model.visual_head.training
    model.visual_head.eval()  # 关 dropout——两支必须共享全部随机性，否则 Δ 被污染（与训练 _effect_forward 一致）
    _, v_pos, _ = model.visual_head(cond_pos.float(), z_h, noise=eps, t=0.0, return_pred=True)
    _, v_nul, _ = model.visual_head(cond_nul.float(), z_h, noise=eps, t=0.0, return_pred=True)
    zhk_pos = predict_visual_from_noise(model, cond_pos, eps, steps=vis_steps)  # 多步想象（解码更清晰）
    zhk_nul = predict_visual_from_noise(model, cond_nul, eps, steps=vis_steps)
    if was:
        model.visual_head.train()

    delta = v_pos.float() - v_nul.float()  # [B,N,D] t=0 反事实效果（训练信号本体）
    ced_cfg = model.config.framework.get("ced")
    tau = float(ced_cfg.get("tau", 1.0)) if ced_cfg is not None else 1.0
    pool_w = torch.softmax(delta.norm(dim=-1) / max(tau, 1e-6), dim=1)
    d_vec = (delta * pool_w.unsqueeze(-1)).sum(dim=1)  # [B,D] 池化效果向量
    probe = model.probe_mlp(d_vec) if getattr(model, "probe_mlp", None) is not None else None
    a_true = batch["action"][:, -model.action_horizon:, : model.action_dim].float()

    return dict(z_t=z_t, z_h=z_h, delta=delta, zhk_pos=zhk_pos, zhk_nul=zhk_nul,
                zh1_pos=eps + v_pos, zh1_nul=eps + v_nul, d_vec=d_vec, probe=probe,
                a_true=a_true, valid=valid)


def sample_metrics(out: dict, i: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    z_t, z_h, delta = out["z_t"][i], out["z_h"][i], out["delta"][i]
    zhk_pos, zhk_nul = out["zhk_pos"][i], out["zhk_nul"][i]
    delta_map = delta.norm(dim=-1)                 # CED 预测效果定位 [N]
    change_map = (z_h - z_t).norm(dim=-1)          # 真实未来变化 [N]
    pred_change_map = (zhk_pos - z_t).norm(dim=-1) # 模型自身预测的变化
    eff = (zhk_pos - zhk_nul).norm()
    pred_change = (zhk_pos - z_t).norm().clamp_min(1e-6)
    m = {
        "delta_l2": float(delta.norm()),
        "dvec_l2": float(out["d_vec"][i].norm()),
        "rel_effect": float(eff / pred_change),                       # (S) 动作敏感度
        "loc_corr_change": _pearson(delta_map, change_map),           # (L) 相关
        "loc_iou_change@10": _topk_iou(delta_map, change_map, 0.1),   # (L) IoU
        "loc_corr_predchange": _pearson(delta_map, pred_change_map),
    }
    if out["probe"] is not None:
        a = out["a_true"][i].flatten(); p = out["probe"][i]
        m["probe_mse"] = float(((p - a) ** 2).mean())
    return m, delta_map, change_map


# ----------------------------- 主流程 -----------------------------
def run(args: argparse.Namespace) -> dict:
    run_dir = _resolve_path(args.run_dir)
    ckpt = (_resolve_path(args.ckpt_path) if args.ckpt_path else None)
    if ckpt is None:
        cands = sorted(
            list(run_dir.glob("ckpt/steps_*_pytorch_model.pt")) + list(run_dir.glob("checkpoints/steps_*_pytorch_model.pt")),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not cands:
            raise FileNotFoundError(f"在 {run_dir}/{{ckpt,checkpoints}}/ 找不到 steps_*_pytorch_model.pt")
        ckpt = cands[-1]
    out_dir = _resolve_path(args.output_dir, must_exist=False) if args.output_dir else (run_dir / "ced_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 配置：复用参考工具的 _load_config（强制 load_live_backbone=false、路径重定位），再补本地 base_vlm。
    cfg_args = argparse.Namespace(
        config_yaml=str(args.config_yaml) if args.config_yaml else str(run_dir / "config.full.yaml"),
        data_root_dir=args.data_root_dir, data_mix=args.data_mix,
        dino_feature_dir=args.dino_feature_dir, num_inference_timesteps=args.num_inference_timesteps,
    )
    cfg, config_path = _load_config(cfg_args, ckpt)
    cfg.framework.qwenvl.base_vlm = str(args.base_vlm)
    print(f"[ced-viz] ckpt={ckpt}\n[ced-viz] config={config_path}\n[ced-viz] out={out_dir}\n[ced-viz] device={device}", flush=True)

    register_jointflow_mixtures()
    dataset = build_joint_dataset(cfg, mode=args.mode)
    examples = collect_examples(dataset, num_samples=int(args.num_samples), seed=int(args.seed),
                                start_index=args.start_index, include_invalid=bool(args.include_invalid))
    if not examples:
        raise RuntimeError("没采到样本，检查数据路径 / future_valid 过滤。")

    model, incompatible = load_model(cfg, ckpt, device, strict=not args.non_strict_load)
    if args.non_strict_load:
        print(f"[ced-viz] non-strict: missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}")
    model.set_null_action_table(compute_null_action_table(dataset))  # a₀ 表（与训练同源，从活 Normalizer 算）
    has_probe = getattr(model, "probe_mlp", None) is not None
    print(f"[ced-viz] ced.enabled={getattr(model, 'ced_enabled', False)} probe={'有' if has_probe else '无(非CED ckpt)'}", flush=True)

    # RAE 解码器
    rae = None
    if not args.no_rae:
        rae_image_size = _infer_rae_image_size(examples, int(args.rae_patch_size), args.rae_image_size)
        rae = RAEDinoDecoder(
            repo_dir=_resolve_path(args.rae_repo), decoder_dir=_resolve_path(args.rae_decoder_dir),
            decoder_config=_resolve_path(args.rae_decoder_config), device=device,
            image_size=int(rae_image_size), patch_size=int(args.rae_patch_size),
            dino_dim=int(model.d_dino), allow_token_interpolate=True,
        )
        print(f"[ced-viz] RAE 解码器就绪 image_size={rae_image_size} tokens={_infer_dino_token_count(examples[0])}", flush=True)

    vis_steps = int(args.vis_steps) if args.vis_steps else int(model.visual_head.num_inference_timesteps)
    tile = int(args.tile_size)
    rows: list[dict] = []
    all_probe_pred, all_probe_true = [], []
    saved = 0

    for start in range(0, len(examples), int(args.batch_size)):
        chunk = examples[start:start + int(args.batch_size)]
        out = ced_effect(model, chunk, seed=int(args.seed), vis_steps=vis_steps)
        valid = out["valid"].tolist()
        for i, ex in enumerate(chunk):
            if not valid[i]:
                continue
            m, delta_map, change_map = sample_metrics(out, i)
            m.update(_metadata(ex))
            rows.append(m)
            if out["probe"] is not None:
                all_probe_pred.append(out["probe"][i].detach().cpu())
                all_probe_true.append(out["a_true"][i].flatten().detach().cpu())

            if saved >= int(args.max_visual_samples):
                continue
            # ---- 定性面板 ----
            cur = _read_rgb_views(dataset, ex, "base_index")
            fut = _read_rgb_views(dataset, ex, "future_index")
            vidx = _select_view_index_from_example(model, ex)
            o_t = list(cur.values())[vidx] if len(cur) > vidx else (list(cur.values())[0] if cur else None)
            o_f = list(fut.values())[vidx] if len(fut) > vidx else (list(fut.values())[0] if fut else None)
            blank = Image.new("RGB", (tile, tile), (230, 230, 230))
            o_t = o_t or blank; o_f = o_f or blank

            dec_zt = _decode(model, rae, out["z_t"][i])
            dec_zh = _decode(model, rae, out["z_h"][i])
            dec_pos = _decode(model, rae, out["zhk_pos"][i])
            dec_nul = _decode(model, rae, out["zhk_nul"][i])
            tiles = [_tile(o_t, "o_t 当前", tile), _tile(o_f, "o_t+H 真实未来", tile)]
            if rae is not None:
                tiles += [
                    _tile(dec_zt, "RAE(z_t) 自检", tile),
                    _tile(dec_zh, "RAE(z_H真实) 上限", tile),
                    _tile(dec_pos, "想象·真实动作", tile),
                    _tile(dec_nul, "想象·空动作a0", tile),
                    _tile(_pixel_diff(dec_pos, dec_nul, tile), "|真实-空|像素差", tile),
                ]
            tiles += [
                _tile(_overlay(o_t, delta_map, tile), "Δ叠加 o_t", tile),
                _tile(_overlay(o_t, change_map, tile), "真实变化叠加", tile),
                _tile(_heat(delta_map, tile), f"Δ热力 corr={m['loc_corr_change']:.2f}", tile),
                _tile(_heat(change_map, tile), "真实变化热力", tile),
            ]
            meta = _metadata(ex)
            name = f"panel_{saved:03d}_{_sanitize_name(str(meta.get('dataset_name','')))}_traj{meta.get('trajectory_id','na')}_b{meta.get('base_index','na')}"
            _hstack(tiles).save(out_dir / f"{name}.png")
            np.savez(out_dir / f"{name}.npz", delta_map=_grid2d(delta_map), change_map=_grid2d(change_map))
            saved += 1
        print(f"[ced-viz] batch@{start} valid={sum(valid)} 已存面板={saved}", flush=True)

    # ---- 汇总 + 判读 ----
    agg = {f"{k}_median": _median([r.get(k) for r in rows]) for k in
           ["rel_effect", "loc_corr_change", "loc_iou_change@10", "loc_corr_predchange", "delta_l2", "dvec_l2"]}
    probe_r2 = None
    if all_probe_pred:
        P = torch.stack(all_probe_pred); T = torch.stack(all_probe_true)
        sse = float(((P - T) ** 2).sum()); sst = float(((T - T.mean(0, keepdim=True)) ** 2).sum())
        probe_r2 = 1.0 - sse / max(sst, 1e-8)
        agg["probe_mse_overall"] = float(((P - T) ** 2).mean())
        agg["probe_r2_overall"] = probe_r2

    S = agg.get("rel_effect_median"); Lc = agg.get("loc_corr_change_median"); Li = agg.get("loc_iou_change@10_median")
    checks = {
        "S_sensitivity": (S is not None and S >= THRESH["sensitivity"], S, THRESH["sensitivity"]),
        "L_localization": ((Lc is not None and Lc >= THRESH["loc_corr"]) and (Li is not None and Li >= THRESH["loc_iou"]),
                           f"corr={Lc} iou={Li}", f"corr≥{THRESH['loc_corr']} & iou≥{THRESH['loc_iou']}"),
        "I_identifiability": (probe_r2 is not None and probe_r2 >= THRESH["probe_r2"], probe_r2, THRESH["probe_r2"]) if has_probe
        else (None, "n/a(非CED ckpt)", THRESH["probe_r2"]),
    }
    passed = [v[0] for v in checks.values() if v[0] is not None]
    verdict = "PASS（机制有效：若 SR 仍平 → 归因于 LIBERO 饱和，应换更难基准）" if all(passed) and passed \
        else "FAIL（Δ 未抓到因果结构 → 平掉的 SR 是诚实结论，不能怪模拟器太简单）"

    summary = {
        "ckpt": str(ckpt), "config": str(config_path), "run_dir": str(run_dir),
        "num_valid": len(rows), "visuals_saved": saved, "vis_steps": vis_steps,
        "thresholds": THRESH, "aggregate": agg,
        "checks": {k: {"pass": v[0], "value": v[1], "threshold": v[2]} for k, v in checks.items()},
        "verdict": verdict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "CRITERION.md").write_text(CRITERION_MD, encoding="utf-8")

    print("\n================ CED 机制判读 ================", flush=True)
    print(f"  样本(valid)={len(rows)}  面板={saved}  -> {out_dir}", flush=True)
    print(f"  (S) 动作敏感度 rel_effect  中位数={S}   (阈值≥{THRESH['sensitivity']})", flush=True)
    print(f"  (L) 因果定位  corr={Lc} iou@10={Li}   (阈值 corr≥{THRESH['loc_corr']} & iou≥{THRESH['loc_iou']})", flush=True)
    print(f"  (I) 可辨识性  probe R²={probe_r2}   (阈值≥{THRESH['probe_r2']})", flush=True)
    for k, v in checks.items():
        flag = "—" if v[0] is None else ("✓" if v[0] else "✗")
        print(f"      {flag} {k}: {v[1]}", flush=True)
    print(f"  判读：{verdict}", flush=True)
    print("  阈值含义与后续动作见 CRITERION.md", flush=True)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CED 反事实效果可视化 + 定量判读。",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir", required=True, help="Z_ws/CED/<RUN_NAME> 目录（含 config.full.yaml 与 ckpt/）。")
    p.add_argument("--ckpt_path", default=None, help="不给则自动取 run_dir 下 step 最大的 ckpt。")
    p.add_argument("--config_yaml", default=None, help="不给则用 <run_dir>/config.full.yaml。")
    p.add_argument("--output_dir", default=None, help="默认 <run_dir>/ced_viz。")
    p.add_argument("--data_root_dir", default=DEFAULT_DATA)
    p.add_argument("--data_mix", default=None, help="覆盖 data_mix（默认用配置里的 libero_all）。")
    p.add_argument("--dino_feature_dir", default=None)
    p.add_argument("--base_vlm", default=DEFAULT_QWEN)
    p.add_argument("--mode", default="train", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--start_index", type=int, default=None)
    p.add_argument("--include_invalid", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--non_strict_load", action="store_true")
    p.add_argument("--max_visual_samples", type=int, default=12)
    p.add_argument("--vis_steps", type=int, default=None, help="多步想象解码步数；默认用 visual head 自带值。")
    p.add_argument("--num_inference_timesteps", type=int, default=None, help="覆盖 visual head 去噪步数（影响 vis 默认）。")
    p.add_argument("--tile_size", type=int, default=160)
    p.add_argument("--no_rae", action="store_true", help="跳过 RAE 解码（只出热力图/叠加，更快）。")
    p.add_argument("--rae_repo", default=DEFAULT_RAE_REPO)
    p.add_argument("--rae_decoder_dir", default=DEFAULT_RAE_DECODER)
    p.add_argument("--rae_decoder_config", default=DEFAULT_RAE_CONFIG)
    p.add_argument("--rae_image_size", type=int, default=None)
    p.add_argument("--rae_patch_size", type=int, default=16)
    return p


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
