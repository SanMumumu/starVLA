#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T


def pick_keys(state_dict, prefixes):
    out = []
    for k in state_dict.keys():
        if any(k.startswith(p) for p in prefixes):
            out.append(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    model = Qwen_GR00T(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model_sd = model.state_dict()
    ckpt_keys = set(ckpt.keys())
    model_keys = set(model_sd.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    print("[DEBUG] total model keys:", len(model_keys))
    print("[DEBUG] total ckpt keys:", len(ckpt_keys))
    print("[DEBUG] missing keys sample:")
    for k in missing[:200]:
        print("  MISSING", k)
    print("[DEBUG] unexpected keys sample:")
    for k in unexpected[:200]:
        print("  UNEXPECTED", k)

    watched = [
        "action_model.model.transformer_blocks.0.world_gate",
        "action_model.model.world_to_temb.weight",
        "action_model.model.world_to_temb.bias",
        "world_adapter",
        "world_pooler",
        "world_qformer",
        "world_fusion",
    ]
    print("[DEBUG] watched tensors in model/ckpt:")
    for key in watched:
        mv = model_sd.get(key, None)
        cv = ckpt.get(key, None)
        print(f"  {key}: model={'yes' if mv is not None else 'no'} ckpt={'yes' if cv is not None else 'no'}")
        if mv is not None:
            print("    model shape", tuple(mv.shape), "mean", float(mv.float().mean()), "std", float(mv.float().std()))
        if cv is not None:
            print("    ckpt  shape", tuple(cv.shape), "mean", float(cv.float().mean()), "std", float(cv.float().std()))

    print("[DEBUG] model keys with world/gate:")
    for k in sorted([k for k in model_keys if any(s in k for s in ["world_gate", "world_to_temb", "world_adapter", "world_pooler", "world_qformer", "world_fusion"])])[:200]:
        print(" ", k)


if __name__ == "__main__":
    main()
