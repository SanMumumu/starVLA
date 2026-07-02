#!/usr/bin/env python
"""Generate full per-experiment YAMLs from base_compare.yaml + experiment_manifest.yaml.

中文注释（plan §8）：base 存公共底座，manifest 只存每个实验的差异；这里用 OmegaConf.merge 合成
完整、独立、可复现的 configs/<name>.yaml，避免人工复制导致配置漂移。

用法：
    python examples/LIBERO/compare/generate_configs.py            # 生成全部
    python examples/LIBERO/compare/generate_configs.py m0q m1_1   # 只生成指定实验
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
BASE = HERE / "base" / "base_compare.yaml"
MANIFEST = HERE / "experiment_manifest.yaml"
OUT_DIR = HERE / "configs"

# 注入结构需要 action DiT 内建 world 子模块（已实现）；标注供查阅。
DIT_MODES = {"alternate_xattn", "dual_xattn", "adaln", "dual_xattn_adaln"}


def main(argv):
    base = OmegaConf.load(BASE)
    manifest = OmegaConf.load(MANIFEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    names = argv or list(manifest.keys())
    unknown = [n for n in names if n not in manifest]
    if unknown:
        raise SystemExit(f"未知实验名: {unknown}\n可选: {list(manifest.keys())}")

    for name in names:
        override = manifest[name]
        cfg = OmegaConf.merge(base, override)
        if "run_id" not in override:
            cfg.run_id = f"cmp_{name}"
        out = OUT_DIR / f"{name}.yaml"
        OmegaConf.save(cfg, out)
        # 解析回读校验 + 标记需要 DiT 的实验
        _ = OmegaConf.load(out)
        mode = str(cfg.framework.wam.guidance.get("mode", "none")).lower()
        tag = "  [DiT-mode]" if mode in DIT_MODES else ""
        sig = cfg.framework.wam.guidance.get("signal", "none")
        en = cfg.framework.wam.guidance.get("enabled", False)
        print(f"✓ {out.name:<34} enabled={en!s:<5} mode={mode:<16} signal={sig}{tag}")

    print(f"\n生成 {len(names)} 个配置 → {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
