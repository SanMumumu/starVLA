#!/usr/bin/env python
"""Generate complete experiment YAMLs from a shared base and manifest.

The base stores common settings while the manifest stores only per-experiment
overrides. OmegaConf.merge produces standalone, reproducible configs and avoids
configuration drift from manual copies.

Usage:
    python examples/LIBERO/compare/generate_configs.py
    python examples/LIBERO/compare/generate_configs.py m0q m1_1
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
BASE = HERE / "base" / "base_compare.yaml"
MANIFEST = HERE / "experiment_manifest.yaml"
OUT_DIR = HERE / "configs"

DIT_MODES = {"alternate_xattn", "dual_xattn", "adaln", "dual_xattn_adaln"}


def main(argv):
    base = OmegaConf.load(BASE)
    manifest = OmegaConf.load(MANIFEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    names = argv or list(manifest.keys())
    unknown = [n for n in names if n not in manifest]
    if unknown:
        raise SystemExit(f"Unknown experiment names: {unknown}\nAvailable: {list(manifest.keys())}")

    for name in names:
        override = manifest[name]
        cfg = OmegaConf.merge(base, override)
        if "run_id" not in override:
            cfg.run_id = f"cmp_{name}"
        out = OUT_DIR / f"{name}.yaml"
        OmegaConf.save(cfg, out)
        _ = OmegaConf.load(out)
        mode = str(cfg.framework.wam.guidance.get("mode", "none")).lower()
        tag = "  [DiT-mode]" if mode in DIT_MODES else ""
        sig = cfg.framework.wam.guidance.get("signal", "none")
        en = cfg.framework.wam.guidance.get("enabled", False)
        print(f"✓ {out.name:<34} enabled={en!s:<5} mode={mode:<16} signal={sig}{tag}")

    print(f"\nGenerated {len(names)} configs in {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
