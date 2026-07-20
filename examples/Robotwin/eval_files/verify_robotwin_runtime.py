#!/usr/bin/env python3
"""Validate the RoboTwin simulator/CuRobo ABI before parallel evaluation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-path", type=Path, required=True)
    args = parser.parse_args()

    root = args.robotwin_path.resolve()
    eval_entry = root / "script" / "eval_policy.py"
    if not eval_entry.is_file():
        raise FileNotFoundError(eval_entry)

    os.chdir(root)
    sys.path.insert(0, str(root))

    import torch

    print(
        "[RoboTwin ABI] "
        f"python={sys.executable} python_version={sys.version.split()[0]} "
        f"torch={torch.__version__} torch_path={torch.__file__} "
        f"torch_cuda={torch.version.cuda} "
        f"cxx11_abi={getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 'unknown')} "
        f"extensions={os.environ.get('TORCH_EXTENSIONS_DIR', '<default>')}",
        flush=True,
    )

    import curobo

    print(f"[RoboTwin ABI] curobo_path={curobo.__file__}", flush=True)
    from envs.robot.planner import CuroboPlanner

    if not isinstance(CuroboPlanner, type):
        raise TypeError(f"CuroboPlanner must be a class, got {type(CuroboPlanner).__name__}")
    print("ROBOTWIN_CUROBO_ABI_PASS", flush=True)


if __name__ == "__main__":
    main()
