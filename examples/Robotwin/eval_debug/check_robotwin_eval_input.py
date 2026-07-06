#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
from PIL import Image

from examples.Robotwin.eval_files.model2robotwin_interface import ModelClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5694)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--horizon", type=int, default=16)
    args = ap.parse_args()

    model = ModelClient(
        policy_ckpt_path=args.ckpt,
        host=args.host,
        port=args.port,
        horizon=args.horizon,
        action_mode="abs",
    )

    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    example = {
        "lang": "debug task",
        "image": [np.asarray(img), np.asarray(img), np.asarray(img)],
        "state": np.zeros(14, dtype=np.float32),
    }
    print("[DEBUG] server_meta action_chunk_size:", model.action_chunk_size)
    print("[DEBUG] stepping once...")
    out = model.step(example, step=0)
    print("[DEBUG] action shape:", out.shape)
    print("[DEBUG] action head sample:", out[:5])


if __name__ == "__main__":
    main()
