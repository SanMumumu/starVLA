# Robotwin eval debug

This folder contains isolated debug helpers for RoboTwin evaluation and WAM checkpoint/input parity checks.

Files:
- `run_robotwin_eval_debug.sh`: runs a low-shot debug evaluation with debug-friendly output paths.
- `run_robotwin_ckpt_and_eval.sh`: runs checkpoint-load inspection plus a single-step eval smoke test.
- `check_robotwin_ckpt_load.py`: inspects checkpoint loading parity and missing/unexpected keys.
- `check_robotwin_eval_input.py`: exercises the eval client input path against a running policy server.
