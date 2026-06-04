#!/usr/bin/env bash
set -euo pipefail

CONFIG_YAML="${1:-starVLA/jointflow/configs/jointflow_libero.yaml}"

CUDA_VISIBLE_DEVICES=7 python -m starVLA.jointflow.data.dino_precompute \
  --config_yaml "${CONFIG_YAML}"

