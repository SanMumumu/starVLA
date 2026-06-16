#!/usr/bin/env bash
set -euo pipefail

CONFIG_YAML="${1:-starVLA/jointflow/configs/jointflow_libero.yaml}"

CUDA_VISIBLE_DEVICES=0 python -m starVLA.jointflow.dino_precompute.dino_precompute \
  --config_yaml "${CONFIG_YAML}"

