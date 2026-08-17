#!/usr/bin/env bash
set -euo pipefail

# The sharded evaluator already owns task-count validation, resumable JSONL
# output, port preflight, and aggregation.  Pinning it to one shard gives the
# maintained single-GPU LIBERO-plus path without duplicating that logic.
export NUM_CLIENTS=1
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_libero_plus_8.sh"
