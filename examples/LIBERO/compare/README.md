# LIBERO Dual-Query World-to-Action Comparison

This directory contains the historical LIBERO comparison matrix for injecting
predicted world features into the QwenGR00T action expert. It belongs to the
Dual-Query project. The files remain available because existing runs and
checkpoints record these config names.

## Configuration layout

```text
compare/
├── base/base_compare.yaml
├── experiment_manifest.yaml
├── generate_configs.py
├── run_compare.sh
└── configs/
```

`base_compare.yaml` contains the common recipe. The manifest contains only the
differences for each experiment. Generated files in `configs/` are complete,
standalone snapshots; edit the base or manifest and regenerate them instead of
editing generated YAMLs by hand.

```bash
# Generate every config.
python examples/LIBERO/compare/generate_configs.py

# Generate selected configs.
python examples/LIBERO/compare/generate_configs.py m0q_dual_query_control m5_dual_gate_best

# Train one configuration; the checkpoint argument is optional.
bash examples/LIBERO/compare/run_compare.sh <experiment_name> [initial_checkpoint]
```

## Experiment matrix

| Experiment | Guidance mode | World signal |
| --- | --- | --- |
| `m0_baseline` | `none` | none |
| `m0q_dual_query_control` | `none` | dual-query control |
| `m1_1_hfuture_concat` | `concat` | `h_future` |
| `m1_2_zpred_concat` | `concat` | `z_pred` |
| `m1_3_dzpred_concat` | `concat` | `delta_z_pred` |
| `warmup_oracle_absolute` | `concat` | oracle absolute future |
| `warmup_predicted_absolute` | `concat` | predicted absolute future |
| `warmup_oracle_delta` | `concat` | oracle delta future |
| `warmup_predicted_delta` | `concat` | predicted delta future |
| `m3_qformer_best` | `qformer` | `z_pred` |
| `m4_alternate_best` | `alternate_xattn` | `z_pred` |
| `m5_dual_gate_best` | `dual_xattn` | `z_pred` |
| `m6_adaln_best` | `adaln` | `z_pred` |
| `m6plus_dual_adaln_best` | `dual_xattn_adaln` | `z_pred` |

The sample variants apply frame-uniform sampling to the corresponding M5 or
M6+ configuration.

## Data and gradient contracts

- Action memory always keeps the original action query and Qwen context.
- Passive world prediction does not consume the action target, avoiding an
  action-to-world-to-action shortcut.
- Oracle features are a training-only bridge and cannot be used during live
  evaluation.
- `detach_world=true` prevents action loss from changing the predictor.
- Evaluation ablations use
  `framework.wam.guidance.world_eval_mode=correct|off|zero|shuffled|wrong_task|gt`.
  The `gt` mode is offline-only because a live rollout has no future frame.

For a controlled warmup comparison, start all post-warmup structures from the
same oracle checkpoint and train them for the same number of optimizer steps:

```bash
CMP=examples/LIBERO/compare/run_compare.sh
OUT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_compare

bash "$CMP" warmup_oracle_absolute
ORACLE_RUN=$(find "$OUT" -maxdepth 1 -type d -name '*_cmp_warmup_oracle_absolute' | sort | tail -1)

MAX_STEPS=50000 bash "$CMP" warmup_predicted_absolute "$ORACLE_RUN"
MAX_STEPS=50000 bash "$CMP" m3_qformer_best "$ORACLE_RUN"
MAX_STEPS=50000 bash "$CMP" m4_alternate_best "$ORACLE_RUN"
MAX_STEPS=50000 bash "$CMP" m5_dual_gate_best "$ORACLE_RUN"
MAX_STEPS=50000 bash "$CMP" m6_adaln_best "$ORACLE_RUN"
MAX_STEPS=50000 bash "$CMP" m6plus_dual_adaln_best "$ORACLE_RUN"
```

This comparison suite is retained for reproducibility. Current RoboTwin and
RoboDojo recipes are documented at the repository root and use the IID-only
contracts in their benchmark-specific YAMLs.
