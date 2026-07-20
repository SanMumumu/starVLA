<p align="center">
  <img src="assets/logo.svg" alt="StarVLA logo" width="10%">
</p>

# StarVLA Action-World Research Workspace

This repository is a focused StarVLA workspace for two independent research
projects: **Dual-Query WAM** and **Action-World Co-Flow**. They share the
dataloader, Qwen VLM wrapper, trainer, checkpoint loader, policy server, and
benchmark adapters. Their model-specific code and experiment configs remain
separate.

The workspace is derived from the
[upstream StarVLA project](https://github.com/starVLA/starVLA). Keep upstream
framework names and checkpoint keys stable when extending this fork.

## Project boundaries

| Project | Framework entry point | Model package | Primary configs |
| --- | --- | --- | --- |
| Dual-Query WAM | `starVLA/model/framework/VLM4A/QwenGR00T.py` | `wam_guidance.py` and the checkpoint-compatible `jointflow/` packages | `examples/Robotwin/train_files/robotwin_wam_*.yaml`, `examples/RoboDojo/train_files/*wam*.yaml` |
| Action-World Co-Flow | `starVLA/model/framework/VLM4A/QwenActionWorldCoFlow.py` | `starVLA/model/framework/VLM4A/action_world_coflow/` | `examples/Robotwin/train_files/robotwin_action_world_coflow_better.yaml` |

The `jointflow` directory name is a historical checkpoint/config compatibility
surface for Dual-Query WAM. It is not the Co-Flow implementation and must not
be renamed without an explicit checkpoint migration.

### Dual-Query WAM

Dual-Query WAM extracts action and future queries from the same Qwen context.
The world predictor is trained first with IID future targets; a detached gate
stage then teaches the action expert to use predicted future tokens without
allowing action loss to modify the predictor.

Current RoboTwin recipes are:

- `robotwin_wam_warmup_rand.yaml`
- `robotwin_wam_warmup_clean.yaml`
- `robotwin_wam_gate_rand2clean.yaml`
- `robotwin_wam_gate_clean2clean.yaml`
- `robotwin_wam_joint_detached_iid.yaml`

The warmup and gate stages retain separate YAMLs because their trainable
parameter sets, checkpoint initialization, and loss contracts differ. Legacy
configs and inference branches remain available for already-trained
checkpoints.

### Action-World Co-Flow

Co-Flow jointly models exactly one H16 action bridge and one t+16 future-latent
bridge with a causal context boundary and a shared noise-plane sampler. The
robot re-observes after executing the chunk. Inference supports:

- `policy`: predict actions from the current observation.
- `diagonal`: jointly sample actions and the future latent state.

There is no second action/future segment and no H32 Co-Flow compatibility path.
Use the Co-Flow checkpoint verifier before launching policy servers.

## Repository layout

```text
deployment/                         shared policy server and checkpoint contract
examples/
  LIBERO/                           active LIBERO training/evaluation utilities
  LIBERO-plus/                      active LIBERO-plus evaluation utilities
  RoboDojo/                         RoboDojo data, training, and evaluation adapters
  Robotwin/                         RoboTwin data, training, and evaluation adapters
starVLA/
  config/                           shared configuration and DeepSpeed templates
  dataloader/                       dataset, state, image, and normalization contracts
  model/framework/VLM4A/            Dual-Query and Co-Flow framework entry points
  model/modules/                    reusable VLM, action, and world-model modules
  training/                         shared training loop, optimizer, logging, validation
tools/                              offline diagnostics and repository checks
```

Local external repositories, downloaded models/datasets, diagnostic workspaces,
debug checkpoints, caches, generated plots, and cluster runbooks are not managed
source. They are intentionally preserved locally and excluded from source-code
hygiene decisions.

## Shared training and inference contracts

Training and evaluation must agree on all of the following:

1. **State input**: active RoboTwin and RoboDojo WAM/Co-Flow recipes use
   `datasets.vla_data.include_state: true`.
2. **Normalization**: state and action statistics use the FastWAM-compatible
   z-score transform selected by the dataset config.
3. **Image layout**: RoboDojo combines high, left-wrist, and right-wrist views
   into the same FastWAM composite image in both training and inference.
4. **Noise**: current IID recipes set `use_correlated_noise: false`; clean data
   never reuses a randomized-data Cholesky factor.
5. **Future stride**: the data target, framework horizon, checkpoint contract,
   server metadata, and client replanning horizon must be checked together.
6. **Checkpoint compatibility**: full saved config metadata is authoritative;
   do not infer an old checkpoint contract from a newly edited YAML.

## Training entry points

All distributed jobs enter through the same launcher:

```bash
bash run_aidi_rbtw.sh examples/Robotwin/train_files/robotwin_wam_warmup_rand.yaml
bash run_aidi_rbtw.sh examples/Robotwin/train_files/robotwin_action_world_coflow_better.yaml
```

Cluster job YAMLs supply worker count, GPUs per worker, and the exact config
path. Keep DeepSpeed micro-batch size, YAML per-device batch size, gradient
accumulation, and expected global batch size consistent.

## Evaluation entry points

RoboTwin:

```bash
# Shared eight-server launcher.
bash examples/Robotwin/eval_files/run_policy_servers_8.sh

# Dual-Query/FastWAM evaluation.
bash examples/Robotwin/eval_files/eval_robotwin_8clients_fastwam.sh

# Co-Flow policy or diagonal evaluation.
bash examples/Robotwin/eval_files/eval_robotwin_8clients_action_world_coflow.sh
```

RoboDojo:

```bash
bash examples/RoboDojo/eval_files/run_aidi_robodojo_fast_full.sh
bash examples/RoboDojo/eval_files/run_aidi_robodojo_visualize.sh
```

Run the matching verifier before starting servers:

```bash
python examples/Robotwin/eval_files/verify_fastwam_checkpoint_contract.py --checkpoint /path/to/checkpoint.pt
python examples/Robotwin/eval_files/verify_action_world_coflow_checkpoint_contract.py --checkpoint /path/to/checkpoint.pt
python examples/RoboDojo/eval_files/verify_robodojo_checkpoint_contract.py --checkpoint /path/to/checkpoint.pt
```

## Development checks

Run focused regressions after changing shared model, trainer, dataloader, or
deployment code:

```bash
python -m compileall -q deployment examples starVLA tools
pytest -q starVLA/model/framework/VLM4A/action_world_coflow/test_action_world_coflow.py
pytest -q starVLA/model/framework/VLM4A/test_wam_fastwam_contract.py
pytest -q starVLA/dataloader/test_fastwam_robotwin_dataset.py
```

Shell launchers should also pass `bash -n`. Repository hygiene checks enforce
English source and documentation outside the explicitly exempt local runbook
tree.

## Compatibility policy

- Do not rename framework registry keys, checkpoint state-dict prefixes, or
  historical config fields without a migration path.
- Do not silently reuse a correlation matrix across datasets.
- Do not change training-only preprocessing without updating the policy client.
- Add new experiments as new YAMLs; do not mutate a config used to produce a
  checkpoint whose evaluation still matters.
- Preserve the baseline, warmup, gate, FastWAM, and historical H32 evaluation
  paths while new H16 experiments are active.

See `LICENSE` for the repository license and the upstream project for general
StarVLA documentation.
