# LIBERO Automatic Evaluation

These internal scripts distribute checkpoint-by-suite evaluations across a GPU
pool. They are environment-specific; adapt paths and environment activation to
the target cluster. The evaluation protocol itself follows
`examples/LIBERO/README.md`.

## Entry points

| Script | Purpose |
| --- | --- |
| `auto_eval_libero.sh` | Select checkpoints, task suites, GPUs, and schedule jobs. |
| `eval_libero_parall.sh` | Start one policy server, run one suite, then stop the server. |
| `see_sr_auto.sh` | Aggregate success rates from evaluation logs. |
| `rm_video.sh` | Remove evaluation videos when they are no longer needed. |

## Configure and run

Edit the variables near the top of `auto_eval_libero.sh`:

```bash
CKPT_DIR="results/Checkpoints/my_experiment/checkpoints"
CKPT_LIST=()
TASK_SUITES=(libero_10 libero_goal libero_object libero_spatial)
GPU_LIST=(0 1 2 3 4 5 6 7)
```

Then run:

```bash
bash examples/LIBERO/eval_files/auto_eval_scripts/auto_eval_libero.sh
```

Each job receives a GPU by `job_index % num_gpus` and a distinct port at
`BASE_PORT + job_index`. A new wave starts after every active GPU job in the
current wave finishes.

Aggregate the default experiment or specify another result directory:

```bash
bash examples/LIBERO/eval_files/auto_eval_scripts/see_sr_auto.sh
bash examples/LIBERO/eval_files/auto_eval_scripts/see_sr_auto.sh results/Checkpoints/my_experiment
```
