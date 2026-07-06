# LIBERO AIDI 8-GPU Evaluation

This folder runs LIBERO evaluation on AIDI with:

- 1 server job: starts 8 policy servers.
- 1 client job: starts 8 LIBERO clients and aggregates results.

## 1. Server

Edit only this block in `run_aidi_server.sh`:

```bash
DEFAULT_CKPT="/path/to/pytorch_model.pt"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_SERVERS="8"
```

Submit:

```bash
cd examples/LIBERO/eval_horizon
aidi-inf-cli job submit -f job_server.yaml -q project-h20-robot-lab-acloud-langfang
cd -
```

If the job cannot find `run_aidi_server.sh`, check `RUN_SCRIPTS` in `job_server.yaml`:

```yaml
RUN_SCRIPTS: "${WORKING_PATH}/starVLA/examples/LIBERO/eval_horizon/run_aidi_server.sh"
```

Find this block in the server log:

```text
CLIENT_HOST=10.x.x.x
BASE_PORT=6698
PORT_RANGE=6698-6705
NUM_SERVERS=8
```

Copy `CLIENT_HOST` for the client job.

## 2. Client

Edit only this block in `run_aidi_client.sh`:

```bash
DEFAULT_CKPT="/path/to/pytorch_model.pt"
DEFAULT_HOST="10.x.x.x"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_CLIENTS="8"
DEFAULT_TASK_SUITES="libero_spatial libero_object libero_goal libero_10"
DEFAULT_NUM_TRIALS_PER_TASK="50"
```

`DEFAULT_CKPT` and `DEFAULT_BASE_PORT` must match the server job.

Submit:

```bash
cd examples/LIBERO/eval_horizon
aidi-inf-cli job submit -f job_client.yaml -q project-h20-robot-lab-acloud-langfang
cd -
```

`job_client_libero.yaml` is the same client job config.

## Outputs

Results are written to:

```text
<model_root>/eval_results_<ckpt_tag>_8shard/
```

Check:

- `aggregate.txt`
- `aggregate_results.json`
- `client_logs/`
- `shard_*/<suite>/eval.log`
