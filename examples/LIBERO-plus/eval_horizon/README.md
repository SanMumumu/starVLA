# LIBERO-Plus AIDI 8-GPU Evaluation

This folder runs LIBERO-Plus evaluation on AIDI:

- 1 server job: starts 8 policy servers.
- 1 client job: starts 8 LIBERO-Plus clients and aggregates results.

## 1. Server

Edit only this block in `run_aidi_server.sh`:

```bash
DEFAULT_CKPT="/path/to/pytorch_model.pt"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_SERVERS="8"
```

Submit:

```bash
cd examples/LIBERO-plus/eval_horizon
aidi-inf-cli job submit -f job_server.yaml -q project-h20-horizon-labs-acloud
```

If the job cannot find `run_aidi_server.sh`, check `RUN_SCRIPTS` in `job_server.yaml`:

```yaml
RUN_SCRIPTS: "${WORKING_PATH}/starVLA/examples/LIBERO-plus/eval_horizon/run_aidi_server.sh"
```

Copy `CLIENT_HOST` from the server log:

```text
CLIENT_HOST=10.x.x.x
BASE_PORT=6698
PORT_RANGE=6698-6705
NUM_SERVERS=8
```

## 2. Client

Edit only this block in `run_aidi_client.sh`:

```bash
DEFAULT_CKPT="/path/to/pytorch_model.pt"
DEFAULT_HOST="10.x.x.x"
DEFAULT_BASE_PORT="6698"
DEFAULT_NUM_CLIENTS="8"
DEFAULT_SAVE_VIDEO="0"
DEFAULT_EXPECTED_TOTAL="10030"
```

`DEFAULT_CKPT` and `DEFAULT_BASE_PORT` must match the server job.

Submit:

```bash
cd examples/LIBERO-plus/eval_horizon
aidi-inf-cli job submit -f job_client.yaml -q project-h20-horizon-labs-acloud
```

## Outputs

Results are written to:

```text
<model_root>/libero_plus_eval_results_<run_name>_<ckpt_tag>_8shard/
```

Check:

- `aggregate.txt`
- `aggregate_results.json`
- `logs/`
- `shards/`
