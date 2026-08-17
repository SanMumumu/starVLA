# Current Rynn H8 recipe (WAM-exp5-aligned hypers)

The canonical config is `train_files/rynn_base_h8_50k.yaml`. Model ABI is Rynn:
third-person + wrist → one 512x256 `[third-person | wrist]` composite, 7-D
`delta_qpos`, H8 full-chunk execution. Non-model train controls follow WAM
`0620_wam_exp5_2b_bigbs_policy` (LIBERO-plus ~81.7%): **`include_state: false`**,
**60K steps**, warmup 3K, action LR `5e-5`, empty `freeze_modules`, logging 100.
Text / subtask supervision stays **off**. Global batch remains **256** on one
16-GPU node (`16 × 16 × 1`); WAM's `accum=8` is not copied (Rynn MoT OOM risk).

Train on one 16-GPU node from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
LIBERO_DATA_ROOT=/path/to/libero_lerobot_root \
RYNN_BASE_VLM=/path/to/rynnbrain1.1-2B \
RUN_ROOT_DIR=/path/to/outputs \
bash examples/LIBERO/train_files/run_libero_train.sh
```

The maintained contract is global batch 256, **60K** steps, and a checkpoint every
5K steps. The default topology is `16 x 16 GPUs x 1 accumulation = 256`; use
`NUM_PROCESSES=1` for the supported `4 x 1 x 64 = 256` single-GPU mode.

## Dense-current-DINO variant

`train_files/rynn_base_h8_current_dino_fullres_50k.yaml` is the single-variable
variant without DINO pooling on the current observation. Its clean current
prefix retains the full 16x32 grid (512 tokens), while the future denoising
target keeps `dino_pool: 2` and the original 8x16 grid (128 tokens). H8, 60K
steps, WAM-aligned optimizer settings, nostate data ABI, and global batch 256
are unchanged.

Train it directly with:

```bash
LIBERO_DATA_ROOT=/path/to/libero_lerobot_root \
RYNN_BASE_VLM=/path/to/rynnbrain1.1-2B \
bash examples/LIBERO/train_files/run_rynn_base_h8_current_dino_fullres_50k.sh
```

For evaluation, start `run_policy_server.sh` with the variant checkpoint and
run the matching simulator wrapper:

```bash
CKPT=/path/to/steps_50000_pytorch_model.pt HOST=127.0.0.1 PORT=6694 \
bash examples/LIBERO/eval_files/eval_libero_rynn_h8_current_dino_fullres.sh
```

For ordinary evaluation, use two terminals. Start the server in the StarVLA
environment:

```bash
CKPT=/path/to/steps_50000_pytorch_model.pt GPU_ID=0 PORT=6694 \
bash examples/LIBERO/eval_files/run_policy_server.sh
```

Run the simulator in the LIBERO environment:

```bash
CKPT=/path/to/steps_50000_pytorch_model.pt HOST=127.0.0.1 PORT=6694 \
LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/libero/python \
bash examples/LIBERO/eval_files/eval_libero.sh
```

Evaluation is single-server/single-client by default. The current Rynn
checkpoint must use `STRIP_DINO_KEYS=0`; the maintained source launcher in
`执行脚本/LIBERO/run.sh` (copied to
`/home/users/sen02.wang/workspace/starvla_dev/LIBERO/run.sh` on the cluster)
makes that the default and also exposes a single-client full LIBERO-plus
evaluation.

The LIBERO recipe deliberately does not set an action FP32/BF16 override.
Model precision inherits the standard StarVLA training and serving runtime.

## Upstream / WAM control-variable alignment

Comparable optimizer fields still match StarVLA's upstream
`starvla_cotrain_libero.yaml`. Schedule and state conditioning follow WAM
exp5 (`0620_wam_exp5_2b_bigbs_policy`) instead of the older Rynn 50K+state recipe.

| Contract | WAM exp5 (81.7%) | Maintained Rynn recipe |
| --- | --- | --- |
| action | 7-D `delta_qpos`, H8 | same |
| state | `include_state: false` | same |
| data | `libero_all`, no sequential sampling | same |
| text / subtask loss | none for LIBERO-plus | `text_supervision` / `text_annotations` false |
| VLA micro-batch | 16/GPU | same on the 16-GPU topology |
| steps / warmup / save | 60K / 3K / 5K | same |
| base / interface / action LR | 2.5e-5 / 1e-5 / **5e-5** | same |
| freeze | effectively empty | `freeze_modules: ""` |
| logging | 100 | same |
| observation | WAM default views | Rynn 512x256 dual-view composite (model ABI) |
| backbone | Qwen3-VL-2B + WAM | RynnBrain1.1 + causal-DINO MoT (model ABI) |
| effective batch | yaml `accum=8` (GPU-count dependent) | fixed **256** (`accum=1` on 16 GPU) |

Global batch is the intentional non-copy: WAM's `accum=8` on a full node would
push Rynn MoT far past the maintained 256 contract and likely OOM.

---

# 🚀 LIBERO Evaluation

This document provides instructions for reproducing our **experimental results** with LIBERO.  
The evaluation process consists of two main parts:  

1. Setting up the `LIBERO` environment and dependencies.  
2. Running the evaluation by launching services in both `starVLA` and `LIBERO` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.  

> 💡 **AMD GPU Support:** Community members have verified that starVLA also works on **AMD Instinct MI300X** GPUs with ROCm 6.4 — with zero source code changes. The only modification needed is setting `--framework.qwenvl.attn_implementation sdpa`. For a detailed setup guide and benchmark results, see [Issue #254](https://github.com/starVLA/starVLA/issues/254).

---


## ⬇️ 0. Download Checkpoints


We provide a collection of pretrained checkpoints on Hugging Face to make community evaluation easier: [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/bench-libero). Their corresponding results on LIBERO are summarized in the table below.

### 📊 Experimental Results

| Model               | Steps | Epochs | Spatial | Object | Goal | Long  | Avg   |
|---------------------|-------|--------|---------|--------|------|-------|-------|
| $\pi_0$+FAST | -     | -      | 96.4    | 96.8   | 88.6 | 60.2  | 85.5  |
| OpenVLA-OFT | 175K  | 223    | 97.6    | 98.4   | 97.9 | 94.5  | 97.1  |
| $\pi_0$             | -     | -      | 96.8    | 98.8   | 95.8 | 85.2  | 94.1  |
| GR00T-N1.5 | 20K   | 203    | 92.0    | 92.0   | 86.0 | 76.0  | 86.5  |
| **StarVLA-FAST (Qwen2.5-VL)** | 30K   | 9.54   | 97.3    | 97.2   | 96.1 | 90.2  | 95.2  |
| **StarVLA-OFT (Qwen2.5-VL)**  | 30K   | 9.54   | 97.4    | 98.0   | 96.8 | 92.0  | 96.1  |
| **StarVLA-π (Qwen2.5-VL)**    | 30K   | 9.54   | 98.2    | 99.2   | 95.6 | 88.4  | 95.4  |
| **StarVLA-GR00T (Qwen2.5-VL)**| 30K   | 9.54   | 97.8    | 98.2   | 94.6 | 90.8  | 95.4  |
| **StarVLA-FAST (Qwen3-VL)**   | 30K   | 9.54   | 97.3    | 97.4   | 96.3 | 90.6  | 95.4  |
| **StarVLA-OFT (Qwen3-VL)**    | 30K   | 9.54   | 97.8    | 98.6   | 96.2 | 93.8  | 96.6  |
| **StarVLA-π (Qwen3-VL)**      | 30K   | 9.54   | 98.8    | 99.6   | 95.8 | 88.4  | 95.7  |
| **StarVLA-GR00T (Qwen3-VL)**  | 30K   | 9.54   | 97.8    | 98.8   | 97.4 | 92.0  | 96.5  |

We train one policy for all 4 suites. All
scores are averaged over 500 trials for each task suite (10 tasks × 50 episodes).

---


## 📦 1. Environment Setup

To set up the environment, please first follow the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) to install the base `LIBERO` environment.  

⚠️ **Common issue:** LIBERO defaults to Python 3.8, but the syntax updates between 3.8 and 3.10 are substantial. **We verified that using Python 3.10 avoids many issues**. 


Afterwards, inside the `LIBERO` environment, install the following dependencies:  

```bash
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4 mujoco==3.2.3
```

---

## 🚀 2. Evaluation Workflow

The evaluation should be run **from the repository root** using **two separate terminals**, one for each environment:  

- **starVLA environment**: runs the inference server.  
- **LIBERO environment**: runs the simulation.  

### Step 1. Start the server (starVLA environment)

In the first terminal, activate the `starVLA` conda environment and run:  

```bash
CKPT=/path/to/checkpoint.pt bash examples/LIBERO/eval_files/run_policy_server.sh
```

Pass the checkpoint through `CKPT`; the script no longer embeds a historical default.


---

### Step 2. Start the simulation (LIBERO environment)

In the second terminal, activate the `LIBERO` conda environment and run:  

```bash
CKPT=/path/to/checkpoint.pt HOST=127.0.0.1 \
bash examples/LIBERO/eval_files/eval_libero.sh
```
Use the same `CKPT` as the server so results and normalization metadata are written under the correct run.

Also ensure the environment variables at the top of `eval_libero.sh` are correctly set.

Finally, each result will also save a video for visualization, as shown below:

![Example](example.gif)

---


# 🚀 LIBERO Training

## 📦 Step 0: Download the training dataset
Download the datasets to the playground/Datasets/LEROBOT_LIBERO_DATA directory:
- [LIBERO-spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot)
- [LIBERO-object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot)
- [LIBERO-goal](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot)
- [LIBERO-10](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)

And move `modality.json` to each `$LEROBOT_LIBERO_DATA/subset/meta/modality.json`.

You could quickly prepare these by running:
```bash
# Set DEST to the directory where you want to store the data
export DEST=/path/to/your/data/directory
bash examples/LIBERO/data_preparation.sh
```


## 🚀 Step1: Start Training

Most of the required training files have been organized in [train_files](train_files).  

Please run the following command to start training:

```bash
bash examples/LIBERO/train_files/run_libero_train.sh
```
⚠️ **Note:** Please ensure that you specify the correct path in `examples/LIBERO/train_files/run_libero_train.sh`
