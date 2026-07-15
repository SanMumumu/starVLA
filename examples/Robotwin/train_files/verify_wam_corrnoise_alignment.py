#!/usr/bin/env python3
"""Prove that the four WAM YAMLs are FastWAM-corrnoise module ablations."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
BASELINE_PATH = HERE / "starvla_qwengroot_robotwin_fastwam_corrnoise.yaml"
WAM_NAMES = (
    "robotwin_wam_warmup_rand.yaml",
    "robotwin_wam_warmup_clean.yaml",
    "robotwin_wam_gate_rand2clean.yaml",
    "robotwin_wam_gate_clean2clean.yaml",
)
DINO_LOCAL_DIR = "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/DINO/"
CLEAN_DATA_ROOT = "/horizon-bucket/robot_lab/users/sen.wang-labs/robotwin2.0-fastwam-clean-only"


def _job_dir() -> Path:
    """Resolve submission YAMLs in the real AIDI package layout.

    Cluster source: /home/users/sen02.wang/workspace/starvla_dev/{RBT,starVLA}
    Packaged as:    /running_package/starvla_dev/{RBT,starVLA}
    """

    candidates = []
    if os.environ.get("ROBOTWIN_JOB_DIR"):
        candidates.append(Path(os.environ["ROBOTWIN_JOB_DIR"]))
    candidates.extend(
        [
            REPO_ROOT.parent / "RBT",
            REPO_ROOT / "RBT",
            REPO_ROOT / "执行脚本" / "RBT",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find RoboTwin job directory; checked: {[str(path) for path in candidates]}")


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _flatten(value, prefix="") -> dict:
    if not isinstance(value, dict):
        return {prefix: value}
    out = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        out.update(_flatten(child, path))
    return out


def _diff(left: dict, right: dict) -> list[str]:
    lflat, rflat = _flatten(left), _flatten(right)
    return sorted(key for key in lflat.keys() | rflat.keys() if lflat.get(key) != rflat.get(key))


def main(*, check_files: bool = False, target: str | None = None) -> None:
    baseline = _load(BASELINE_PATH)
    configs = {name: _load(HERE / name) for name in WAM_NAMES}
    job_dir = _job_dir()
    errors = []
    baseline_cholesky = str(Path(baseline["run_root_dir"]) / baseline["run_id"] / "action_correlation_cholesky.npy")
    clean_warmup = configs["robotwin_wam_warmup_clean.yaml"]
    clean_warmup_cholesky = str(
        Path(clean_warmup["run_root_dir"])
        / clean_warmup["run_id"]
        / "action_correlation_cholesky.npy"
    )

    for name, config in configs.items():
        clean = "warmup_clean" in name or "clean2clean" in name
        gate = "gate" in name

        for key in ("seed", "wandb_entity", "wandb_project", "is_debug", "version_id"):
            if config.get(key) != baseline.get(key):
                errors.append(f"{name}: top-level {key} differs from baseline")

        framework = config["framework"]
        if framework.get("name") != baseline["framework"].get("name"):
            errors.append(f"{name}: framework.name differs from baseline")
        if framework.get("qwenvl") != baseline["framework"].get("qwenvl"):
            errors.append(f"{name}: framework.qwenvl differs from baseline")

        action = deepcopy(framework["action_model"])
        source = action.pop("correlation_cholesky_path", None)
        action_diff = _diff(action, baseline["framework"]["action_model"])
        if action_diff:
            errors.append(f"{name}: action_model differs from baseline at {action_diff}")
        if clean and not gate:
            if source is not None:
                errors.append(
                    f"{name}: clean warmup must estimate its own Cholesky, got external source {source!r}"
                )
        else:
            expected_source = clean_warmup_cholesky if clean else baseline_cholesky
            if source != expected_source:
                errors.append(f"{name}: correlation source expected {expected_source}, got {source!r}")
        if "flow_matching_steps" in framework["action_model"]:
            errors.append(f"{name}: flow_matching_steps must not replace baseline repeated_diffusion_steps")
        if "n_action_query" in framework["action_model"]:
            errors.append(f"{name}: effective n_action_query must inherit action_horizon=32")

        data = deepcopy(config["datasets"]["vla_data"])
        baseline_data = deepcopy(baseline["datasets"]["vla_data"])
        wam_targets = data.pop("fastwam_wam_targets", None)
        wam_target = data.pop("fastwam_wam_target", None)
        future_stride = data.pop("fastwam_future_stride", None)
        domain = data.pop("fastwam_domain", "all")
        expected_domain_count = data.pop("fastwam_expected_domain_episodes", None)
        expected_episode_count = data.pop("fastwam_expected_episodes", None)
        expected_frame_count = data.pop("fastwam_expected_frames", None)
        data_root = data.pop("data_root_dir", None)
        stats_path = data.pop("fastwam_dataset_stats_path", None)
        include_state = data.pop("include_state", None)
        baseline_root = baseline_data.pop("data_root_dir", None)
        baseline_stats_path = baseline_data.pop("fastwam_dataset_stats_path", None)
        baseline_data.pop("include_state", None)
        data_diff = _diff(data, baseline_data)
        if data_diff:
            errors.append(f"{name}: shared dataset config differs from baseline at {data_diff}")
        if wam_targets is not True or wam_target != "composite" or future_stride != 32:
            errors.append(
                f"{name}: requires fastwam_wam_targets=true, fastwam_wam_target=composite and future stride 32"
            )
        if clean:
            if domain != "clean" or expected_domain_count != 2500:
                errors.append(f"{name}: clean subset must be provenance-filtered to exactly 2500 episodes")
            if data_root != CLEAN_DATA_ROOT or stats_path != f"{CLEAN_DATA_ROOT}/dataset_stats.json":
                errors.append(f"{name}: clean setting must use the exported clean-only dataset at {CLEAN_DATA_ROOT}")
            if expected_episode_count != 2500 or expected_frame_count != 0:
                errors.append(
                    f"{name}: clean preflight must expect 2500 episodes and derive its frame count from metadata"
                )
            if include_state is not True:
                errors.append(f"{name}: clean setting must use include_state=true")
        else:
            if domain != "all" or expected_domain_count is not None:
                errors.append(f"{name}: rand/full setting must use the complete FastWAM release")
            if data_root != baseline_root or stats_path != baseline_stats_path:
                errors.append(f"{name}: rand/full setting must reuse the corrnoise baseline dataset and statistics")
            if expected_episode_count is not None or expected_frame_count is not None:
                errors.append(f"{name}: rand/full setting must retain the full-release preflight defaults")
            if include_state is not False:
                errors.append(f"{name}: rand/full setting must use include_state=false")

        trainer = deepcopy(config["trainer"])
        pretrained = trainer.pop("pretrained_checkpoint", None)
        is_resume = trainer.pop("is_resume", None)
        max_train_steps = int(trainer.pop("max_train_steps", 0))
        trainer.pop("log_grad_norms", None)
        trainer.pop("deepspeed_skip_unused_param_anchors", None)
        baseline_trainer = deepcopy(baseline["trainer"])
        baseline_max_train_steps = int(baseline_trainer.pop("max_train_steps", 0))
        trainer_diff = _diff(trainer, baseline_trainer)
        if trainer_diff:
            errors.append(f"{name}: shared trainer hyperparameters differ from baseline at {trainer_diff}")
        expected_steps = 20000 if gate else baseline_max_train_steps
        if max_train_steps != expected_steps:
            errors.append(
                f"{name}: max_train_steps expected {expected_steps} "
                f"({'80k warmup + 20k gate = 100k' if gate else 'stage-1 warmup'}), got {max_train_steps}"
            )
        if is_resume is not False:
            errors.append(f"{name}: is_resume must be false")
        if gate:
            if not str(config.get("run_id", "")).endswith("_20k"):
                errors.append(f"{name}: 20k gate run_id must end with '_20k'")
            warmup = "clean" if clean else "rand"
            wanted = (
                "/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_robotwin/"
                f"robotwin_wam_m5_fastwam_warmup_{warmup}_h32_jitx_corrnoise/final_model/pytorch_model.pt"
            )
            if pretrained != wanted:
                errors.append(f"{name}: pretrained_checkpoint expected {wanted!r}, got {pretrained!r}")
        elif pretrained is not None:
            errors.append(f"{name}: warmup must start without a pretrained WAM checkpoint")

        if check_files and (target is None or name == target):
            if source is not None:
                matrix_path = Path(source)
                if not matrix_path.is_file():
                    errors.append(f"{name}: missing configured Cholesky {matrix_path}")
                else:
                    matrix = np.load(matrix_path, allow_pickle=False)
                    if matrix.shape != (448, 448) or not np.isfinite(matrix).all():
                        errors.append(f"{name}: invalid Cholesky shape/content at {matrix_path}")
            dino_cfg = framework.get("dino", {})
            if dino_cfg.get("loader") in {"auto", "torchhub"}:
                repo = Path(str(dino_cfg.get("repo_or_dir", "")))
                if not repo.is_dir():
                    errors.append(f"{name}: missing local DINOv3 torch.hub repo {repo}")
            teacher_weights = (
                dino_cfg.get("weights") or os.environ.get("DINOV3_WEIGHTS") or os.environ.get("DINOV3_VITL16_WEIGHTS")
            )
            if teacher_weights and not Path(str(teacher_weights)).exists():
                errors.append(f"{name}: configured DINO teacher weights do not exist: {teacher_weights}")
            elif dino_cfg.get("loader") == "hf" and not (Path(str(teacher_weights)) / "config.json").is_file():
                errors.append(f"{name}: local DINO HF snapshot is missing config.json: {teacher_weights}")
            elif dino_cfg.get("loader") == "hf":
                snapshot = Path(str(teacher_weights))
                hf_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
                if hf_config.get("hidden_size") != 1024:
                    errors.append(
                        f"{name}: local DINO snapshot must be ViT-L hidden_size=1024, "
                        f"got {hf_config.get('hidden_size')!r}"
                    )
                weight_files = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("pytorch_model*.bin"))
                if not any(path.stat().st_size > 100 * 1024 * 1024 for path in weight_files):
                    errors.append(
                        f"{name}: local DINO snapshot has no weight file larger than 100 MiB; "
                        "model.safetensors may be missing or only a Git-LFS pointer"
                    )
            if gate and pretrained and not Path(pretrained).is_file():
                errors.append(f"{name}: warmup checkpoint does not exist yet: {pretrained}")

        wam = framework.get("wam", {})
        guidance = wam.get("guidance", {})
        expected_bridge = "predicted" if gate else "oracle"
        expected_method = {
            "enabled": True,
            "mode": "dual_xattn",
            "signal": "z_pred",
            "prompt_mode": "dual_query",
            "bridge_source": expected_bridge,
            "detach_world": True,
            "gate_init": 0.0,
        }
        if not wam.get("enabled") or wam.get("world_target") != "absolute":
            errors.append(f"{name}: WAM/absolute future target must be enabled")
        if wam.get("action_gradient_checkpointing") is not True:
            errors.append(f"{name}: action_gradient_checkpointing must stay enabled for baseline bs=16")
        for key, expected in expected_method.items():
            if guidance.get(key) != expected:
                errors.append(f"{name}: wam.guidance.{key} expected {expected!r}, got {guidance.get(key)!r}")
        if framework.get("tasks", {}).get("weights") != {"policy": 1.0, "passive": 1.0}:
            errors.append(f"{name}: tasks must be exactly policy+passive")

        dino = framework.get("dino", {})
        if dino.get("model_size") != "large" or dino.get("embed_dim") != 1024:
            errors.append(f"{name}: future target must use DINO-L/1024")
        if dino.get("hf_model_id") != DINO_LOCAL_DIR:
            errors.append(f"{name}: dino.hf_model_id must use local snapshot {DINO_LOCAL_DIR!r}")
        if dino.get("weights") != DINO_LOCAL_DIR or dino.get("loader") != "hf":
            errors.append(f"{name}: DINO-L must use loader=hf with local weights {DINO_LOCAL_DIR!r}")
        if dino.get("future_view_keys") != ["video.robotwin_composite"]:
            errors.append(f"{name}: DINO target must be the single three-view FastWAM composite")
        if dino.get("image_size") != [384, 320] or dino.get("patch_size") != 16:
            errors.append(f"{name}: DINO must preserve the 384x320 composite with patch_size=16")
        visual = framework.get("visual_model", {})
        if visual.get("max_target_tokens") != 480 or visual.get("max_seq_len") != 480:
            errors.append(f"{name}: visual head must allocate exactly one 24x20=480-token composite target grid")
        if "n_query" in visual or "max_image_queries" in visual:
            errors.append(f"{name}: unused JointFlow query-capacity fields must not be carried into WAM")
        forbidden_data = {
            "dino_target_latents",
            "dino_feature_dir",
            "require_precomputed_dino_targets",
            "dino_target_view_keys",
            "online_dino",
        }
        present = forbidden_data & set(config["datasets"]["vla_data"])
        if present:
            errors.append(f"{name}: stale standard-RoboTwin latent fields remain: {sorted(present)}")

        job = _load(job_dir / name)
        command = job.get("REQUIRED", {}).get("RUN_SCRIPTS")
        wanted_command = f"${{WORKING_PATH}}/run_aidi_rbtw.sh examples/Robotwin/train_files/{name}"
        if command != wanted_command:
            errors.append(f"{name}: AIDI job command expected {wanted_command!r}, got {command!r}")
        if job.get("REQUIRED", {}).get("WORKER_MIN_NUM") != 6 or job.get("REQUIRED", {}).get("GPU_PER_WORKER") != 8:
            errors.append(f"{name}: AIDI scale must remain 6x8")
        remark = str(job.get("OPTIONAL", {}).get("REMARK", "")).lower()
        for token in ("h32", "composite", "bs16", "proprio" if clean else "no-state"):
            if token not in remark:
                errors.append(f"{name}: AIDI remark missing {token!r}")
        if gate and "20k" not in remark:
            errors.append(f"{name}: AIDI remark must declare the 20k gate budget")

    model_source = (REPO_ROOT / "starVLA/model/framework/VLM4A/QwenGR00T.py").read_text(encoding="utf-8")
    for symbol in (
        "def _wam_action_state_and_mask",
        "def _wam_action_loss",
        '"action_is_pad"',
        "repeated_diffusion_steps",
        "state=state.to(head_dtype)",
        "def _wam_visual_loss",
    ):
        if symbol not in model_source:
            errors.append(f"WAM implementation is missing strict-baseline behavior: {symbol}")
    dit_source = (REPO_ROOT / "starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py").read_text(
        encoding="utf-8"
    )
    if "self.gradient_checkpointing" not in dit_source or "checkpoint(block" not in dit_source:
        errors.append("WAM action-DiT activation checkpointing implementation is missing")

    dataset_source = (REPO_ROOT / "starVLA/dataloader/fastwam_robotwin_dataset.py").read_text(encoding="utf-8")
    composite_contract_symbols = (
        '_COMPOSITE_VIEW_KEY = "video.robotwin_composite"',
        'for video_key in self.modality_keys["video"]',
        'future_views = [data[key][1] for key in self.modality_keys["video"]]',
        "future_composite = build_robotwin_composite(future_views)",
        '"image_1": [future_composite]',
        '"dino_target_view_keys": [_COMPOSITE_VIEW_KEY]',
    )
    for symbol in composite_contract_symbols:
        if symbol not in dataset_source:
            errors.append(f"FastWAM dataset is missing future three-view-composite contract: {symbol}")
    if 'indices["video.cam_high"]' in dataset_source or "future_main =" in dataset_source:
        errors.append("FastWAM dataset still contains the old future-main-view-only target path")

    if errors:
        raise SystemExit("WAM/FastWAM strict-ablation audit failed:\n- " + "\n- ".join(errors))
    print(
        "WAM/FastWAM contract PASS: warmup=80k and gate=20k (two-stage total=100k); "
        "shared action/data/trainer fields equal the corrnoise baseline; "
        "rand reuses the full-release correlation, clean estimates/reuses only its own correlation, and the "
        "declared factors are future-composite prediction/gating plus clean=true/rand=false proprio conditioning."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--target", choices=WAM_NAMES)
    args = parser.parse_args()
    main(check_files=args.check_files, target=args.target)
