"""Focused regression checks for the checkpoint-compatible FastWAM ABI."""

# ruff: noqa: E402 - insert repository root before importing the local package

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK_DIR_NAME = "\u6267\u884c\u811a\u672c"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from omegaconf import OmegaConf

from starVLA.dataloader.fastwam_robotwin_dataset import (
    FastWAMEpochSampler,
    FastWAMRobotWinDataset,
    build_fastwam_dataset_metadata,
    build_robotwin_composite,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.lerobot_datasets import get_vla_dataset


def _write_fixture(root: Path, episode_lengths=(40, 41, 42, 43)) -> None:
    (root / "meta").mkdir()
    features = {
        "observation.state": {"dtype": "float32", "shape": [14]},
        "action": {"dtype": "float32", "shape": [14]},
    }
    camera_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    for name in camera_names:
        original = f"observation.images.{name}"
        features[original] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.fps": 50,
                "video.channels": 3,
            },
        }

    total_frames = sum(episode_lengths)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": len(episode_lengths),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(camera_names) * len(episode_lengths),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 50,
        "splits": {"train": f"0:{len(episode_lengths)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    episodes = []
    for episode, length in enumerate(episode_lengths):
        domain = "Clean" if episode < len(episode_lengths) // 2 else "Randomized"
        episodes.append(
            {
                "episode_index": episode,
                "length": length,
                "raw_file_name": f"RoboTwin/{domain}/click_bell/episode_{episode}.pkl",
            }
        )
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes), encoding="utf-8"
    )
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "click the bell"}) + "\n", encoding="utf-8"
    )

    stats = {}
    for name in ("min", "max", "mean", "std", "q01", "q99"):
        if name == "std":
            values = (1.0 + np.arange(14) / 10).tolist()
        elif name == "mean":
            values = (np.arange(14) / 10).tolist()
        elif name == "min":
            values = np.arange(14).tolist()
        elif name == "max":
            values = (np.arange(14) + 20).tolist()
        else:
            values = np.zeros(14).tolist()
        stats[f"global_{name}"] = values
    (root / "dataset_stats.json").write_text(
        json.dumps({"state": {"default": stats}, "action": {"default": stats}}), encoding="utf-8"
    )

    frame_cursor = 0
    source = np.arange(14, dtype=np.float32) / 100
    for episode, length in enumerate(episode_lengths):
        parquet = root / f"data/chunk-000/episode_{episode:06d}.parquet"
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "observation.state": [source.copy() for _ in range(length)],
                "action": [source.copy() for _ in range(length)],
                "task_index": np.zeros(length, dtype=np.int64),
                "timestamp": np.arange(length, dtype=np.float32) / 50,
                "episode_index": np.full(length, episode, dtype=np.int64),
                "frame_index": np.arange(length, dtype=np.int64),
                "index": np.arange(frame_cursor, frame_cursor + length, dtype=np.int64),
            }
        ).to_parquet(parquet)
        frame_cursor += length
        for camera in camera_names:
            video = root / f"videos/chunk-000/observation.images.{camera}/episode_{episode:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.touch()


def _config(root: Path, **overrides):
    data = {
        "data_root_dir": str(root),
        "data_mix": "robotwin_fastwam",
        "lerobot_version": "v2.0",
        "fastwam_dataset_stats_path": str(root / "dataset_stats.json"),
        "fastwam_expected_fps": 50,
        "fastwam_val_fraction": 0.0,
        "fastwam_split": "all",
        "fastwam_split_seed": 42,
        "fastwam_direct_frame_sampling": True,
        "include_state": True,
        "action_mode": "abs",
        "video_backend": "pyav",
    }
    data.update(overrides)
    return OmegaConf.create(data)


def _mock_video(dataset: FastWAMRobotWinDataset):
    colors = {"video.cam_high": 20, "video.cam_left_wrist": 80, "video.cam_right_wrist": 140}

    def get_video(_trajectory_id, key, _base_index):
        count = len(dataset.delta_indices[key])
        return np.stack([np.full((480, 640, 3), colors[key] + index, dtype=np.uint8) for index in range(count)])

    dataset.get_video = get_video


def test_fastwam_metadata_and_composite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)
        metadata = build_fastwam_dataset_metadata(root, EmbodimentTag.NEW_EMBODIMENT)
        assert metadata.statistics.action["left_joints"].mean.shape == (6,)
        assert metadata.statistics.action["right_gripper"].mean.shape == (1,)

        images = [np.full((480, 640, 3), value, dtype=np.uint8) for value in (20, 80, 140)]
        composite = np.asarray(build_robotwin_composite(images))
        assert composite.shape == (384, 320, 3)
        assert int(composite[10, 10, 0]) == 20
        assert int(composite[300, 10, 0]) == 80
        assert int(composite[300, 250, 0]) == 140


def test_fastwam_checkpoint_sample_and_direct_sampler() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)
        dataset = get_vla_dataset(_config(root), balance_dataset_weights=False, balance_trajectory_weights=False)
        assert isinstance(dataset, FastWAMRobotWinDataset)
        assert len(dataset) == 166
        _mock_video(dataset)

        raw = dataset.get_step_data(0, 35)
        sample = dataset._pack_sample(dataset.transforms(raw))
        assert sample["action"].shape == (32, 14)
        assert sample["action"].dtype == np.float32
        assert sample["state"].shape == (1, 14)
        assert sample["state"].dtype == np.float32
        assert sample["action_is_pad"].shape == (32,)
        assert int(sample["action_is_pad"].sum()) == 27
        assert len(sample["image"]) == 1
        assert sample["image"][0].size == (320, 384)
        assert sample["lang"] == "click the bell"

        source = np.arange(14, dtype=np.float32) / 100
        normalized = (source - np.arange(14, dtype=np.float32) / 10) / (
            1.0 + np.arange(14, dtype=np.float32) / 10 + 1e-8
        )
        np.testing.assert_allclose(sample["action"][0], normalized, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(sample["state"][0], normalized, rtol=2e-5, atol=2e-5)

        sampler = FastWAMEpochSampler(dataset, seed=42)
        order0 = list(sampler)
        assert sorted(order0) == list(range(len(dataset)))
        sampler.set_epoch(1)
        assert order0 != list(sampler)

        stats_path = root / "saved_dataset_statistics.json"
        dataset.save_dataset_statistics(stats_path)
        saved = json.loads(stats_path.read_text(encoding="utf-8"))
        tag_stats = saved[next(iter(saved))]
        assert tag_stats["action"]["min"] == list(range(14))
        assert tag_stats["state"]["min"] == list(range(14))
        assert tag_stats["action"]["mask"] == [True] * 14

        run_dir = root / "fake_run"
        checkpoint = run_dir / "checkpoints/steps_1_pytorch_model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        source_config = REPO_ROOT / "examples/Robotwin/train_files/starvla_qwengroot_robotwin_fastwam.yaml"
        (run_dir / "config.yaml").write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "dataset_statistics.json").write_text(stats_path.read_text(encoding="utf-8"), encoding="utf-8")
        from deployment.model_server.policy_norm_processor import PolicyNormProcessor

        restored = PolicyNormProcessor(str(checkpoint)).unapply_actions(sample["action"])
        np.testing.assert_allclose(restored[0], source, rtol=2e-5, atol=2e-5)
        from deployment.model_server.policy_wrapper import PolicyServerWrapper

        wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
        wrapper._default_unnorm_key = "new_embodiment"
        wrapper._build_state_normalizer(yaml.safe_load(source_config.read_text(encoding="utf-8")), saved)
        wrapper._expects_state = True
        prepared = wrapper._prepare_examples([{"state": source.copy(), "image": sample["image"]}])[0]
        np.testing.assert_allclose(prepared["state"][0], normalized, rtol=2e-5, atol=2e-5)
        wrapper._expects_state = False
        assert "state" not in wrapper._prepare_examples([{"state": source.copy(), "image": sample["image"]}])[0]
        from examples.Robotwin.eval_files.verify_fastwam_checkpoint_contract import verify

        assert verify(checkpoint, replan_steps=24)["replan"] == 24


def test_fastwam_split_domain_and_wam_composite_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)
        clean = get_vla_dataset(
            _config(
                root,
                fastwam_domain="clean",
                fastwam_expected_domain_episodes=2,
                fastwam_wam_targets=True,
                fastwam_wam_target="composite",
                fastwam_future_stride=32,
            )
        )
        assert clean.trajectory_ids.tolist() == [0, 1]
        _mock_video(clean)
        sample = clean._pack_sample(clean.transforms(clean.get_step_data(0, 0)))
        assert len(sample["image_0"]) == 1
        assert sample["image_0"][0].size == (320, 384)
        assert len(sample["image_1"]) == 1
        assert sample["image_1"][0].size == (320, 384)
        future_composite = np.asarray(sample["image_1"][0])
        assert int(future_composite[10, 10, 0]) == 21
        assert int(future_composite[300, 10, 0]) == 81
        assert int(future_composite[300, 250, 0]) == 141
        assert sample["dino_target_view_keys"] == ["video.robotwin_composite"]
        assert sample["future_valid"] == 1
        assert len(clean.delta_indices["video.cam_high"]) == 2
        assert len(clean.delta_indices["video.cam_left_wrist"]) == 2
        assert len(clean.delta_indices["video.cam_right_wrist"]) == 2

        split = get_vla_dataset(_config(root, fastwam_val_fraction=0.25, fastwam_split="train"))
        order = list(range(4))
        np.random.default_rng(42).shuffle(order)
        assert split.trajectory_ids.tolist() == order[:3]

        no_state = get_vla_dataset(_config(root, include_state=False))
        _mock_video(no_state)
        no_state_sample = no_state._pack_sample(no_state.transforms(no_state.get_step_data(0, 0)))
        assert "state" not in no_state_sample


def test_fastwam_action_world_coflow_h16_single_future_contract() -> None:
    """The closed-loop recipe loads exactly t/t+16 and a 16-step action chunk."""

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)
        dataset = get_vla_dataset(
            _config(
                root,
                data_mix="robotwin_fastwam_h16",
                fastwam_action_world_coflow_targets=True,
            )
        )
        _mock_video(dataset)
        assert dataset.delta_indices["video.cam_high"].tolist() == [0, 16]
        sample = dataset._pack_sample(dataset.transforms(dataset.get_step_data(0, 0)))
        assert sample["action"].shape == (16, 14)
        assert sample["action_is_pad"].shape == (16,)
        assert sample["future_valid_16"] == 1
        assert "coflow_future_strides" not in sample
        assert sample["image_16"][0].size == (320, 384)

        with pytest.raises(ValueError, match="was removed"):
            get_vla_dataset(
                _config(
                    root,
                    data_mix="robotwin_fastwam_h16",
                    fastwam_action_world_coflow_targets=True,
                    fastwam_coflow_future_strides=[16],
                )
            )
        with pytest.raises(ValueError, match="enable exactly one"):
            get_vla_dataset(
                _config(
                    root,
                    data_mix="robotwin_fastwam_h16",
                    fastwam_wam_targets=True,
                    fastwam_action_world_coflow_targets=True,
                )
            )


def test_fastwam_train_infer_order_and_contract() -> None:
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from examples.Robotwin.eval_files.model2robotwin_fastwam_interface import FastWAMRobotWinModelClient
    from examples.Robotwin.eval_files.model2robotwin_interface import resolve_replan_steps

    config_path = REPO_ROOT / "examples/Robotwin/train_files/starvla_qwengroot_robotwin_fastwam.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert PolicyServerWrapper._config_expects_state(config)
    assert resolve_replan_steps(24, 32) == 24

    client = FastWAMRobotWinModelClient.__new__(FastWAMRobotWinModelClient)
    images = [np.full((480, 640, 3), value, dtype=np.uint8) for value in (20, 80, 140)]
    prepared = client._prepare_images(images)
    assert len(prepared) == 1 and prepared[0].shape == (384, 320, 3)
    state = np.arange(14, dtype=np.float32)
    client.expects_state = True
    np.testing.assert_array_equal(client._prepare_state_for_server(state), state.reshape(1, 14))
    client.expects_state = False
    assert client._prepare_state_for_server(state) is None
    np.testing.assert_array_equal(client._prepare_action_for_env(state), state)
    client.coflow_inference_mode = "policy"
    client.coflow_inference_horizon = 16
    client.coflow_inference_seed = 7
    client._coflow_query_index = 0
    assert client._extra_inference_request_kwargs() == {
        "coflow_inference_mode": "policy",
        "coflow_inference_horizon": 16,
        "coflow_inference_seed": 7,
    }
    assert client._coflow_query_index == 1


def test_policy_server_metadata_exposes_loaded_gate_ft_contract() -> None:
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    class GateFramework:
        wam_enabled = True
        wam_guidance = {
            "enabled": True,
            "mode": "dual_xattn",
            "action_world_bypass": False,
            "bridge_source": "predicted",
            "world_eval_mode": "correct",
            "baseline_action_context": True,
        }

        @staticmethod
        def _wam_world_gate_metrics():
            return {
                "world_gate_openness": torch.tensor(0.2),
                "world_gate_signed_mean": torch.tensor(-0.05),
                "world_gate_max_openness": torch.tensor(0.7),
            }

    contract = {
        "framework": {"wam": {"enabled": True, "guidance": {"enabled": True}}},
        "trainer": {
            "wam_two_stage_phase": "gate_ft",
            "wam_two_stage_recipe": "baseline_preserving_v3",
        },
    }
    metadata = PolicyServerWrapper._build_wam_runtime_metadata(GateFramework(), contract)
    assert metadata["wam_two_stage_phase"] == "gate_ft"
    assert metadata["wam_two_stage_recipe"] == "baseline_preserving_v3"
    assert metadata["wam_action_world_bypass"] is False
    assert metadata["wam_bridge_source"] == "predicted"
    assert metadata["wam_baseline_action_context"] is True
    assert metadata["wam_gate_openness"] == pytest.approx(0.2)
    assert metadata["wam_gate_signed_mean"] == pytest.approx(-0.05)
    assert metadata["wam_gate_max_openness"] == pytest.approx(0.7)


def test_policy_server_metadata_accepts_physical_no_world2action_baseline() -> None:
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    framework = SimpleNamespace(
        wam_enabled=True,
        wam_guidance={
            "enabled": True,
            "mode": "dual_xattn",
            "action_world_bypass": True,
            "world_to_action_enabled": False,
            "bridge_source": "predicted",
            "world_eval_mode": "correct",
            "baseline_action_context": True,
        },
    )
    contract = {
        "framework": {"wam": {"enabled": True, "guidance": {"enabled": True}}},
        "trainer": {
            "wam_two_stage_phase": "predictor_warmup",
            "wam_two_stage_recipe": "baseline_preserving_v3",
        },
    }
    metadata = PolicyServerWrapper._build_wam_runtime_metadata(framework, contract)
    assert metadata["wam_world_to_action_enabled"] is False
    assert metadata["wam_action_world_bypass"] is True
    assert metadata["wam_gate_openness"] is None


def test_fastwam_real_cluster_config_snapshots() -> None:
    """The accessed config may omit state while config.full.yaml preserves the ABI."""

    from deployment.model_server.checkpoint_contract import (
        load_checkpoint_contract_config,
        resolve_config_expects_state,
    )
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from examples.Robotwin.eval_files.verify_fastwam_checkpoint_contract import verify

    snapshot_dir = REPO_ROOT / RUNBOOK_DIR_NAME / "\u96c6\u7fa4FastWAM IID checkpoint yaml"
    accessed_source = snapshot_dir / "config.yaml"
    full_source = snapshot_dir / "config.full.yaml"
    accessed = yaml.safe_load(accessed_source.read_text(encoding="utf-8"))
    full = yaml.safe_load(full_source.read_text(encoding="utf-8"))
    assert "include_state" not in accessed["datasets"]["vla_data"]
    assert full["datasets"]["vla_data"]["include_state"] is True
    explicit_false = yaml.safe_load(accessed_source.read_text(encoding="utf-8"))
    explicit_false["datasets"]["vla_data"]["include_state"] = False
    assert resolve_config_expects_state(explicit_false) == (
        False,
        "datasets.vla_data.include_state=False",
    )

    class _RuntimeFramework:
        def __init__(self):
            self.config = OmegaConf.create({"datasets": {"vla_data": {}}})

        def _uses_action_state(self):
            return bool(self.config.datasets.vla_data.get("include_state", False))

    runtime_framework = _RuntimeFramework()
    wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
    wrapper._expects_state = True
    wrapper._state_contract_source = "datasets.vla_data.include_state=True"
    wrapper._contract_cfg_path = "config.full.yaml"
    wrapper._sync_framework_state_contract(runtime_framework)
    assert runtime_framework._uses_action_state() is True
    wrapper._expects_state = False
    wrapper._state_contract_source = "datasets.vla_data.include_state=False"
    wrapper._sync_framework_state_contract(runtime_framework)
    assert runtime_framework._uses_action_state() is False

    values = [0.0] * 14
    modality_stats = {
        "min": values,
        "max": values,
        "mean": values,
        "std": [1.0] * 14,
        "q01": values,
        "q99": values,
        "mask": [True] * 14,
    }
    checkpoint_stats = {
        "new_embodiment": {
            "state": modality_stats,
            "action": modality_stats,
        }
    }

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        checkpoint = run_dir / "checkpoints/steps_80000_pytorch_model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        (run_dir / "config.yaml").write_text(accessed_source.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "config.full.yaml").write_text(full_source.read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "dataset_statistics.json").write_text(json.dumps(checkpoint_stats), encoding="utf-8")

        contract_cfg, contract_path = load_checkpoint_contract_config(checkpoint, accessed_config=accessed)
        assert contract_path.name == "config.full.yaml"
        assert resolve_config_expects_state(contract_cfg) == (
            True,
            "datasets.vla_data.include_state=True",
        )
        summary = verify(checkpoint, replan_steps=24)
        assert Path(summary["contract_config"]).name == "config.full.yaml"
        assert summary["expects_state"] is True

        full_no_state = yaml.safe_load(full_source.read_text(encoding="utf-8"))
        full_no_state["datasets"]["vla_data"]["include_state"] = False
        (run_dir / "config.full.yaml").write_text(yaml.safe_dump(full_no_state), encoding="utf-8")
        no_state_summary = verify(checkpoint, replan_steps=24)
        assert no_state_summary["expects_state"] is False
        assert no_state_summary["state"] == "disabled (request state is omitted)"

        # If an old run has no full snapshot and its compact config omitted the
        # switch, inference keeps the historical no-state default.
        (run_dir / "config.full.yaml").unlink()
        fallback_cfg, fallback_path = load_checkpoint_contract_config(checkpoint, accessed_config=accessed)
        expects_state, source = resolve_config_expects_state(fallback_cfg)
        assert fallback_path.name == "config.yaml"
        assert expects_state is False
        assert source == "datasets.vla_data.include_state missing; default=False"
        assert verify(checkpoint, replan_steps=24)["state_contract_source"] == source


def test_fastwam_cluster_preflight_fixture() -> None:
    from examples.Robotwin.train_files.verify_fastwam_robotwin_data import verify_fastwam_robotwin_data

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_fixture(root)
        summary = verify_fastwam_robotwin_data(
            root,
            root / "dataset_stats.json",
            expected_episodes=4,
            expected_frames=166,
            expected_tasks=1,
        )
        assert summary["fps"] == 50
        assert summary["domain_counts"] == {"clean": 2, "randomized": 2, "unknown": 0}


def test_correlated_noise_artifact_is_a_strict_train_deploy_contract() -> None:
    from unittest.mock import patch

    import deployment.model_server.policy_wrapper as wrapper_module
    from starVLA.dataloader.action_correlation import validate_action_correlation_cholesky

    expected_size = 4
    valid = np.eye(expected_size, dtype=np.float32)
    np.testing.assert_array_equal(
        validate_action_correlation_cholesky(valid, expected_size=expected_size),
        valid,
    )
    invalid_upper = valid.copy()
    invalid_upper[0, 1] = 0.1
    with np.testing.assert_raises_regex(ValueError, "lower triangular"):
        validate_action_correlation_cholesky(invalid_upper, expected_size=expected_size)
    invalid_diagonal = valid.copy()
    invalid_diagonal[-1, -1] = 0.0
    with np.testing.assert_raises_regex(ValueError, "positive diagonal"):
        validate_action_correlation_cholesky(invalid_diagonal, expected_size=expected_size)

    class _Framework:
        def __init__(self):
            self.injected = None

        def set_action_correlation(self, value):
            self.injected = np.asarray(value)

        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

    config = {
        "framework": {
            "action_model": {
                "use_correlated_noise": True,
                "action_horizon": 2,
                "action_dim": 2,
            }
        },
        "datasets": {"vla_data": {"include_state": False}},
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        checkpoint = run_dir / "checkpoints" / "steps_1_pytorch_model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        framework = _Framework()

        def construct():
            with (
                patch.object(wrapper_module.baseframework, "from_pretrained", return_value=framework),
                patch.object(wrapper_module, "read_mode_config", return_value=(config, {})),
                patch.object(
                    wrapper_module,
                    "load_checkpoint_contract_config",
                    return_value=(config, run_dir / "config.full.yaml"),
                ),
            ):
                return wrapper_module.PolicyServerWrapper(str(checkpoint), device="cpu")

        with np.testing.assert_raises_regex(FileNotFoundError, "required run artifact"):
            construct()

        np.save(run_dir / "action_correlation_cholesky.npy", valid)
        construct()
        np.testing.assert_array_equal(framework.injected, valid)

        # The new joint-E2E runs explicitly use IID noise. Deployment must not
        # require or inject a Cholesky artifact when the checkpoint says false.
        config["framework"]["action_model"]["use_correlated_noise"] = False
        (run_dir / "action_correlation_cholesky.npy").unlink()
        framework.injected = None
        construct()
        assert framework.injected is None


def test_robotwin_causal_query_and_coflow_yaml_contracts() -> None:
    """The active WAM warmup is state-free, online DINO-B, and gate-ready."""

    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform

    config_dir = REPO_ROOT / "examples/Robotwin/train_files"
    cfg = yaml.safe_load(
        (config_dir / "robotwin_wam_query_warmup.yaml").read_text(encoding="utf-8")
    )
    action_cfg = cfg["framework"]["action_model"]
    dino_cfg = cfg["framework"]["dino"]
    data_cfg = cfg["datasets"]["vla_data"]
    guidance = cfg["framework"]["wam"]["guidance"]
    assert action_cfg["use_correlated_noise"] is False
    assert action_cfg["action_horizon"] == action_cfg["n_action_query"] == 32
    assert action_cfg["state_dim"] == 0
    assert data_cfg["include_state"] is False
    assert data_cfg["per_device_batch_size"] == 16
    assert cfg["trainer"]["expected_global_batch_size"] == 1024
    assert cfg["trainer"]["max_train_steps"] == 80000
    assert cfg["trainer"]["wam_two_stage_phase"] == "predictor_warmup"
    assert guidance["causal_query_suffix"] is True
    assert guidance["include_context_in_action_memory"] is False
    assert guidance["include_context_in_world_memory"] is False
    assert guidance["world_condition_on_state"] is False
    assert guidance["concat_current_dino"] is False
    assert guidance["world_to_action_enabled"] is True
    assert guidance["action_world_bypass"] is True
    assert guidance["freeze_world_to_action_in_warmup"] is True
    assert dino_cfg["model_size"] == "base"
    assert dino_cfg["weights"].endswith("/DINO-B/")
    assert dino_cfg["load_live_backbone"] is True
    assert dino_cfg["force_online"] is True

    assert not (config_dir / "robotwin_action_world_coflow.yaml").exists()
    coflow = yaml.safe_load(
        (config_dir / "robotwin_action_world_coflow_better.yaml").read_text(encoding="utf-8")
    )
    assert coflow["framework"]["name"] == "QwenActionWorldCoFlow"
    assert coflow["framework"]["enable_action_world_coflow"] is True
    assert coflow["framework"]["action_model"]["use_correlated_noise"] is False
    assert coflow["framework"]["action_model"]["action_horizon"] == 16
    coflow_model = coflow["framework"]["action_world_coflow"]
    assert "segment_boundaries" not in coflow_model
    assert "future_strides" not in coflow_model
    assert "z32_loss_weight" not in coflow_model
    coflow_data = coflow["datasets"]["vla_data"]
    assert coflow_data["data_mix"] == "robotwin_fastwam_h16"
    assert "fastwam_coflow_future_strides" not in coflow_data
    assert coflow_data["include_state"] is True
    assert coflow_data["per_device_batch_size"] == 12
    assert coflow_data["num_workers"] == 4
    assert coflow["trainer"]["gradient_accumulation_steps"] == 1
    assert coflow["trainer"]["expected_global_batch_size"] == 768

    data_config = ROBOT_TYPE_CONFIG_MAP["robotwin_fastwam"]
    state_action_transforms = [
        transform for transform in data_config.transform().transforms if isinstance(transform, StateActionTransform)
    ]
    assert len(state_action_transforms) == 2
    assert all(
        set(transform.normalization_modes.values()) == {"fastwam_zscore"}
        for transform in state_action_transforms
    )


def test_coflow_checkpoint_verifier_accepts_only_h16_single_bridge() -> None:
    from examples.Robotwin.eval_files.verify_action_world_coflow_checkpoint_contract import verify

    source = REPO_ROOT / "examples/Robotwin/train_files/robotwin_action_world_coflow_better.yaml"
    h16_config = yaml.safe_load(source.read_text(encoding="utf-8"))
    values = [0.0] * 14
    modality = {
        "min": values,
        "max": values,
        "mean": values,
        "std": [1.0] * 14,
        "q01": values,
        "q99": values,
        "mask": [True] * 14,
    }
    stats = {"new_embodiment": {"state": modality, "action": modality}}

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "coflow"
        checkpoint = run_dir / "checkpoints/steps_1_pytorch_model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        (run_dir / "config.yaml").write_text(yaml.safe_dump(h16_config), encoding="utf-8")
        (run_dir / "config.full.yaml").write_text(yaml.safe_dump(h16_config), encoding="utf-8")
        (run_dir / "dataset_statistics.json").write_text(json.dumps(stats), encoding="utf-8")

        h16 = verify(checkpoint, 16, inference_mode="policy", inference_horizon=16)
        assert h16["checkpoint_chunk"] == h16["inference_horizon"] == 16

        invalid = copy.deepcopy(h16_config)
        invalid["framework"]["action_model"]["action_horizon"] = 32
        (run_dir / "config.full.yaml").write_text(yaml.safe_dump(invalid), encoding="utf-8")
        with pytest.raises(ValueError, match="action_horizon=16"):
            verify(checkpoint, 16, inference_mode="diagonal", inference_horizon=16)


def test_robodojo_train_and_deploy_contracts() -> None:
    """RoboDojo must keep one 25 Hz state/image/action ABI end to end."""

    from starVLA.dataloader.gr00t_lerobot.registry import (
        DATASET_NAMED_MIXTURES,
        ROBOT_TYPE_CONFIG_MAP,
    )
    from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform

    train_dir = REPO_ROOT / "examples/RoboDojo/train_files"
    configs = {
        name: yaml.safe_load((train_dir / name).read_text(encoding="utf-8"))
        for name in (
            "starvla_qwengroot_robodojo_baseline.yaml",
            "starvla_qwengroot_robodojo_wam_warmup.yaml",
            "starvla_qwengroot_robodojo_wam_gate.yaml",
        )
    }
    source_views = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    for cfg in configs.values():
        action_cfg = cfg["framework"]["action_model"]
        data_cfg = cfg["datasets"]["vla_data"]
        assert action_cfg["action_dim"] == action_cfg["state_dim"] == 14
        assert action_cfg["action_horizon"] == 16
        assert data_cfg["data_root_dir"] == "/horizon-bucket/robot_lab/users/sen.wang-labs/RoboDojo"
        assert data_cfg["include_state"] is True
        assert data_cfg["video_backend"] == "pyav"
        assert data_cfg["image_layout"] == "fastwam_composite"
        assert data_cfg["composite_source_view_keys"] == source_views
        assert data_cfg["composite_view_key"] == "video.fastwam_composite"
        assert data_cfg["obs_image_size"] == [320, 384]

    baseline = configs["starvla_qwengroot_robodojo_baseline.yaml"]
    warmup = configs["starvla_qwengroot_robodojo_wam_warmup.yaml"]
    gate = configs["starvla_qwengroot_robodojo_wam_gate.yaml"]
    assert baseline["datasets"]["vla_data"]["data_mix"] == "robodojo_v21"
    assert warmup["datasets"]["vla_data"]["data_mix"] == "robodojo_v21_language_optional"
    assert gate["datasets"]["vla_data"]["data_mix"] == "robodojo_v21_language_optional"
    assert baseline["datasets"]["vla_data"]["dataset_py"] == "lerobot_datasets"
    assert "correlation_cholesky_path" not in baseline["framework"]["action_model"]
    for cfg in (warmup, gate):
        assert cfg["datasets"]["vla_data"]["dataset_py"] == "jointflow"
        assert cfg["datasets"]["vla_data"]["action_horizon"] == 16
        assert cfg["datasets"]["vla_data"]["world_model"]["future_stride"] == 16
        assert cfg["datasets"]["vla_data"]["future_valid_requires_full_stride"] is True
        assert cfg["framework"]["dino"]["image_size"] == [384, 320]
        assert cfg["framework"]["dino"]["future_view_keys"] == ["video.fastwam_composite"]
        assert cfg["framework"]["visual_model"]["max_target_tokens"] == 480
        assert cfg["framework"]["dino"]["force_online"] is True
        assert cfg["framework"]["dino"]["weights"].endswith("/DINO-B/")
        assert cfg["framework"]["action_model"]["use_correlated_noise"] is False
        assert not any(
            str(key).startswith("correlation_")
            for key in cfg["framework"]["action_model"]
        )

    assert gate["trainer"]["pretrained_checkpoint"].startswith(
        f"{gate['run_root_dir']}/{warmup['run_id']}/"
    )
    assert warmup["trainer"]["wam_two_stage_phase"] == "predictor_warmup"
    assert gate["trainer"]["wam_two_stage_phase"] == "gate_ft"
    assert warmup["framework"]["wam"]["guidance"]["bridge_source"] == "predicted"
    assert warmup["framework"]["wam"]["guidance"]["oracle_ratio"] == 0.0
    assert gate["framework"]["wam"]["guidance"]["bridge_source"] == "predicted"
    assert gate["framework"]["wam"]["guidance"]["oracle_ratio"] == 0.0
    assert warmup["framework"]["wam"]["guidance"]["action_world_bypass"] is True
    assert gate["framework"]["wam"]["guidance"]["action_world_bypass"] is False
    assert gate["trainer"]["reset_world_gates_after_pretrained_load"] is True

    data_config = ROBOT_TYPE_CONFIG_MAP["robodojo_arx_x5"]
    expected_state = [
        "state.left_joints",
        "state.left_gripper",
        "state.right_joints",
        "state.right_gripper",
    ]
    expected_action = [key.replace("state.", "action.") for key in expected_state]
    assert data_config.video_keys == source_views
    assert data_config.state_keys == expected_state
    assert data_config.action_keys == expected_action
    assert DATASET_NAMED_MIXTURES["robodojo_v21"] == [
        ("RoboDojo_lerobot_v21_video", 1.0, "robodojo_arx_x5")
    ]
    assert DATASET_NAMED_MIXTURES["robodojo_v21_language_optional"] == [
        ("RoboDojo_lerobot_v21_language_v1", 1.0, "robodojo_arx_x5")
    ]
    normalization = [
        transform
        for transform in data_config.transform().transforms
        if isinstance(transform, StateActionTransform)
    ]
    assert len(normalization) == 2
    assert all(
        set(transform.normalization_modes.values()) == {"fastwam_zscore"}
        for transform in normalization
    )

    eval_dir = REPO_ROOT / "examples/RoboDojo/eval_files"
    eval_script = (eval_dir / "eval_robodojo.sh").read_text(encoding="utf-8")
    model_script = (eval_dir / "robodojo_model.py").read_text(encoding="utf-8")
    assert "build_robotwin_composite" in model_script
    assert "task_instruction" in model_script
    assert '"state": state' in model_script
    assert "Third_github" not in eval_script


def test_robodojo_checkpoint_preflight() -> None:
    """The selected state/corrnoise checkpoint must carry every server artifact."""

    from examples.RoboDojo.eval_files.verify_robodojo_checkpoint_contract import verify

    source = REPO_ROOT / "examples/RoboDojo/train_files/starvla_qwengroot_robodojo_baseline.yaml"
    zeros = [0.0] * 14
    stats = {
        "new_embodiment": {
            modality: {
                "min": zeros,
                "max": zeros,
                "mean": zeros,
                "std": [1.0] * 14,
                "q01": zeros,
                "q99": zeros,
                "mask": [True] * 14,
            }
            for modality in ("state", "action")
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "robodojo_baseline"
        checkpoint = run_dir / "checkpoints/steps_70000_pytorch_model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        config_text = source.read_text(encoding="utf-8")
        (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
        (run_dir / "config.full.yaml").write_text(config_text, encoding="utf-8")
        (run_dir / "dataset_statistics.json").write_text(json.dumps(stats), encoding="utf-8")
        np.save(run_dir / "action_correlation_cholesky.npy", np.eye(16 * 14, dtype=np.float32))

        summary = verify(str(checkpoint))
        assert summary["include_state"] is True
        assert summary["action_chunk"] == [16, 14]
        assert summary["use_correlated_noise"] is True
        assert summary["correlation_shape"] == [224, 224]

        (run_dir / "action_correlation_cholesky.npy").unlink()
        try:
            verify(str(checkpoint))
        except FileNotFoundError as exc:
            assert "Cholesky artifact is missing" in str(exc)
        else:
            raise AssertionError("corrnoise checkpoint without its Cholesky artifact was accepted")


if __name__ == "__main__":
    test_fastwam_metadata_and_composite()
    test_fastwam_checkpoint_sample_and_direct_sampler()
    test_fastwam_split_domain_and_wam_composite_target()
    test_fastwam_action_world_coflow_h16_single_future_contract()
    test_fastwam_train_infer_order_and_contract()
    test_policy_server_metadata_exposes_loaded_gate_ft_contract()
    test_fastwam_real_cluster_config_snapshots()
    test_fastwam_cluster_preflight_fixture()
    test_correlated_noise_artifact_is_a_strict_train_deploy_contract()
    test_robotwin_causal_query_and_coflow_yaml_contracts()
    test_coflow_checkpoint_verifier_accepts_only_h16_single_bridge()
    test_robodojo_train_and_deploy_contracts()
    test_robodojo_checkpoint_preflight()
