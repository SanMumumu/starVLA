"""Focused regression checks for the checkpoint-compatible FastWAM ABI."""

# ruff: noqa: E402 - insert repository root before importing the local package

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
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
    np.testing.assert_array_equal(client._prepare_state_for_server(state), state.reshape(1, 14))
    np.testing.assert_array_equal(client._prepare_action_for_env(state), state)


def test_fastwam_real_cluster_config_snapshots() -> None:
    """The accessed config may omit state while config.full.yaml preserves the ABI."""

    from deployment.model_server.checkpoint_contract import (
        load_checkpoint_contract_config,
        resolve_config_expects_state,
    )
    from examples.Robotwin.eval_files.verify_fastwam_checkpoint_contract import verify

    snapshot_dir = REPO_ROOT / "执行脚本/集群FastWAM IID checkpoint yaml"
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
        "explicit datasets.vla_data.include_state=False",
    )

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
            "explicit datasets.vla_data.include_state=True",
        )
        summary = verify(checkpoint, replan_steps=24)
        assert Path(summary["contract_config"]).name == "config.full.yaml"

        # Old runs without a full snapshot are accepted only through the
        # distinctive, strict FastWAM IID marker set.
        (run_dir / "config.full.yaml").unlink()
        fallback_cfg, fallback_path = load_checkpoint_contract_config(checkpoint, accessed_config=accessed)
        expects_state, source = resolve_config_expects_state(fallback_cfg)
        assert fallback_path.name == "config.yaml"
        assert expects_state is True
        assert source.startswith("legacy FastWAM IID ABI inferred")
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


if __name__ == "__main__":
    test_fastwam_metadata_and_composite()
    test_fastwam_checkpoint_sample_and_direct_sampler()
    test_fastwam_split_domain_and_wam_composite_target()
    test_fastwam_train_infer_order_and_contract()
    test_fastwam_real_cluster_config_snapshots()
    test_fastwam_cluster_preflight_fixture()
