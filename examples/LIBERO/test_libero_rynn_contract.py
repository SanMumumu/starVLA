from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.LIBERO.eval_files import model2libero_interface
from examples.LIBERO.train_files.verify_rynn_h8_recipe import verify


CONFIG_PATH = (
    REPO_ROOT
    / "examples/LIBERO/train_files/rynn_base_h8_current_dino_fullres_50k.yaml"
)
TRAIN_FILES_DIR = CONFIG_PATH.parent
EXEC_DIR = REPO_ROOT / "执行脚本" / "LIBERO"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def server_metadata() -> dict:
    return {
        "action_chunk_size": 8,
        "image_layout": "separate_views",
        "obs_image_size": [224, 224],
        "composite_view_key": None,
        "composite_source_view_keys": [],
        "expects_state": False,
    }


def test_libero_uses_official_separate_view_contract() -> None:
    from starVLA.dataloader.gr00t_lerobot.data_config import (
        Libero4in1DataConfig,
    )

    config = load_config()
    data = config["datasets"]["vla_data"]
    assert Libero4in1DataConfig.video_keys == [
        "video.primary_image",
        "video.wrist_image",
    ]
    assert data["image_layout"] == "separate_views"
    assert data["obs_image_size"] == [224, 224]
    assert "composite_source_view_keys" not in data
    assert "composite_view_key" not in data


def test_jointflow_selects_only_future_third_person() -> None:
    from starVLA.dataloader.jointflow.joint_dataset import (
        _select_future_target_views,
    )

    third = np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8)
    wrist = np.full((256, 256, 3), [0, 0, 255], dtype=np.uint8)
    views, keys = _select_future_target_views(
        ["video.primary_image", "video.wrist_image"],
        [third, wrist],
        {"future_target_view_keys": ["video.primary_image"]},
    )

    assert keys == ["video.primary_image"]
    assert len(views) == 1
    assert np.array_equal(views[0], third)


def test_libero_recipe_has_fixed_bs128_no_accumulation() -> None:
    summary = verify(load_config(), num_processes=16)
    assert summary["global_batch_size"] == 128
    assert summary["micro_batch_size"] == 8
    assert summary["gradient_accumulation_steps"] == 1
    assert summary["max_train_steps"] == 50000
    assert summary["save_interval"] == 5000
    assert summary["future_target_view_keys"] == ["video.primary_image"]
    assert summary["future_dino_image_size"] == [224, 224]
    assert summary["current_view_count"] == 2
    assert summary["future_view_count"] == 1
    assert summary["pooled_dino_grid"] == [14, 14]

    with pytest.raises(ValueError, match="exactly 16 processes"):
        verify(load_config(), num_processes=1)


def test_libero_fullres_dino_uses_two_current_and_one_future_view() -> None:
    config = load_config()
    summary = verify(config, num_processes=16)

    assert config["framework"]["dino"]["current_dino_pool"] is None
    assert config["framework"]["dino"]["dino_pool"] == 1
    assert config["framework"]["world_action_mot"]["num_current_world_views"] == 2
    assert summary["current_dino_grid"] == [28, 14]
    assert summary["pooled_dino_grid"] == [14, 14]
    assert summary["global_batch_size"] == 128


def test_libero_recipe_rejects_composite_layout_drift() -> None:
    config = deepcopy(load_config())
    config["datasets"]["vla_data"]["image_layout"] = (
        "libero_dual_view_composite"
    )
    with pytest.raises(ValueError, match="image_layout"):
        verify(config, num_processes=16)


def test_libero_recipe_rejects_future_target_view_drift() -> None:
    config = deepcopy(load_config())
    config["datasets"]["vla_data"]["future_target_view_keys"] = [
        "video.wrist_image"
    ]
    with pytest.raises(ValueError, match="future_target_view_keys"):
        verify(config, num_processes=16)


@pytest.mark.parametrize("mode", ["inherit", "fp32_shell"])
def test_libero_recipe_leaves_precision_to_starvla_runtime(mode: str) -> None:
    config = deepcopy(load_config())
    assert "action_precision_mode" not in config["framework"]["world_action_mot"]

    config["framework"]["world_action_mot"]["action_precision_mode"] = mode
    with pytest.raises(ValueError, match="action_precision_mode.*must be omitted"):
        verify(config, num_processes=16)


def test_libero_train_files_have_no_early_experiment_variants() -> None:
    assert not list(TRAIN_FILES_DIR.glob("run_libero_train_exp*.sh"))
    assert not list(TRAIN_FILES_DIR.glob("run_libero_train_add*.sh"))
    assert not list(TRAIN_FILES_DIR.glob("starvla_wam_*.yaml"))
    assert not list(TRAIN_FILES_DIR.glob("*_old.yaml"))


def test_libero_recipe_rejects_upstream_control_drift() -> None:
    config = deepcopy(load_config())
    config["trainer"]["learning_rate"]["base"] = 3.0e-05
    with pytest.raises(ValueError, match="upstream-control drift.*learning_rate.base"):
        verify(config, num_processes=16)


def test_libero_client_preserves_separate_official_views(monkeypatch) -> None:
    requests = []

    class FakePolicy:
        def __init__(self, host, port):
            assert host == "server"
            assert port == 6698

        def get_server_metadata(self):
            return server_metadata()

        def predict_action(self, payload):
            requests.append(payload)
            return {
                "ok": True,
                "data": {
                    "actions": np.zeros((1, 8, 7), dtype=np.float32),
                },
            }

    monkeypatch.setattr(
        model2libero_interface,
        "WebsocketClientPolicy",
        FakePolicy,
    )
    client = model2libero_interface.ModelClient(
        host="server",
        port=6698,
        action_ensemble=False,
    )

    third = np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8)
    wrist = np.full((256, 256, 3), [0, 0, 255], dtype=np.uint8)
    client.step(
        {
            "image": [third, wrist],
            "lang": "pick up the object",
        },
        step=0,
    )

    sent = requests[0]["examples"][0]
    assert len(sent["image"]) == 2
    primary_sent = np.asarray(sent["image"][0])
    wrist_sent = np.asarray(sent["image"][1])
    assert primary_sent.shape == (224, 224, 3)
    assert wrist_sent.shape == (224, 224, 3)
    assert np.all(primary_sent == [255, 0, 0])
    assert np.all(wrist_sent == [0, 0, 255])
    assert "state" not in sent


def test_libero_recipe_matches_wam_exp5_nostate_contract() -> None:
    """Maintained contract: no proprio, 50K schedule, empty freeze list."""
    config = load_config()
    assert config["datasets"]["vla_data"]["include_state"] is False
    assert config["framework"]["action_model"]["state_dim"] == 0
    assert config["trainer"]["max_train_steps"] == 50000
    assert config["trainer"]["num_warmup_steps"] == 3000
    assert config["trainer"]["learning_rate"]["action_model"] == 5.0e-05
    assert config["trainer"]["freeze_modules"] == ""
    assert config["trainer"]["logging_frequency"] == 100
    assert config["framework"]["planner"]["text_supervision"]["enabled"] is False
    assert config["datasets"]["vla_data"]["text_annotations"]["enabled"] is False


def test_libero_aidi_jobs_use_16gpu_train_and_8gpu_eval() -> None:
    fullres_train = yaml.safe_load(
        (EXEC_DIR / "job_train_16gpu_current_dino_fullres.yaml").read_text(
            encoding="utf-8"
        )
    )
    server = yaml.safe_load(
        (EXEC_DIR / "job_server.yaml").read_text(encoding="utf-8")
    )
    client = yaml.safe_load(
        (EXEC_DIR / "job_client_libero_plus.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert fullres_train["REQUIRED"]["WORKER_MIN_NUM"] == 1
    assert fullres_train["REQUIRED"]["GPU_PER_WORKER"] == 16
    assert "rynn_base_h8_current_dino_fullres_50k.yaml" in fullres_train["REQUIRED"][
        "RUN_SCRIPTS"
    ]
    assert server["REQUIRED"]["GPU_PER_WORKER"] == 8
    assert client["REQUIRED"]["GPU_PER_WORKER"] == 8
    assert client["REQUIRED"]["environment"]["NUM_CLIENTS"] == "8"
    assert not (EXEC_DIR / "job_train_1gpu.yaml").exists()
    assert not (EXEC_DIR / "job_train_16gpu.yaml").exists()
    assert not (
        TRAIN_FILES_DIR / "rynn_base_h8_50k.yaml"
    ).exists()
    assert "launcher.sh" in fullres_train["REQUIRED"]["RUN_SCRIPTS"]
    assert "bs128-50k" in fullres_train["REQUIRED"]["JOB_NAME"]
    assert "dino-fullres" in fullres_train["REQUIRED"]["JOB_NAME"]
    launcher = (EXEC_DIR / "run.sh").read_text(encoding="utf-8")
    assert "run_policy_servers_8.sh" in launcher
    assert "eval_libero_plus_8.sh" in launcher
    assert "NUM_CLIENTS=8" in launcher
    assert "NUM_SERVERS=8" in launcher
