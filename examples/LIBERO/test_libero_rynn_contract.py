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

from deployment.libero_image import (
    LIBERO_COMPOSITE_LAYOUT,
    LIBERO_COMPOSITE_SIZE,
    LIBERO_COMPOSITE_SOURCE_VIEW_KEYS,
    LIBERO_COMPOSITE_VIEW_KEY,
    build_libero_composite,
)
from examples.LIBERO.eval_files import model2libero_interface
from examples.LIBERO.train_files.verify_rynn_h8_recipe import verify


CONFIG_PATH = REPO_ROOT / "examples/LIBERO/train_files/rynn_base_h8_50k.yaml"
FULLRES_CONFIG_PATH = (
    REPO_ROOT
    / "examples/LIBERO/train_files/rynn_base_h8_current_dino_fullres_50k.yaml"
)
TRAIN_FILES_DIR = CONFIG_PATH.parent
EXEC_DIR = REPO_ROOT.parent / "LIBERO"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def server_metadata() -> dict:
    return {
        "action_chunk_size": 8,
        "image_layout": LIBERO_COMPOSITE_LAYOUT,
        "obs_image_size": list(LIBERO_COMPOSITE_SIZE),
        "composite_view_key": LIBERO_COMPOSITE_VIEW_KEY,
        "composite_source_view_keys": list(
            LIBERO_COMPOSITE_SOURCE_VIEW_KEYS
        ),
        "expects_state": False,
    }


def test_libero_composite_preserves_camera_order_and_pixels() -> None:
    third = np.zeros((256, 256, 3), dtype=np.uint8)
    third[..., 0] = 255
    wrist = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist[..., 2] = 255

    composite = np.asarray(build_libero_composite([third, wrist]))
    assert composite.shape == (256, 512, 3)
    assert np.all(composite[:, :256] == third)
    assert np.all(composite[:, 256:] == wrist)

    with pytest.raises(ValueError, match="third_person, wrist"):
        build_libero_composite([third])


def test_jointflow_routes_libero_layout_to_shared_compositor() -> None:
    from starVLA.dataloader.jointflow.joint_dataset import _composite_contract

    size, view_key, builder = _composite_contract(LIBERO_COMPOSITE_LAYOUT)
    assert size == LIBERO_COMPOSITE_SIZE
    assert view_key == LIBERO_COMPOSITE_VIEW_KEY
    assert builder is build_libero_composite


@pytest.mark.parametrize(
    ("num_processes", "accumulation"),
    [(1, 64), (16, 1)],
)
def test_libero_recipe_has_fixed_bs256_topologies(
    num_processes: int,
    accumulation: int,
) -> None:
    summary = verify(load_config(), num_processes=num_processes)
    assert summary["global_batch_size"] == 256
    assert summary["gradient_accumulation_steps"] == accumulation
    assert summary["max_train_steps"] == 60000
    assert summary["save_interval"] == 5000
    assert summary["pooled_dino_grid"] == [8, 16]


def test_libero_fullres_current_dino_variant_only_changes_current_pool() -> None:
    base = load_config()
    variant = yaml.safe_load(FULLRES_CONFIG_PATH.read_text(encoding="utf-8"))
    summary = verify(variant, num_processes=16)

    assert variant["framework"]["dino"]["current_dino_pool"] == 1
    assert variant["framework"]["dino"]["dino_pool"] == 2
    assert summary["current_dino_grid"] == [16, 32]
    assert summary["pooled_dino_grid"] == [8, 16]
    assert summary["global_batch_size"] == 256

    expected = deepcopy(base)
    expected["run_id"] = variant["run_id"]
    expected["framework"]["dino"]["current_dino_pool"] = 1
    assert variant == expected


def test_libero_recipe_rejects_camera_order_drift() -> None:
    config = deepcopy(load_config())
    config["datasets"]["vla_data"]["composite_source_view_keys"].reverse()
    with pytest.raises(ValueError, match="composite_source_view_keys"):
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


def test_libero_client_builds_the_same_composite(monkeypatch) -> None:
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
    assert len(sent["image"]) == 1
    composite = np.asarray(sent["image"][0])
    assert composite.shape == (256, 512, 3)
    assert np.all(composite[:, :256] == third)
    assert np.all(composite[:, 256:] == wrist)
    assert "state" not in sent


def test_libero_recipe_matches_wam_exp5_nostate_contract() -> None:
    """WAM exp5 alignment: no proprio, 60K schedule, empty freeze list."""
    config = load_config()
    assert config["datasets"]["vla_data"]["include_state"] is False
    assert config["framework"]["action_model"]["state_dim"] == 0
    assert config["trainer"]["max_train_steps"] == 60000
    assert config["trainer"]["num_warmup_steps"] == 3000
    assert config["trainer"]["learning_rate"]["action_model"] == 5.0e-05
    assert config["trainer"]["freeze_modules"] == ""
    assert config["trainer"]["logging_frequency"] == 100
    assert config["framework"]["planner"]["text_supervision"]["enabled"] is False
    assert config["datasets"]["vla_data"]["text_annotations"]["enabled"] is False


def test_libero_aidi_jobs_use_16gpu_train_and_8gpu_eval() -> None:
    train = yaml.safe_load(
        (EXEC_DIR / "job_train_16gpu.yaml").read_text(encoding="utf-8")
    )
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
    assert train["REQUIRED"]["WORKER_MIN_NUM"] == 1
    assert train["REQUIRED"]["GPU_PER_WORKER"] == 16
    assert fullres_train["REQUIRED"]["WORKER_MIN_NUM"] == 1
    assert fullres_train["REQUIRED"]["GPU_PER_WORKER"] == 16
    assert "rynn_base_h8_current_dino_fullres_50k.yaml" in fullres_train["REQUIRED"][
        "RUN_SCRIPTS"
    ]
    assert server["REQUIRED"]["GPU_PER_WORKER"] == 8
    assert client["REQUIRED"]["GPU_PER_WORKER"] == 8
    assert client["REQUIRED"]["environment"]["NUM_CLIENTS"] == "8"
    assert not (EXEC_DIR / "job_train_1gpu.yaml").exists()
    assert "launcher.sh" in train["REQUIRED"]["RUN_SCRIPTS"]
    assert "launcher.sh" in fullres_train["REQUIRED"]["RUN_SCRIPTS"]
    launcher = (EXEC_DIR / "run.sh").read_text(encoding="utf-8")
    assert "run_policy_servers_8.sh" in launcher
    assert "eval_libero_plus_8.sh" in launcher
    assert "NUM_CLIENTS=8" in launcher
    assert "NUM_SERVERS=8" in launcher
