from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


EVAL_DIR = Path(__file__).resolve().parent
MODULE_PATH = EVAL_DIR / "robodojo_fast_rollout.py"
SPEC = importlib.util.spec_from_file_location("_robodojo_fast_rollout_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_official_protocol_shape() -> None:
    assert len(MODULE.CANONICAL_TASKS) == 42
    assert len(MODULE.DEFAULT_CONFIGS) == 54
    assert len(MODULE.GENERALIZATION) == 12
    assert all(f"{task}_random" in MODULE.DEFAULT_CONFIGS for task in MODULE.GENERALIZATION)


def test_one_isaac_process_per_gpu(monkeypatch) -> None:
    monkeypatch.delenv("TASKS_PER_GPU_LAYOUT", raising=False)
    monkeypatch.delenv("TASKS_PER_GPU", raising=False)
    gpu_ids = [str(index) for index in range(8)]
    assert MODULE.parse_slots_per_gpu(gpu_ids) == [1] * 8
    assert MODULE.iter_slot_assignments(gpu_ids, [1] * 8) == [
        (str(index), 0) for index in range(8)
    ]


def test_full_launcher_restores_vector_client_contract() -> None:
    launcher = (EVAL_DIR / "run_aidi_robodojo_fast_full.sh").read_text(encoding="utf-8")
    assert 'NUM_CLIENTS="${NUM_CLIENTS:-8}"' in launcher
    assert 'TASKS_PER_GPU=1' in launcher
    assert 'ROBODOJO_ENVS_PER_CLIENT="${ROBODOJO_ENVS_PER_CLIENT:-6}"' in launcher
    assert 'ROBODOJO_REPLAN_STEPS:-12' in launcher
    assert 'NUM_SERVERS="${NUM_SERVERS:-8}"' in launcher


def test_full_scheduler_records_and_forwards_rtc_configuration() -> None:
    scheduler = MODULE_PATH.read_text(encoding="utf-8")
    for name in (
        "ROBODOJO_RTC_ENABLED",
        "ROBODOJO_RTC_EXECUTION_HORIZON",
        "ROBODOJO_RTC_INFERENCE_DELAY",
        "ROBODOJO_RTC_MAX_GUIDANCE_WEIGHT",
        "ROBODOJO_RTC_PREFIX_ATTENTION_SCHEDULE",
        "ROBODOJO_RTC_DEBUG_MAX_REPLANS",
    ):
        assert name in scheduler
    assert '"rtc": {' in scheduler


def test_aidi_jobs_reserve_two_eight_gpu_nodes() -> None:
    runbook_dir = EVAL_DIR.parents[2] / "执行脚本/Robodojo"
    client = yaml.safe_load((runbook_dir / "job_client_robodojo.yaml").read_text(encoding="utf-8"))
    server = yaml.safe_load((runbook_dir / "job_server_rynnbrain.yaml").read_text(encoding="utf-8"))
    assert client["REQUIRED"]["GPU_PER_WORKER"] == 8
    assert server["REQUIRED"]["GPU_PER_WORKER"] == 8
    environment = client["REQUIRED"]["environment"]
    assert environment["NUM_CLIENTS"] == "8"
    assert environment["ROBODOJO_ENVS_PER_CLIENT"] == "6"
    assert environment["ROBODOJO_REPLAN_STEPS"] == "12"


def test_eight_servers_use_standard_non_batching_entrypoint() -> None:
    launcher = (EVAL_DIR / "run_robodojo_policy_servers_8.sh").read_text(encoding="utf-8")
    assert 'NUM_SERVERS="${NUM_SERVERS:-8}"' in launcher
    assert "--max_batch_size" not in launcher
    assert "--batch_timeout_ms" not in launcher
