"""Lightweight regression coverage for RoboTwin runtime resolution."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[2]


def _fake_robotwin(root: Path) -> None:
    (root / "script").mkdir(parents=True)
    (root / "script/eval_policy.py").write_text("# fixture\n", encoding="utf-8")
    (root / "envs/robot").mkdir(parents=True)
    (root / "envs/__init__.py").write_text("", encoding="utf-8")
    (root / "envs/robot/__init__.py").write_text("", encoding="utf-8")
    (root / "envs/robot/planner.py").write_text(
        "class CuroboPlanner:\n    pass\n", encoding="utf-8"
    )
    (root / "curobo.py").write_text("__version__ = 'fixture'\n", encoding="utf-8")


def test_runtime_verifier_and_shell_resolver(tmp_path: Path) -> None:
    robotwin = tmp_path / "RoboTwin"
    _fake_robotwin(robotwin)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(robotwin)
    env["PYTHONNOUSERSITE"] = "1"
    env["ROBOTWIN_TORCH_EXTENSIONS_ROOT"] = str(tmp_path / "extensions")

    verified = subprocess.run(
        [
            sys.executable,
            str(EVAL_DIR / "verify_robotwin_runtime.py"),
            "--robotwin-path",
            str(robotwin),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ROBOTWIN_CUROBO_ABI_PASS" in verified.stdout

    command = "\n".join(
        [
            "set -euo pipefail",
            f"export ROBOTWIN_PYTHON={shlex.quote(sys.executable)}",
            "export ROBOTWIN_POLICY_NAME=model2robotwin_fastwam_interface",
            f"source {shlex.quote(str(EVAL_DIR / 'robotwin_runtime.sh'))}",
            f"prepare_robotwin_runtime {shlex.quote(str(robotwin))} 0",
            'test -x "${ROBOTWIN_PYTHON}"',
            'test -d "${TORCH_EXTENSIONS_DIR}"',
        ]
    )
    resolved = subprocess.run(
        ["bash", "-c", command],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ROBOTWIN_CUROBO_ABI_PASS" in resolved.stdout
    assert "ROBOTWIN_FASTWAM_ADAPTER_PASS" in resolved.stdout
    assert "selected python=" in resolved.stdout


def test_fastwam_eval_compositor_does_not_import_training_dependencies() -> None:
    """The simulator environment must not need Accelerate or dataloader deps."""

    script = "\n".join(
        [
            "import sys",
            "sys.modules['accelerate'] = None",
            "sys.modules['starVLA.dataloader'] = None",
            "import numpy as np",
            "from deployment.fastwam_image import build_robotwin_composite",
            "images = [np.full((480, 640, 3), value, dtype=np.uint8) for value in (20, 80, 140)]",
            "composite = np.asarray(build_robotwin_composite(images))",
            "assert composite.shape == (384, 320, 3)",
            "assert tuple(composite[10, 10]) == (20, 20, 20)",
            "assert tuple(composite[300, 10]) == (80, 80, 80)",
            "assert tuple(composite[300, 250]) == (140, 140, 140)",
        ]
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    adapter = (EVAL_DIR / "model2robotwin_fastwam_interface.py").read_text(encoding="utf-8")
    assert "from deployment.fastwam_image import build_robotwin_composite" in adapter
    assert "starVLA.dataloader" not in adapter


def test_fastwam_client_requires_the_requested_gate_checkpoint_and_phase() -> None:
    sys.path.insert(0, str(EVAL_DIR))
    try:
        import model2robotwin_fastwam_interface as adapter
    finally:
        sys.path.pop(0)

    requested = "/models/gate/checkpoints/steps_20000_pytorch_model.pt"

    def fake_standard_init(self, *args, **kwargs):
        served = kwargs.get("policy_ckpt_path")
        self.action_chunk_size = 32
        self.replan_steps = 24
        self.server_meta = {
            "ckpt_path": served,
            "action_chunk_size": 32,
            "expects_state": True,
            "framework_name": "QwenGR00T",
            "wam_two_stage_phase": "gate_ft",
            "wam_guidance_enabled": True,
            "wam_action_world_bypass": False,
            "wam_bridge_source": "predicted",
            "wam_world_eval_mode": "correct",
            "wam_gate_openness": 0.125,
        }

    with patch.object(adapter.StandardModelClient, "__init__", fake_standard_init):
        client = adapter.FastWAMRobotWinModelClient(
            policy_ckpt_path=requested,
            wam_expected_phase="gate_ft",
        )
        assert client.wam_expected_phase == "gate_ft"

    def fake_wrong_server(self, *args, **kwargs):
        fake_standard_init(self, *args, **kwargs)
        self.server_meta["ckpt_path"] = "/models/warmup/checkpoints/steps_80000_pytorch_model.pt"

    with patch.object(adapter.StandardModelClient, "__init__", fake_wrong_server):
        with pytest.raises(RuntimeError, match="different checkpoint"):
            adapter.FastWAMRobotWinModelClient(
                policy_ckpt_path=requested,
                wam_expected_phase="gate_ft",
            )


def test_fastwam_client_rejects_wrong_world_to_action_architecture() -> None:
    sys.path.insert(0, str(EVAL_DIR))
    try:
        import model2robotwin_fastwam_interface as adapter
    finally:
        sys.path.pop(0)

    requested = "/models/no_w2a/checkpoints/steps_80000_pytorch_model.pt"

    def fake_standard_init(self, *args, **kwargs):
        self.action_chunk_size = 32
        self.replan_steps = 24
        self.server_meta = {
            "ckpt_path": kwargs.get("policy_ckpt_path"),
            "expects_state": True,
            "wam_two_stage_phase": "predictor_warmup",
            "wam_guidance_enabled": True,
            "wam_action_world_bypass": True,
            "wam_world_to_action_enabled": False,
            "wam_bridge_source": "predicted",
            "wam_world_eval_mode": "correct",
        }

    with patch.object(adapter.StandardModelClient, "__init__", fake_standard_init):
        client = adapter.FastWAMRobotWinModelClient(
            policy_ckpt_path=requested,
            wam_expected_phase="predictor_warmup",
            wam_expected_world_to_action=False,
        )
        assert client.wam_expected_world_to_action is False

        with pytest.raises(RuntimeError, match="wrong world-to-action architecture"):
            adapter.FastWAMRobotWinModelClient(
                policy_ckpt_path=requested,
                wam_expected_phase="predictor_warmup",
                wam_expected_world_to_action=True,
            )
