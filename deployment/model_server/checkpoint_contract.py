"""Read inference ABI fields without changing checkpoint model construction.

``config.yaml`` is intentionally an accessed-only snapshot.  Runtime contract
fields can therefore be absent even though they are present in the complete
``config.full.yaml`` saved beside it.  Model construction must keep using the
accessed snapshot for backwards compatibility, while deployment code may use
the complete snapshot to recover input/output ABI facts such as whether the
checkpoint was conditioned on proprioception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import yaml


def find_run_file(checkpoint: Path | str, name: str) -> Path:
    """Find a run artifact above a checkpoint path."""

    checkpoint = Path(checkpoint)
    for parent in checkpoint.parents:
        candidate = parent / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {name} above checkpoint {checkpoint}")


def load_checkpoint_contract_config(
    checkpoint: Path | str,
    *,
    accessed_config: Optional[dict] = None,
) -> Tuple[dict, Path]:
    """Load the best config snapshot for ABI checks.

    Prefer ``config.full.yaml`` when available.  Fall back to the accessed-only
    ``config.yaml`` for older runs that predate the full snapshot.  The caller
    may pass an already-loaded accessed config to avoid parsing it twice.
    """

    accessed_path = find_run_file(checkpoint, "config.yaml")
    full_path = accessed_path.with_name("config.full.yaml")
    selected_path = full_path if full_path.is_file() else accessed_path

    if selected_path == accessed_path and accessed_config is not None:
        config = accessed_config
    else:
        with selected_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise TypeError(f"Checkpoint config must be a mapping, got {type(config).__name__}: {selected_path}")
    return config, selected_path


def _get(config: dict, path: str, default: Any = None) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no"}
    return bool(value)


def _is_legacy_fastwam_iid_contract(config: dict) -> bool:
    """Recognize the original FastWAM IID ABI when no full config survives."""

    markers = {
        "framework.name": "QwenGR00T",
        "framework.action_model.action_dim": 14,
        "framework.action_model.state_dim": 14,
        "framework.action_model.action_horizon": 32,
        "datasets.vla_data.dataset_py": "lerobot_datasets",
        "datasets.vla_data.data_mix": "robotwin_fastwam",
        "datasets.vla_data.fastwam_expected_fps": 50,
        "datasets.vla_data.fastwam_direct_frame_sampling": True,
        "datasets.vla_data.action_mode": "abs",
        "datasets.vla_data.obs_image_size": [320, 384],
    }
    return all(_get(config, path) == wanted for path, wanted in markers.items())


def resolve_config_expects_state(config: dict) -> Tuple[bool, str]:
    """Resolve whether inference must provide state, plus an audit reason.

    An explicit ``include_state`` always wins, including explicit ``false``.
    The strict FastWAM fallback is only for old accessed-only snapshots where
    the key is absent and every other distinctive IID ABI marker is present.
    """

    vla_cfg = (config.get("datasets") or {}).get("vla_data") or {}
    if "include_state" in vla_cfg:
        enabled = _truthy(vla_cfg["include_state"])
        return enabled, f"explicit datasets.vla_data.include_state={enabled}"

    state_cfg = (config.get("framework") or {}).get("state") or {}
    if str(state_cfg.get("inject_mode", "none")) == "token":
        return True, "framework.state.inject_mode=token"

    if _is_legacy_fastwam_iid_contract(config):
        return True, "legacy FastWAM IID ABI inferred from strict 50Hz/32-step/320x384 markers"

    return False, "no explicit state-conditioning contract"


__all__ = [
    "find_run_file",
    "load_checkpoint_contract_config",
    "resolve_config_expects_state",
]
