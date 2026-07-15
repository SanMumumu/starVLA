"""Read inference ABI fields without changing checkpoint model construction.

``config.yaml`` is intentionally an accessed-only snapshot.  Runtime contract
fields can therefore be absent even though they are present in the complete
``config.full.yaml`` saved beside it.  Model construction must keep using the
accessed snapshot for backwards compatibility, while deployment code may use
the complete snapshot to recover input/output ABI facts such as whether the
checkpoint was conditioned on proprioception. The server may synchronize such
an input-routing flag after strict model construction because it does not alter
the checkpoint's parameter structure.
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


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no"}
    return bool(value)


def resolve_config_expects_state(config: dict) -> Tuple[bool, str]:
    """Read the training YAML's state switch for inference.

    ``config.full.yaml`` preserves the submitted YAML.  Do not infer this ABI
    from run names, dataset markers, or AIDI metadata: an absent field keeps the
    historical no-state behavior.
    """

    vla_cfg = (config.get("datasets") or {}).get("vla_data") or {}
    if "include_state" in vla_cfg:
        enabled = _truthy(vla_cfg["include_state"])
        return enabled, f"datasets.vla_data.include_state={enabled}"

    return False, "datasets.vla_data.include_state missing; default=False"


__all__ = [
    "find_run_file",
    "load_checkpoint_contract_config",
    "resolve_config_expects_state",
]
