import json
from pathlib import Path


def _local_model_type(model_name: str) -> str | None:
    """Read a local HF config without asking an older Transformers to parse it."""

    config_path = Path(str(model_name)).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    model_type = payload.get("model_type")
    return str(model_type).lower() if model_type is not None else None


def resolve_vlm_family(model_name: str) -> str:
    """Resolve local paths and Hub IDs to one StarVLA VLM implementation."""

    lowered = str(model_name).lower()
    normalized = lowered.replace("_", "").replace("-", "").replace(".", "")
    model_type = _local_model_type(model_name)
    if model_type == "qwen3_5" or "qwen35" in normalized or "rynnbrain11" in normalized:
        return "qwen3_5"
    if "qwen2.5-vl" in lowered or "nora" in lowered:
        return "qwen2_5"
    if "qwen3-vl" in lowered:
        return "qwen3"
    if "gemma-4" in lowered or "gemma4" in lowered:
        return "gemma4"
    if "molmo2" in lowered:
        return "molmo2"
    if "minicpm-v" in lowered or "minicpmv" in lowered:
        return "minicpm_v"
    if "florence" in lowered:
        return "florence"
    if "cosmos-reason2" in lowered:
        return "cosmos_reason2"
    raise NotImplementedError(f"VLM model {model_name} not implemented")


def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm
    family = resolve_vlm_family(vlm_name)

    if family == "qwen2_5":
        from .QWen2_5 import _QWen_VL_Interface

        return _QWen_VL_Interface(config)
    elif family == "qwen3":
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif family == "qwen3_5":
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    elif family == "gemma4":
        from .Gemma4 import _Gemma4_VL_Interface

        return _Gemma4_VL_Interface(config)
    elif family == "molmo2":
        from .Molmo2 import _Molmo2_VL_Interface

        return _Molmo2_VL_Interface(config)
    elif family == "minicpm_v":
        from .MiniCPM_V import _MiniCPM_VL_Interface

        return _MiniCPM_VL_Interface(config)
    elif family == "florence":
        from .Florence2 import _Florence_Interface

        return _Florence_Interface(config)
    elif family == "cosmos_reason2":
        # Cosmos-Reason2 is architecturally Qwen3-VL (VLM), but implemented
        # in world_model/ for historical reasons. Import directly.
        from starVLA.model.modules.vlm.CosmosReason2 import _CosmosReason2_Interface

        return _CosmosReason2_Interface(config)
    raise AssertionError(f"Unhandled VLM family {family!r}")
