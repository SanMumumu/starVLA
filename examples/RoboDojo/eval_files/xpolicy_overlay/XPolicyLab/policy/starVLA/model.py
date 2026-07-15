"""XPolicyLab import shim for the repository-owned RoboDojo adapter."""

import sys
from pathlib import Path


# Do not import through ``examples.RoboDojo``. Some policy environments ship a
# regular third-party package called ``examples`` which masks this repository's
# namespace directory. Resolve from this shim's location instead.
_EVAL_FILES = Path(__file__).resolve().parents[4]
if str(_EVAL_FILES) not in sys.path:
    sys.path.insert(0, str(_EVAL_FILES))

from robodojo_model import Model

__all__ = ["Model"]
