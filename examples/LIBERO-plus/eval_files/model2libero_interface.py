"""Deprecated: use ``examples.LIBERO.eval_files.model2libero_interface`` instead.

LIBERO-Plus eval imports the maintained LIBERO client, which:
  - builds the 512x256 [third-person | wrist] composite from server metadata
  - forwards 8-D proprioception when ``expects_state=true``
  - leaves action un-normalization to the policy server
"""

from examples.LIBERO.eval_files.model2libero_interface import *  # noqa: F403
