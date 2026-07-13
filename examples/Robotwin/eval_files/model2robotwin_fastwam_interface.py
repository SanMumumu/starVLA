"""RoboTwin client for StarVLA checkpoints trained with FastWAM's data ABI."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # RoboTwin adds this eval_files directory directly to PYTHONPATH.
    from model2robotwin_interface import ModelClient as StandardModelClient
    from model2robotwin_interface import eval, reset_model  # noqa: A004 - RoboTwin plugin API
except ImportError:  # Package-style import used by repository tests.
    from .model2robotwin_interface import ModelClient as StandardModelClient
    from .model2robotwin_interface import eval, reset_model  # noqa: A004 - RoboTwin plugin API
from starVLA.dataloader.fastwam_image import build_robotwin_composite


class FastWAMRobotWinModelClient(StandardModelClient):
    """Use the same composite/state/action ordering as FastWAM training."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.action_chunk_size != 32:
            raise RuntimeError(f"FastWAM checkpoint must expose action_chunk_size=32, got {self.action_chunk_size}")
        if self.server_meta.get("expects_state") is not True:
            raise RuntimeError(
                "FastWAM server did not declare expects_state=true. Refusing to evaluate because "
                "dropping the checkpoint's 14-D proprio input would violate its training ABI. "
                f"server_meta={self.server_meta}"
            )

    def _prepare_images(self, images: list[np.ndarray]) -> list[np.ndarray]:
        composite = build_robotwin_composite(images)
        return [np.asarray(composite, dtype=np.uint8)]

    def _prepare_state_for_server(self, state: Optional[np.ndarray]) -> np.ndarray:
        if state is None:
            raise ValueError("FastWAM checkpoint requires the current 14-D proprio state")
        state = np.asarray(state, dtype=np.float32)
        if state.size != 14:
            raise ValueError(f"FastWAM checkpoint requires exactly 14 state values, got shape={state.shape}")
        return state.reshape(1, 14)

    def _prepare_action_for_env(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action)
        if action.shape[-1] != 14:
            raise ValueError(f"FastWAM checkpoint must return 14-D actions, got shape={action.shape}")
        # Release/checkpoint and RoboTwin environment both use
        # [left6, left_gripper, right6, right_gripper].
        return action


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided in config")
    return FastWAMRobotWinModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 5694),
        unnorm_key=usr_args.get("unnorm_key"),
        action_mode=usr_args.get("action_mode", "abs"),
        normalization_mode=usr_args.get(
            "action_normalization_mode",
            usr_args.get("normalization_mode", "fastwam_zscore"),
        ),
        replan_steps=usr_args.get("replan_steps", 24),
    )


__all__ = ["FastWAMRobotWinModelClient", "eval", "get_model", "reset_model"]
