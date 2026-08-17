"""RoboTwin client for StarVLA checkpoints trained with FastWAM's data ABI."""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

try:  # RoboTwin adds this eval_files directory directly to PYTHONPATH.
    from model2robotwin_interface import ModelClient as StandardModelClient
    from model2robotwin_interface import eval, reset_model  # noqa: A004 - RoboTwin plugin API
except ImportError:  # Package-style import used by repository tests.
    from .model2robotwin_interface import ModelClient as StandardModelClient
    from .model2robotwin_interface import eval, reset_model  # noqa: A004 - RoboTwin plugin API
from deployment.fastwam_image import build_robotwin_composite


class FastWAMRobotWinModelClient(StandardModelClient):
    """Use FastWAM's composite/action ABI and its checkpoint-declared state ABI."""

    def __init__(
        self,
        *args,
        wam_expected_phase: Optional[str] = None,
        wam_expected_world_to_action: Optional[bool] = None,
        coflow_inference_mode: Optional[str] = None,
        coflow_inference_horizon: Optional[int] = None,
        coflow_inference_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        requested_checkpoint = kwargs.get("policy_ckpt_path")
        if requested_checkpoint is None and args:
            requested_checkpoint = args[0]
        super().__init__(*args, **kwargs)
        served_checkpoint = self.server_meta.get("ckpt_path")
        if not isinstance(served_checkpoint, str) or not served_checkpoint:
            raise RuntimeError(
                "FastWAM server did not advertise the checkpoint it loaded; refusing to "
                f"evaluate against ambiguous metadata: {self.server_meta}"
            )
        if requested_checkpoint is None or os.path.normpath(str(requested_checkpoint)) != os.path.normpath(
            served_checkpoint
        ):
            raise RuntimeError(
                "FastWAM client connected to a different checkpoint than requested: "
                f"requested={requested_checkpoint!r}, served={served_checkpoint!r}. "
                "Use the HOST/port belonging to this checkpoint's server job."
            )
        if self.action_chunk_size not in {16, 32, 50}:
            raise RuntimeError(
                "FastWAM adapter supports H16, FastWAM/Rynn H32, or legacy Rynn H50 checkpoints, got "
                f"action_chunk_size={self.action_chunk_size}"
            )
        expects_state = self.server_meta.get("expects_state")
        if not isinstance(expects_state, bool):
            raise RuntimeError(
                "FastWAM server did not declare a boolean expects_state contract. Refusing to guess whether "
                "the checkpoint was trained with proprioception. "
                f"server_meta={self.server_meta}"
            )
        self.expects_state = expects_state
        actual_wam_recipe = self.server_meta.get("wam_two_stage_recipe")
        if actual_wam_recipe in {"isolated_queries_v4", "causal_action_world_queries_v1"}:
            if self.server_meta.get("wam_pretraining_aligned_queries") is not True:
                raise RuntimeError(
                    f"{actual_wam_recipe} server did not activate its pretraining-aligned ACT query path"
                )
            if self.server_meta.get("wam_baseline_action_context") is not False:
                raise RuntimeError(
                    f"{actual_wam_recipe} must condition the policy with ACT queries, not the query-free "
                    "native action context"
                )
            if self.server_meta.get("wam_action_query_count") != self.action_chunk_size:
                raise RuntimeError(
                    f"{actual_wam_recipe} ACT query count does not match the action chunk: "
                    f"queries={self.server_meta.get('wam_action_query_count')}, "
                    f"chunk={self.action_chunk_size}"
                )
            future_capacity = self.server_meta.get("wam_future_query_capacity")
            if isinstance(future_capacity, bool) or not isinstance(future_capacity, int) or future_capacity <= 0:
                raise RuntimeError(
                    f"{actual_wam_recipe} server did not expose a valid FUTURE query bank capacity"
                )
            if (
                actual_wam_recipe == "causal_action_world_queries_v1"
                and self.server_meta.get("wam_future_query_through_qwen") is not True
            ):
                raise RuntimeError(
                    "causal_action_world_queries_v1 server did not activate FUTURE-query Qwen injection"
                )
            if actual_wam_recipe == "causal_action_world_queries_v1":
                action_query_last = self.server_meta.get("wam_action_query_last") is True
                expected_pattern = (
                    "causal_future_then_act"
                    if action_query_last
                    else "causal_act_then_future"
                )
                if self.server_meta.get("wam_query_attention_pattern") != expected_pattern:
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server query order disagrees with "
                        f"action_query_last={action_query_last}"
                    )
                if self.server_meta.get("wam_single_qwen_forward") is not True:
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server reports an obsolete two-pass Qwen path"
                    )
                if self.server_meta.get("wam_queries_are_final_suffix") is not True:
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server did not declare both query groups "
                        "as its physical final token suffix"
                    )
                if self.server_meta.get("wam_future_query_count") != 64:
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server does not use exactly 64 FUTURE queries"
                    )
                if self.server_meta.get("wam_detach_action_query_in_world_pass") is not False:
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server detached ACT intent from the world objective"
                    )
                if self.server_meta.get("qwen_attn_implementation") != "flash_attention_2":
                    raise RuntimeError(
                        "causal_action_world_queries_v1 server is not using flash_attention_2"
                    )
        self.wam_expected_phase = (
            None if wam_expected_phase is None else str(wam_expected_phase).lower()
        )
        if self.wam_expected_phase not in {None, "predictor_warmup", "gate_ft"}:
            raise ValueError(
                "wam_expected_phase must be predictor_warmup/gate_ft or null, got "
                f"{self.wam_expected_phase!r}"
            )
        if self.wam_expected_phase is not None:
            actual_phase = self.server_meta.get("wam_two_stage_phase")
            if actual_phase != self.wam_expected_phase:
                raise RuntimeError(
                    "Connected checkpoint has the wrong WAM two-stage phase: "
                    f"expected={self.wam_expected_phase!r}, actual={actual_phase!r}, "
                    f"server_meta={self.server_meta}"
                )
            if self.server_meta.get("wam_guidance_enabled") is not True:
                raise RuntimeError("Expected WAM guidance, but the connected server reports it disabled")
            if self.server_meta.get("wam_bridge_source") != "predicted":
                raise RuntimeError(
                    "Two-stage WAM evaluation must use the predicted future, got "
                    f"{self.server_meta.get('wam_bridge_source')!r}"
                )
            if self.server_meta.get("wam_world_eval_mode") != "correct":
                raise RuntimeError(
                    "Two-stage WAM evaluation must use world_eval_mode='correct', got "
                    f"{self.server_meta.get('wam_world_eval_mode')!r}"
                )
            if self.wam_expected_phase == "gate_ft":
                if self.server_meta.get("wam_action_world_bypass") is not False:
                    raise RuntimeError(
                        "gate_ft evaluation requires the trained world-guidance path, but the "
                        "server reports action_world_bypass enabled"
                    )
                gate_openness = self.server_meta.get("wam_gate_openness")
                if isinstance(gate_openness, bool) or not isinstance(gate_openness, (int, float)) or not math.isfinite(
                    float(gate_openness)
                ):
                    raise RuntimeError(
                        "gate_ft server did not expose a finite loaded gate openness; the 20k "
                        f"gate weights cannot be verified: {gate_openness!r}"
                    )
        self.wam_expected_world_to_action = wam_expected_world_to_action
        if self.wam_expected_world_to_action is not None:
            if not isinstance(self.wam_expected_world_to_action, bool):
                raise ValueError(
                    "wam_expected_world_to_action must be bool or null, got "
                    f"{self.wam_expected_world_to_action!r}"
                )
            actual_world_to_action = self.server_meta.get("wam_world_to_action_enabled")
            if actual_world_to_action is not self.wam_expected_world_to_action:
                raise RuntimeError(
                    "Connected checkpoint has the wrong world-to-action architecture: "
                    f"expected={self.wam_expected_world_to_action!r}, "
                    f"actual={actual_world_to_action!r}, server_meta={self.server_meta}"
                )
        self.coflow_inference_mode = (
            None if coflow_inference_mode is None else str(coflow_inference_mode).lower()
        )
        self.coflow_inference_horizon = (
            None if coflow_inference_horizon is None else int(coflow_inference_horizon)
        )
        self.coflow_inference_seed = (
            None if coflow_inference_seed is None else int(coflow_inference_seed)
        )
        self._coflow_query_index = 0
        if self.coflow_inference_mode not in {None, "policy", "diagonal"}:
            raise ValueError(
                "coflow_inference_mode must be policy/diagonal or null, got "
                f"{self.coflow_inference_mode!r}"
            )
        if self.coflow_inference_seed is not None:
            if self.coflow_inference_mode is None:
                raise ValueError("coflow_inference_seed requires coflow_inference_mode")
            if self.coflow_inference_seed < 0:
                raise ValueError("coflow_inference_seed must be non-negative")
        if self.coflow_inference_mode is not None:
            if self.server_meta.get("framework_name") != "QwenActionWorldCoFlow":
                raise RuntimeError(
                    "Co-Flow inference controls were requested, but the connected server is not a "
                    "QwenActionWorldCoFlow checkpoint. Check HOST/port and synchronize policy_wrapper.py. "
                    f"server_meta={self.server_meta}"
                )
            if self.action_chunk_size != 16 or self.server_meta.get("coflow_bridge_horizon") != 16:
                raise RuntimeError(
                    "Co-Flow evaluation requires the single H16 bridge contract; "
                    f"server_meta={self.server_meta}"
                )
            supported_modes = self.server_meta.get("coflow_supported_inference_modes", [])
            if self.coflow_inference_mode not in supported_modes:
                raise RuntimeError(
                    f"server does not advertise Co-Flow mode {self.coflow_inference_mode!r}: "
                    f"{supported_modes!r}"
                )
        if self.coflow_inference_horizon is not None:
            if self.coflow_inference_mode is None:
                raise ValueError("coflow_inference_horizon requires coflow_inference_mode")
            if self.coflow_inference_horizon != 16:
                raise ValueError("single-bridge coflow_inference_horizon must be 16")
            if self.replan_steps > self.coflow_inference_horizon:
                raise ValueError(
                    f"replan_steps={self.replan_steps} exceeds returned Co-Flow horizon "
                    f"{self.coflow_inference_horizon}"
                )

    def _prepare_images(self, images: list[np.ndarray]) -> list[np.ndarray]:
        composite = build_robotwin_composite(images)
        return [np.asarray(composite, dtype=np.uint8)]

    def _prepare_state_for_server(self, state: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not self.expects_state:
            return None
        if state is None:
            raise ValueError("This FastWAM checkpoint was trained with include_state=true and requires 14-D state")
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

    def reset(self, task_description: str) -> None:
        super().reset(task_description)
        self._coflow_query_index = 0

    def _extra_inference_request_kwargs(self) -> dict:
        if self.coflow_inference_mode is None:
            return {}
        request = {"coflow_inference_mode": self.coflow_inference_mode}
        if self.coflow_inference_horizon is not None:
            request["coflow_inference_horizon"] = self.coflow_inference_horizon
        if self.coflow_inference_seed is not None:
            request["coflow_inference_seed"] = self.coflow_inference_seed + self._coflow_query_index
            self._coflow_query_index += 1
        return request


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
        wam_expected_phase=usr_args.get("wam_expected_phase"),
        wam_expected_world_to_action=usr_args.get("wam_expected_world_to_action"),
        coflow_inference_mode=usr_args.get("coflow_inference_mode"),
        coflow_inference_horizon=usr_args.get("coflow_inference_horizon"),
        coflow_inference_seed=usr_args.get("coflow_inference_seed"),
    )


__all__ = ["FastWAMRobotWinModelClient", "eval", "get_model", "reset_model"]
