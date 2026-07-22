"""RoboDojo simulation data registration for StarVLA.

The released LeRobot v2.1 dataset stores one 14-D vector in the order
``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.  Keep that
order explicit here: RobotWin's historical ARX-X5 config concatenates the same
groups in a different order and therefore must not be reused.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RoboDojoArxX5DataConfig:
    """Three source views, one FastWAM composite, state, and a 16-step chunk."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]
    state_keys = [
        "state.left_joints",
        "state.left_gripper",
        "state.right_joints",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.left_gripper",
        "action.right_joints",
        "action.right_gripper",
    ]
    state_key_dims = {
        "state.left_joints": 6,
        "state.left_gripper": 1,
        "state.right_joints": 6,
        "state.right_gripper": 1,
    }
    action_key_dims = {
        "action.left_joints": 6,
        "action.left_gripper": 1,
        "action.right_joints": 6,
        "action.right_gripper": 1,
    }
    modality_key_ranges = {
        "state": {
            "left_joints": (0, 6),
            "left_gripper": (6, 7),
            "right_joints": (7, 13),
            "right_gripper": (13, 14),
        },
        "action": {
            "left_joints": (0, 6),
            "left_gripper": (6, 7),
            "right_joints": (7, 13),
            "right_gripper": (13, 14),
        },
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        # Match the FastWAM data ABI exactly: normalize every continuous state
        # and action component with (x - mean) / (std + 1e-8), clipped to
        # [-5, 5].  The simulated grippers remain continuous; do not threshold
        # them as binary joints.  PolicyNormProcessor reuses this same transform
        # at inference, so action denormalization stays checkpoint-consistent.
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "fastwam_zscore" for key in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "fastwam_zscore" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "robodojo_arx_x5": RoboDojoArxX5DataConfig(),
}

# The central registry derives this map from each DataConfig's class variable.
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "robodojo_v21": [
        ("RoboDojo_lerobot_v21_video", 1.0, "robodojo_arx_x5"),
    ],
    # One dataset contains both annotated and unannotated tasks. Optional
    # per-frame subtask fields are masked sample-by-sample by the joint loader.
    "robodojo_v21_language_optional": [
        ("RoboDojo_lerobot_v21_language_v1", 1.0, "robodojo_arx_x5"),
    ],
}
