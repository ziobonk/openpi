"""
Dual-arm Piper data mapping for openpi — converts dual Piper observations/actions
to the standard openpi model format and back.

Model expects 3 image slots:
    base_0_rgb        ← global camera
    left_wrist_0_rgb  ← left wrist camera
    right_wrist_0_rgb ← right wrist camera

State:  [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]  = 14D
Actions: [left_pose(6), left_gripper(1), right_pose(6), right_gripper(1)]     = 14D

Usage:
Copy this file to /home/rhr/openpi/src/openpi/policies/dual_piper_policy.py
Then add the corresponding DataConfig in training/config.py.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_dual_piper_example() -> dict:
    """Create random input sample for dry-run testing."""
    return {
        "observation/state": np.random.rand(14).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the block with both arms",
    }


def _parse_image(image) -> np.ndarray:
    """Normalize image format: ensure uint8 (H, W, C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # (C, H, W) → (H, W, C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DualPiperInputs(transforms.DataTransformFn):
    """Convert dual Piper observation dict to standard model Observation format.

    Expected input (after RepackTransform):
        observation/state:                float32 (14,)
            [left_j1..j6(rad), left_gripper(m), right_j1..j6(rad), right_gripper(m)]
        observation/image:                uint8 (H, W, 3)  global / base camera
        observation/wrist_image_left:     uint8 (H, W, 3)  left wrist camera
        observation/wrist_image_right:    uint8 (H, W, 3)  right wrist camera
        prompt:                           str              language instruction

    Output:
        Standard openpi Observation dict (state, image, image_mask, prompt, actions).
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/wrist_image_left"])
        right_wrist_image = _parse_image(data["observation/wrist_image_right"])

        # All three camera slots are real (not padded zeros)
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, left_wrist_image, right_wrist_image)
        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Actions only present during training
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # Prompt
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = str(data["prompt"])

        return inputs


@dataclasses.dataclass(frozen=True)
class DualPiperOutputs(transforms.DataTransformFn):
    """Convert model action chunk to dual Piper executable actions.

    Model outputs 32-dim actions. We extract the first `dual_piper_action_dim` dims.

    Output:
        actions: float32 (action_horizon, dual_piper_action_dim)
            [left_pose(6), left_gripper(1), right_pose(6), right_gripper(1)]
    """

    dual_piper_action_dim: int = 14  # 6+1+6+1

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :self.dual_piper_action_dim])}
