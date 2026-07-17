"""
Piper 机械臂的数据映射 — 负责 Piper ↔ openpi Observation/Actions 格式转换。

包含三个类:
  - PiperInputs : 原始 Piper observation dict → 模型标准 Observation
  - PiperOutputs: 模型 action chunk → Piper 可执行的动作
  - make_piper_example(): 生成随机输入样本（用于测试）
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_piper_example() -> dict:
    """创建随机输入样本（用于 dry-run 测试）。"""
    return {
        "observation/state": np.random.rand(7).astype(np.float32),  # 6 joints + 1 gripper
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }


def _parse_image(image) -> np.ndarray:
    """统一图像格式: 确保为 uint8 (H, W, C)。"""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # (C, H, W) → (H, W, C)
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """将 Piper observation dict 转换为模型标准的 Observation 格式。

    期望输入:
        observation/state:          float32 (7,)  [j1..j6(rad), gripper(raw)]
        observation/image:          uint8 (H, W, 3)  基座/外部相机
        observation/wrist_image:    uint8 (H, W, 3)  腕部相机 (可选，无则自动填零)
        prompt:                     str              语言指令

    输出:
        标准 openpi Observation dict (image, image_mask, state, prompt)。
    """

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data.get("observation/wrist_image", data["observation/image"]))

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)  # 遮住 padding
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)  # FAST 不 mask padding
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Action 仅训练时存在
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # Prompt
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = str(data["prompt"])

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """将模型输出的 action chunk 转换为 Piper 可执行的动作。

    模型输出 actions 维度默认为 32（config.action_dim），我们只取前
    `piper_action_dim` 维 (6 关节 + 1 夹爪 = 7)。

    输出:
        actions: float32 (action_horizon, piper_action_dim)
            [j1..j6(rad), gripper(raw_0.001mm)]
    """

    # Piper 的有效动作维度 (默认 7: 6 关节 + 1 夹爪)
    piper_action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.piper_action_dim])}
