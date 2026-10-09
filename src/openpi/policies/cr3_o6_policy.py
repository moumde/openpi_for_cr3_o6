import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    """Convert LeRobot image format to OpenPI image format.

    LeRobot video decoding returns images as:
        torch.Tensor / numpy.ndarray
        [C, H, W]
        float32 in [0, 1]

    OpenPI expects:
        numpy.ndarray
        [H, W, C]
        uint8 in [0, 255]
    """
    image = np.asarray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


@dataclasses.dataclass(frozen=True)
class CR3O6Inputs(transforms.DataTransformFn):
    """Input transform for Dobot CR3 + LinkerHand O6.

    Expected input after RepackTransform:

        data = {
            "images": {
                "base_0_rgb": ...,
                "left_wrist_0_rgb": ...,
                "right_wrist_0_rgb": ...,
            },
            "state": [12],
            "actions": [action_horizon, 12],   # training only
            "prompt": str,
        }

    State/action layout:

        0:  CR3 q1
        1:  CR3 q2
        2:  CR3 q3
        3:  CR3 q4
        4:  CR3 q5
        5:  CR3 q6
        6:  O6 thumb_flex
        7:  O6 thumb_yaw
        8:  O6 index_flex
        9:  O6 middle_flex
        10: O6 ring_flex
        11: O6 little_flex

    No robot-specific coordinate conversion is performed here.
    The values are already in the representation used by the
    CR3/O6 controller.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]

        unexpected = set(in_images) - set(self.EXPECTED_CAMERAS)
        if unexpected:
            raise ValueError(
                f"Unexpected camera names: {unexpected}. "
                f"Expected cameras: {self.EXPECTED_CAMERAS}"
            )

        if "base_0_rgb" not in in_images:
            raise ValueError(
                "CR3O6 policy requires base_0_rgb camera."
            )

        # Convert LeRobot CHW Tensor/array to OpenPI HWC uint8.
        base_image = _parse_image(in_images["base_0_rgb"])

        images = {
            "base_0_rgb": base_image,
        }

        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Wrist cameras are optional at the transform level.
        # If one is missing, provide a black image and mark it invalid.
        for camera_name in (
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ):
            if camera_name in in_images:
                images[camera_name] = _parse_image(
                    in_images[camera_name]
                )
                image_masks[camera_name] = np.True_
            else:
                images[camera_name] = np.zeros_like(base_image)
                image_masks[camera_name] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
        }

        # Actions only exist during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])

            if actions.shape[-1] != 12:
                raise ValueError(
                    f"Expected 12D CR3+O6 actions, "
                    f"got shape {actions.shape}"
                )

            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CR3O6Outputs(transforms.DataTransformFn):
    """Output transform for Dobot CR3 + LinkerHand O6.

    Pi05 uses a 32-dimensional action head, while the physical CR3/O6
    dataset and controller use 12 dimensions.  The model transform pads the
    12-dimensional training action to the model width; here we remove those
    padding dimensions and return the physical action contract.

    The first six values are CR3 absolute joint targets in radians.  The last
    six values are O6 absolute register targets in the native 0..255 range.
    No additional scaling is applied because ``data_v21`` stores O6 values in
    that native range and its normalization statistics use the same units.
    """

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)

        if actions.ndim == 0 or actions.shape[-1] < 12:
            raise ValueError(
                f"Expected model actions with at least 12 dimensions, got shape {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("CR3/O6 model actions contain NaN or Inf")

        return {
            "actions": actions[..., :12].copy(),
        }
