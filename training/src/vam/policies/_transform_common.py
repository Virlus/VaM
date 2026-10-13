"""Shared input/output transform primitives for robot policies.

These helpers were previously copy-pasted across the per-robot ``*_policy.py`` modules
(``_parse_image``/``_valid_mask``/``_invalid_mask_like`` had 11/5/3 near-identical copies).
Robot-specific kinematics and data-contract handling stay in each policy module; only the
generic mechanics live here.
"""

import dataclasses

import numpy as np

from vam import transforms


def parse_image(image) -> np.ndarray:
    """Normalize an image to HWC uint8 (channel-last), handling optional batch/history dims.

    Floats in [0, 1] are scaled to uint8. A channel-first layout (``... c h w`` with c==3) is
    rearranged to channel-last (``... h w c``). This is the superset of the older channel-first-only
    variants and is safe for real image shapes.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim >= 3 and image.shape[-3] == 3 and image.shape[-1] != 3:
        # ``... c h w -> ... h w c`` (works for 3-D CHW and batched/history 4-D+ tensors).
        image = np.moveaxis(image, -3, -1)
    return image


def valid_mask(data: dict, key: str) -> np.ndarray:
    """Return the per-frame validity mask for ``key`` derived from its ``<key>_is_pad`` column.

    Absent pad column -> scalar True (all valid).
    """
    pad_key = f"{key}_is_pad"
    if pad_key not in data:
        return np.True_
    return np.logical_not(np.asarray(data[pad_key], dtype=bool))


def invalid_mask_like(data: dict, key: str) -> np.ndarray:
    """Return an all-invalid mask shaped like ``valid_mask(data, key)``."""
    return np.zeros_like(valid_mask(data, key), dtype=bool)


def advantage_conditioning_token(advantage) -> str:
    """Map a scalar advantage to the pi06 conditioning prompt token.

    Centralizes the ``"Advantage:" + ("Positive"|"Negative")`` mapping that was copy-pasted across
    the async input transforms (and once mis-parenthesized — see the Tier-0 precedence fix).
    """
    return "Advantage:" + ("Positive" if advantage > 0.5 else "Negative")


@dataclasses.dataclass(frozen=True)
class BaseActionValueOutputs(transforms.DataTransformFn):
    """Convert model outputs back to the dataset format: sliced ``actions`` plus optional scalar ``value``.

    Shared by the async and RoboInter output transforms, which differ only in ``action_dim``.
    """

    action_dim: int

    def __call__(self, data: dict) -> dict:
        outputs = {}
        if "actions" in data:
            outputs["actions"] = np.asarray(data["actions"][:, : self.action_dim])
        if "value" in data and data["value"] is not None:
            outputs["value"] = np.asarray(data["value"][..., 0])
        if len(outputs) == 0:
            raise ValueError("No transformable outputs found")
        return outputs
