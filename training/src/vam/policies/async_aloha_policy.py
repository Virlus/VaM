import dataclasses

import numpy as np

from vam import transforms
from vam.models import model as _model
from vam.policies._transform_common import BaseActionValueOutputs
from vam.policies._transform_common import advantage_conditioning_token as _advantage_token
from vam.policies._transform_common import parse_image as _parse_image
from vam.policies._transform_common import valid_mask as _valid_mask


def make_async_aloha_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(14),
        # "observation/prev_state": np.random.rand(14),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AsyncAlohaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    action_horizon: int = 100

    def __call__(self, data: dict) -> dict:
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])
        base_image = _parse_image(data["observation/base_image"])
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": _valid_mask(data, "observation/base_image"),
                "left_wrist_0_rgb": _valid_mask(data, "observation/left_wrist_image"),
                "right_wrist_0_rgb": _valid_mask(data, "observation/right_wrist_image"),
            },
        }
        if "observation/state_is_pad" in data:
            inputs["state_mask"] = _valid_mask(data, "observation/state")
        if "value_prediction" in data:
            inputs["value_prediction"] = data["value_prediction"]
            if "value_prediction_is_pad" in data:
                inputs["value_prediction_mask"] = _valid_mask(data, "value_prediction")

        if data.get("actions") is not None:
            actions = data["actions"][0][: self.action_horizon]
            inputs["actions"] = actions

        if data.get("value") is not None:
            inputs["value"] = data["value"]

        if data.get("prompt") is not None:
            inputs["prompt"] = data["prompt"]

        if data.get("success") is not None:
            inputs["success"] = data["success"]

        if data.get("reward") is not None:
            inputs["reward"] = data["reward"]

        if data.get("control_flag") is not None:
            inputs["control_flag"] = data["control_flag"]

        if data.get("data_source") is not None:
            inputs["data_source"] = data["data_source"]

        if data.get("terminate") is not None:
            inputs["terminate"] = data["terminate"]

        if data.get("advantage") is not None and self.model_type == _model.ModelType.PI06:
            inputs["advantage"] = _advantage_token(data["advantage"])

        return inputs


@dataclasses.dataclass(frozen=True)
class AsyncAlohaOutputs(BaseActionValueOutputs):
    """Async Aloha model outputs -> dataset format (14-D actions + optional value)."""

    action_dim: int = 14


@dataclasses.dataclass(frozen=True)
class AsyncSequenceAlohaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    action_horizon: int = 100

    def __call__(self, data: dict) -> dict:
        # Because we are using a (s, a, s') triplet for TD learning / Filter BC.
        base_image = _parse_image(data["observation/base_image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                name: _valid_mask(data, key)
                for name, key in (
                    ("base_0_rgb", "observation/base_image"),
                    ("left_wrist_0_rgb", "observation/left_wrist_image"),
                    ("right_wrist_0_rgb", "observation/right_wrist_image"),
                )
            },
        }
        if "observation/next_state" in data:
            inputs["next_state"] = data["observation/next_state"]
            next_keys = {
                "base_0_rgb": "observation/next_base_image",
                "left_wrist_0_rgb": "observation/next_left_wrist_image",
                "right_wrist_0_rgb": "observation/next_right_wrist_image",
            }
            inputs["next_image"] = {name: _parse_image(data[key]) for name, key in next_keys.items()}
            inputs["next_image_mask"] = {name: _valid_mask(data, key) for name, key in next_keys.items()}
        if "observation/state_is_pad" in data:
            inputs["state_mask"] = _valid_mask(data, "observation/state")
        if "observation/next_state_is_pad" in data:
            inputs["next_state_mask"] = _valid_mask(data, "observation/next_state")
        if "value_prediction" in data:
            inputs["value_prediction"] = data["value_prediction"]
            if "value_prediction_is_pad" in data:
                value_prediction_mask = _valid_mask(data, "value_prediction")
                value_prediction_mask[-1] = False
                inputs["value_prediction_mask"] = value_prediction_mask

        if data.get("actions") is not None:
            actions = data["actions"][0][: self.action_horizon]
            inputs["actions"] = actions

        if data.get("next_actions") is not None:
            next_actions = data["next_actions"][0][: self.action_horizon]
            inputs["next_actions"] = next_actions

        if data.get("value") is not None:
            inputs["value"] = data["value"]

        if data.get("prompt") is not None:
            inputs["prompt"] = data["prompt"]

        if data.get("success") is not None:
            inputs["success"] = data["success"]

        if data.get("reward") is not None:
            inputs["reward"] = data["reward"]

        if data.get("control_flag") is not None:
            inputs["control_flag"] = data["control_flag"]

        if data.get("data_source") is not None:
            inputs["data_source"] = data["data_source"]

        if data.get("terminate") is not None:
            inputs["terminate"] = data["terminate"]

        if data.get("advantage") is not None and self.model_type == _model.ModelType.PI06:
            inputs["advantage"] = _advantage_token(data["advantage"])

        return inputs


@dataclasses.dataclass(frozen=True)
class AsyncSequenceAlohaOutputs(BaseActionValueOutputs):
    """Async Aloha (s, a, s') model outputs -> dataset format (14-D actions + optional value)."""

    action_dim: int = 14
