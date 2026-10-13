from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from vam_client import base_policy as _base_policy
import torch
from typing_extensions import override

from vam import transforms as _transforms
from vam.models import model as _model
from vam.shared import array_typing as at
from vam.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        predict_value_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            predict_value_kwargs: Additional keyword arguments to pass to model.predict_value.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._value_output_transform = _transforms.compose(
            [tf for tf in output_transforms if "normalize" not in type(tf).__name__]
        )
        self._sample_kwargs = sample_kwargs or {}
        self._predict_value_kwargs = predict_value_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            if hasattr(model, "predict_value"):
                self._predict_value = model.predict_value
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions,
                static_argnames=("enable_rtc", "cfg_guidance_weight", "prefix_attention_schedule"),
            )
            self._rng = rng or jax.random.key(0)
            if hasattr(model, "predict_value"):
                self._predict_value = nnx_utils.module_jit(model.predict_value)

        # Store previous actions for prefix_actions functionality
        self._prev_actions: np.ndarray | None = None

    def _shift_prefix_actions(self, action_chunk: np.ndarray, time_base: int) -> np.ndarray:
        """
        Shift the transformed joint-action prefix past the elapsed control steps.
        This function converts all actions to relative increments from time_base,
        extracts the remainder of actions starting from time_base,
        and pads with zeros to maintain original length.

        Args:
            action_chunk: Action chunk with shape (action_horizon, action_dim)
            time_base: Index in the original chunk that corresponds to current state (max(k_robot_orig, k_gripper_orig))

        Returns:
            Shifted action chunk with same shape as input, where all actions are relative increments from time_base
        """
        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with 2 dims, got shape {action_chunk.shape}")

        time_base = max(time_base, 0)
        if time_base >= action_chunk.shape[0]:
            raise ValueError(f"time_base ({time_base}) must be less than action chunk length ({action_chunk.shape[0]})")

        # Convert all actions to relative increments from time_base
        # Extract remainder starting from time_base
        remainder = action_chunk[time_base + 1 :]
        # Pad zeros on the right side to maintain the original length
        pad_length = action_chunk.shape[0] - remainder.shape[0]
        if pad_length > 0:
            pad = np.zeros((pad_length, action_chunk.shape[1]), dtype=action_chunk.dtype)
            return np.concatenate([remainder, pad], axis=0)
        return remainder

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        time_base: int | None = None,
        rtc_config: dict[str, float | int | str] | None = None,
        enable_rtc: bool = False,
    ) -> dict:  # type: ignore[misc]
        """Infer one chunk, optionally using the previous chunk as an RTC prefix.

        enable_rtc=True requires time_base and rtc_config. The first call after
        reset uses ordinary sampling to seed the prefix; later calls apply RTC.
        """
        if enable_rtc:
            if time_base is None or rtc_config is None:
                raise ValueError("enable_rtc=True requires time_base and rtc_config.")
            required = ("inference_delay", "prefix_attention_horizon", "max_guidance_weight")
            if any(rtc_config.get(key) is None for key in required):
                raise ValueError(f"rtc_config must contain {', '.join(required)}")
        use_rtc = enable_rtc and self._prev_actions is not None

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare the shared sampler arguments.
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        # Prepare prefix_actions if time_base is provided and we have previous actions
        prefix_actions: np.ndarray | None = None
        if use_rtc and self._prev_actions is not None:
            prefix_actions = np.asarray(self._prev_actions)
            if prefix_actions.ndim == 1:
                prefix_actions = prefix_actions[None, ...]

            # Map original obs (absolute) for reuse
            rtc_inputs = jax.tree.map(lambda x: x, obs)
            state_ref = np.asarray(rtc_inputs["observation/state"])

            rtc_inputs["actions"] = prefix_actions[None, :]
            # Rebase stored absolute joint targets against the current qpos using training transforms.
            rtc_inputs["observation/state"] = state_ref

            rtc_inputs = self._input_transform(rtc_inputs)
            prefix_actions = rtc_inputs["actions"]
            prefix_actions = self._shift_prefix_actions(prefix_actions, time_base)

        sample_kwargs["enable_rtc"] = use_rtc
        if use_rtc:
            prefix_actions_tensor = (
                torch.from_numpy(np.asarray(prefix_actions)).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(prefix_actions)
            )
            sample_kwargs.update(
                prefix_actions=prefix_actions_tensor,
                inference_delay=rtc_config["inference_delay"],
                prefix_attention_horizon=rtc_config["prefix_attention_horizon"],
                max_guidance_weight=rtc_config["max_guidance_weight"],
                prefix_attention_schedule=rtc_config.get(
                    "prefix_attention_schedule", sample_kwargs.get("prefix_attention_schedule", "exp")
                ),
            )

        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if use_rtc and prefix_actions is not None and time_base != 0:
            outputs2 = self._output_transform({"state": inputs["state"], "actions": prefix_actions[None, :]})
            outputs["prefix_actions"] = outputs2["actions"]

        if "actions" in outputs:
            actions_to_save = outputs["actions"]
            # Ensure 2D shape (action_horizon, action_dim)
            self._prev_actions = actions_to_save.copy()

        return outputs

    @override
    def infer_value(self, obs: dict, predicted_action: torch.Tensor) -> dict:
        obs["actions"] = predicted_action  # Add predicted action to obs for Q prediction
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)

        observation = _model.Observation.from_dict(inputs)
        actions = inputs["actions"]
        value = self._predict_value(observation, actions, **self._predict_value_kwargs)
        outputs = {"value": value}

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._value_output_transform(outputs)
        return outputs

    @override
    def infer_batch(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)
        sample_kwargs["enable_rtc"] = False
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        model_output = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        if isinstance(model_output, tuple):
            actions, value = model_output
        elif hasattr(self._model, "predict_value"):
            actions = model_output
            value = self._predict_value(observation, **self._predict_value_kwargs)
        else:
            actions = model_output
            value = None
        outputs = {
            "state": inputs["state"],
            "actions": actions,
            "value": value,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x.detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @override
    def extract_visual_embeddings(
        self,
        obs: dict,
        *,
        batch_sharding: jax.sharding.Sharding | None = None,
    ) -> np.ndarray:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:

            def _to_device_array(value):
                if isinstance(value, jax.Array):
                    return value
                if isinstance(value, np.ndarray):
                    if (
                        batch_sharding is not None
                        and value.ndim >= 1
                        and (np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_))
                    ):
                        try:
                            return jax.device_put(value, batch_sharding)
                        except ValueError:
                            # Fallback to default device placement if sharding is incompatible
                            pass
                    return jnp.asarray(value)
                return value

            inputs = jax.tree.map(_to_device_array, inputs)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device), inputs)

        observation = _model.Observation.from_dict(inputs)
        visual_embeddings = self._model.extract_visual_embeddings(observation)
        visual_embeddings = np.asarray(visual_embeddings)
        return visual_embeddings

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
