"""Checkpoint loading and sharded value inference used by training data annotation."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from flax import nnx
import jax
import numpy as np

from vam.models import model as _model
from vam.shared import array_typing as at
from vam.training import config as _config
from vam.training import sharding
from vam.training import utils as training_utils
from vam.training import weight_loaders as _weight_loaders

logger = logging.getLogger(__name__)


def load_model_fsdp(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    ckpt_path: str,
    mesh: jax.sharding.Mesh | None,
) -> nnx.Module:
    """Load model parameters and optionally distribute them with FSDP shardings."""
    _, model_rng = jax.random.split(init_rng)
    model = config.model.create(model_rng)
    params = nnx.state(model)
    loaded_params = training_utils.load_and_validate_partial_params(
        _weight_loaders.CheckpointWeightLoader(str(training_utils.resolve_checkpoint_params_path(ckpt_path))),
        params.to_pure_dict(),
    )

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)

    params = nnx.state(model)
    params = training_utils.cast_frozen_params_to_bfloat16(params, config.freeze_filter)
    graphdef = nnx.graphdef(model)
    model = nnx.merge(graphdef, params)

    graphdef, state = nnx.split(model)
    pure_dict = state.to_pure_dict()
    shapes = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), pure_dict)

    if mesh is None:
        device = jax.devices()[0]
        single_sharding = jax.sharding.SingleDeviceSharding(device)
        dict_sharding = jax.tree.map(lambda _: single_sharding, shapes)
        logger.info("Single-device sharding applied (device=%s)", device)
    else:
        dict_sharding = sharding.fsdp_sharding(shapes, mesh, log=True)

    sharded_dict = jax.device_put(pure_dict, dict_sharding)
    state.replace_by_pure_dict(sharded_dict)
    return nnx.merge(graphdef, state)


@dataclasses.dataclass(frozen=True)
class ValueDecodeConfig:
    discrete_value: bool
    n_bins: int
    vmin: float
    vmax: float


def make_sharded_predict_value(
    model: nnx.Module,
    mesh: jax.sharding.Mesh,
    *,
    predict_value_kwargs: dict[str, Any] | None = None,
):
    """Create a JIT-compiled predict_value with proper FSDP sharding.

    Unlike module_jit (which has no sharding), this leverages committed arrays:
    the model state has FSDP shardings from jax.device_put, and the observation
    data has DATA_AXIS sharding from jax.make_array_from_process_local_data.
    JAX infers the shardings automatically from these committed arrays.

    The mesh must be set (via sharding.set_mesh) during the first call so that
    activation_sharding_constraint inside the model can apply properly.

    Args:
        model: The model with FSDP-sharded parameters (from load_model_fsdp).
        mesh: The FSDP mesh (needed for activation_sharding_constraint).
    """
    graphdef, state = nnx.split(model)
    predict_value_kwargs = dict(predict_value_kwargs or {})

    def _predict_value(state: nnx.State, observation: _model.Observation, actions: _model.Actions):
        module = nnx.merge(graphdef, state)
        return module.predict_value(observation, actions, **predict_value_kwargs)

    # JAX infers in/out shardings from committed arrays (state + observation).
    jitted_fn = jax.jit(_predict_value)

    def wrapper(observation: _model.Observation, actions: _model.Actions):
        # set_mesh is needed during the first call (JIT tracing) so that
        # activation_sharding_constraint inside the model can resolve the mesh.
        with sharding.set_mesh(mesh):
            return jitted_fn(state, observation, actions)

    return wrapper


def get_value_decode_config(model_config: _model.BaseModelConfig) -> ValueDecodeConfig:
    discrete_value = bool(getattr(model_config, "discrete_value", False))
    n_bins = int(getattr(model_config, "n_bins", 1))
    vmin = float(getattr(model_config, "vmin", 0.0))
    vmax = float(getattr(model_config, "vmax", 1.0))
    return ValueDecodeConfig(discrete_value=discrete_value, n_bins=n_bins, vmin=vmin, vmax=vmax)


def decode_value_predictions(predictions: Any, value_decode_config: ValueDecodeConfig) -> np.ndarray:
    values = np.asarray(predictions)
    if values.ndim == 0:
        raise ValueError("Expected batched value predictions, got scalar output.")

    if value_decode_config.discrete_value:
        support = np.linspace(
            value_decode_config.vmin,
            value_decode_config.vmax,
            value_decode_config.n_bins,
            dtype=np.float32,
        )
        return np.sum(support[np.newaxis, :] * values, axis=-1, dtype=np.float32)

    while values.ndim > 1 and values.shape[-1] == 1:
        values = np.squeeze(values, axis=-1)

    return values.astype(np.float32, copy=False)
