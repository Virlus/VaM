"""History slicing, validity masks and GRU token projection."""

from typing import Optional

import flax.nnx as nnx
import jax.numpy as jnp

from vam.models import model as _model
from vam.models.gru import HistoryGRU
from vam.shared import array_typing as at


def history_gru_token(
    input_proj: nnx.Linear,
    history_rnn: HistoryGRU,
    values: Optional[at.Float[at.Array, "b t d"]],
    mask: Optional[at.Bool[at.Array, "b t"]],
) -> at.Float[at.Array, "b 1 d"]:
    """Project a history sequence through a GRU and return the final hidden state as one token."""
    if values is None:
        raise ValueError("History-token conditioning requires the corresponding observation field.")
    if values.ndim == 2:
        values = values[:, None, :]
    projected = input_proj(values)
    if mask is not None:
        if mask.ndim == 1:
            mask = mask[:, None]
    carry = history_rnn(projected, mask)
    return carry[:, None, :]


def current_history_mask(
    mask: Optional[at.Bool[at.Array, "b"] | at.Bool[at.Array, "b t"]],  # noqa: F821 - jaxtyping shape
    values: at.Float[at.Array, "b ..."],
) -> at.Bool[at.Array, "b"]:  # noqa: F821 - jaxtyping shape
    """Return a per-sample mask reflecting whether the current (last) frame is valid."""
    if mask is None:
        return jnp.ones(values.shape[:1], dtype=jnp.bool_)
    if mask.ndim == 1:
        return mask
    return mask[:, -1]


def _slice_image_history(image, horizon: int):
    if image.ndim == 5:
        return image[:, -horizon:]
    return image


def _slice_observation_history(obs: _model.Observation, horizon: int) -> _model.Observation:
    if horizon <= 0:
        raise ValueError(f"Observation horizon must be positive, got {horizon}")
    if obs.state.ndim < 3:
        return obs

    def slice_history(x):
        if x is None or x.ndim < 2:
            return x
        return x[:, -horizon:]

    with at.disable_typechecking():
        return _model.Observation(
            images={key: slice_history(value) for key, value in obs.images.items()},
            image_masks={key: slice_history(value) for key, value in obs.image_masks.items()},
            state=slice_history(obs.state),
            state_mask=slice_history(obs.state_mask),
            value_prediction=slice_history(obs.value_prediction),
            value_prediction_mask=slice_history(obs.value_prediction_mask),
            tokenized_prompt=obs.tokenized_prompt,
            tokenized_prompt_mask=obs.tokenized_prompt_mask,
            tokenized_value_prompt=obs.tokenized_value_prompt,
            tokenized_value_prompt_mask=obs.tokenized_value_prompt_mask,
            tokenized_advantage=obs.tokenized_advantage,
            tokenized_advantage_mask=obs.tokenized_advantage_mask,
            token_ar_mask=obs.token_ar_mask,
            token_loss_mask=obs.token_loss_mask,
        )
