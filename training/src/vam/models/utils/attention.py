# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token and temporal attention masks."""

import jax.numpy as jnp


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    if mask_ar.ndim == 1:
        mask_ar = mask_ar.astype(jnp.int32)
        cumsum = jnp.cumsum(mask_ar, axis=0)
        attn_mask = cumsum[None, None, :] <= cumsum[None, :, None]
    else:
        mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape).astype(jnp.int32)
        cumsum = jnp.cumsum(mask_ar, axis=1)
        attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = jnp.logical_and(input_mask[:, None, :], input_mask[:, :, None])
    return jnp.logical_and(attn_mask, valid_mask)


def make_temporal_causal_attention_mask(temporal_mask, num_tokens: int):
    """Builds causal temporal masks for independent same-position patch histories."""
    temporal_mask = jnp.asarray(temporal_mask, dtype=jnp.bool_)
    if temporal_mask.ndim != 2:
        raise ValueError(f"Expected temporal_mask with rank 2, got shape {temporal_mask.shape}")
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")

    batch_size, history_len = temporal_mask.shape
    time_index = jnp.arange(history_len)
    causal_mask = time_index[:, None] >= time_index[None, :]
    temporal_attn_mask = jnp.logical_and(causal_mask[None, :, :], temporal_mask[:, None, :])
    temporal_attn_mask = jnp.broadcast_to(
        temporal_attn_mask[:, None, :, :], (batch_size, num_tokens, history_len, history_len)
    )
    return jnp.reshape(temporal_attn_mask, (batch_size * num_tokens, 1, history_len, history_len))
