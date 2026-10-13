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

"""Gemma token generation, rotary positions and expert-layer helpers."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from vam.shared import array_typing as at


@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts left-aligned valid tokens to right-aligned valid tokens."""
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


def generate_tokens(
    llm,
    prefix_embeddings,
    prefix_mask,
    prefix_attn_mask,
    *,
    max_tokens: int,
    temperature: float = 0.0,
    rng: at.KeyArrayLike,
    eos_token: int = 1,
    token_projection: Callable[[jax.Array], jax.Array] | None = None,
):
    """Autoregressively generate tokens from a single-expert Gemma prefix."""
    prefix_embeddings, prefix_mask, prefix_attn_mask = left_to_right_align(
        prefix_embeddings, prefix_mask, prefix_attn_mask
    )
    prefill_size = prefix_embeddings.shape[1]
    prefill_len = jnp.sum(prefix_mask, axis=-1)
    prefix_start = prefill_size - prefill_len

    prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_tokens)))
    prefix_positions = jnp.cumsum(prefix_mask, axis=-1) - 1
    prefill_out, kv_cache = llm(
        [prefix_embeddings],
        mask=prefix_attn_mask,
        positions=prefix_positions,
        decode=True,
    )

    last_logits = llm(prefill_out[0][:, -1:, :], method="decode")
    output_tokens = jnp.zeros((last_logits.shape[0], max_tokens), dtype=jnp.int32)

    def step(carry):
        rng, last_logits, output_tokens, cache, _, step = carry
        rng, rng_step = jax.random.split(rng)
        token = jax.lax.cond(
            temperature > 0.0,
            lambda _: jax.random.categorical(rng_step, last_logits / temperature, axis=-1),
            lambda _: jnp.argmax(last_logits, axis=-1),
            operand=None,
        ).astype(jnp.int32)
        output_tokens = put_along_last_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

        has_eos = jnp.any(token == eos_token, axis=-1)
        all_eos = jnp.all(has_eos)
        token_embedding = llm(token, method="embed")
        if token_projection is not None:
            token_embedding = token_projection(token_embedding)

        positions = prefill_len[:, None] + step + 1
        mask = jnp.logical_and(
            jnp.arange(prefill_size + max_tokens)[None, None, :] >= prefix_start[:, None, None],
            jnp.arange(prefill_size + max_tokens)[None, None, :]
            < (jnp.broadcast_to(prefill_size + step + 1, (prefix_start.shape[0], 1, 1))),
        )
        hidden_states, kv_cache = llm(
            [token_embedding],
            mask=mask,
            positions=positions,
            kv_cache=cache,
            decode=True,
        )
        last_logits = llm(hidden_states[0], method="decode")
        return rng, last_logits, output_tokens, kv_cache, all_eos, step + 1

    def cond(carry):
        _, _, _, _, all_eos, step = carry
        return (~all_eos) & (step < max_tokens)

    _, _, output_tokens, _, _, _ = jax.lax.while_loop(
        cond,
        step,
        (rng, last_logits, output_tokens, kv_cache, False, 0),
    )
    return output_tokens


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
