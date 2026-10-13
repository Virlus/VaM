# GRU-based history encoder for injecting state/value history as prefix tokens.

from typing import Optional

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from vam.shared import array_typing as at


class HistoryGRU(nnx.Module):
    """Small GRU encoder without RNG state in the module tree.

    Designed as a parameter-only NNX module so EMA and checkpoint code only
    see numeric model parameters.
    """

    def __init__(self, width: int, rngs: nnx.Rngs):
        self.dense_i = nnx.Linear(width, 3 * width, rngs=rngs)
        self.dense_h = nnx.Linear(width, 3 * width, use_bias=False, rngs=rngs)
        self.width = width

    def __call__(
        self, inputs: at.Float[at.Array, "b t d"], mask: Optional[at.Bool[at.Array, "b t"]] = None
    ) -> at.Float[at.Array, "b d"]:
        batch_size = inputs.shape[0]
        hidden = jnp.zeros((batch_size, self.width), dtype=inputs.dtype)
        time_major_inputs = jnp.swapaxes(inputs, 0, 1)
        time_major_mask = None if mask is None else jnp.swapaxes(mask.astype(jnp.bool_), 0, 1)

        def step(carry, step_input):
            if time_major_mask is None:
                x = step_input
                valid = None
            else:
                x, valid = step_input
            i_r, i_z, i_n = jnp.split(self.dense_i(x), 3, axis=-1)
            h_r, h_z, h_n = jnp.split(self.dense_h(carry), 3, axis=-1)
            reset = nnx.sigmoid(i_r + h_r)
            update = nnx.sigmoid(i_z + h_z)
            candidate = jnp.tanh(i_n + reset * h_n)
            next_carry = (1.0 - update) * candidate + update * carry
            if valid is not None:
                next_carry = jnp.where(valid[:, None], next_carry, carry)
            return next_carry, None

        scan_input = time_major_inputs if time_major_mask is None else (time_major_inputs, time_major_mask)
        hidden, _ = jax.lax.scan(step, hidden, scan_input)
        return hidden
