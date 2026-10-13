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

"""Spatial, temporal and diffusion-time position embeddings."""

import jax
import jax.numpy as jnp

from vam.shared import array_typing as at


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def posemb_sincos_1d_identity_current(history_len: int, width: int, temperature: float = 10_000.0):
    """Sinusoidal temporal embedding where the current frame receives exactly zero."""
    if history_len < 1:
        raise ValueError(f"history_len must be >= 1, got {history_len}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")

    distances = jnp.arange(-(history_len - 1), 1, dtype=jnp.float32)
    half_width = max(width // 2, 1)
    if half_width == 1:
        omega = jnp.ones((1,), dtype=jnp.float32)
    else:
        omega = jnp.arange(half_width, dtype=jnp.float32) / (half_width - 1)
        omega = 1.0 / (temperature**omega)
    phase = jnp.einsum("t,d->td", distances, omega, precision=jax.lax.Precision.HIGHEST)
    posemb = jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], axis=-1)
    posemb = posemb[:, :width]
    if posemb.shape[-1] < width:
        posemb = jnp.pad(posemb, ((0, 0), (0, width - posemb.shape[-1])))
    posemb = jnp.where(distances[:, None] == 0.0, jnp.zeros_like(posemb), posemb)
    return posemb[None, :, None, :]
