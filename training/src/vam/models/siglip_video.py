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

"""Video SigLIP encoder with interleaved spatial and temporal attention."""

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from vam.models import siglip as _siglip
from vam.models.utils.attention import make_temporal_causal_attention_mask
from vam.models.utils.position_embedding import posemb_sincos_1d_identity_current
from vam.models.utils.siglip import decode_variant
from vam.models.utils.siglip import get_posemb


class TemporalEncoder1DBlock(nn.Module):
    """Single temporal transformer block with an optional frame-validity mask."""

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):  # noqa: FBT002
        out = {}
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y, mask=mask)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = _siglip.MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        return x, out


class VideoEncoder(nn.Module):
    """Transformer encoder with spatial blocks and periodic temporal attention."""

    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    temporal_attention_interval: int = 4
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, temporal_mask=None, deterministic=True):  # noqa: FBT002
        if self.temporal_attention_interval < 1:
            raise ValueError(f"temporal_attention_interval must be >= 1, got {self.temporal_attention_interval}")

        out = {}
        batch_size, history_len, num_tokens, width = x.shape
        if temporal_mask is None:
            temporal_mask = jnp.ones((batch_size, history_len), dtype=jnp.bool_)
        temporal_mask = jnp.asarray(temporal_mask, dtype=jnp.bool_)
        for lyr in range(self.depth):
            if (lyr + 1) % self.temporal_attention_interval == 0:
                temporal_x = jnp.transpose(x, (0, 2, 1, 3))
                temporal_x = jnp.reshape(temporal_x, (batch_size * num_tokens, history_len, width))
                temporal_attn_mask = make_temporal_causal_attention_mask(temporal_mask, num_tokens)
                temporal_x, out[f"block{lyr:02d}"] = TemporalEncoder1DBlock(
                    name=f"temporal_encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )(temporal_x, temporal_attn_mask, deterministic)
                temporal_x = jnp.reshape(temporal_x, (batch_size, num_tokens, history_len, width))
                x = jnp.transpose(temporal_x, (0, 2, 1, 3))
            else:
                spatial_x = jnp.reshape(x, (batch_size * history_len, num_tokens, width))
                spatial_x, out[f"block{lyr:02d}"] = _siglip.Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )(spatial_x, deterministic)
                x = jnp.reshape(spatial_x, (batch_size, history_len, num_tokens, width))

        out["pre_ln"] = x
        x = nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x)
        return x, out


class _Module(nn.Module):
    """ViT model that accepts either images or image histories."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None
    num_heads: int = 12
    posemb: str = "learn"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"
    head_zeroinit: bool = True
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    temporal_attention_interval: int = 4

    @nn.compact
    def __call__(self, image, image_mask=None, *, train=False):
        out = {}

        image = jnp.asarray(image, jnp.float32)
        if image.ndim == 4:
            image = image[:, None, ...]
            if image_mask is not None and jnp.asarray(image_mask).ndim == 1:
                image_mask = jnp.asarray(image_mask)[:, None]
        if image.ndim != 5:
            raise ValueError(f"Expected image with rank 4 or 5, got shape {image.shape}")

        batch_size, history_len = image.shape[:2]
        if image_mask is None:
            image_mask = jnp.ones((batch_size, history_len), dtype=jnp.bool_)
        else:
            image_mask = jnp.asarray(image_mask, dtype=jnp.bool_)
            if image_mask.ndim == 1:
                image_mask = image_mask[:, None]
        flat_image = jnp.reshape(image, (batch_size * history_len, *image.shape[-3:]))

        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(flat_image)

        _, h, w, c = x.shape
        x = jnp.reshape(x, (batch_size, history_len, h * w, c))
        spatial_posemb = get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)
        x = x + spatial_posemb[:, None, :, :]
        if history_len > 1:
            x = x + posemb_sincos_1d_identity_current(history_len, c)
        out["with_posemb"] = x

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, (batch_size, history_len, 1, 1)), x], axis=2)

        x = nn.Dropout(rate=self.dropout)(x, not train)
        x = x.astype(self.dtype_mm)

        if history_len == 1:
            x, out["encoder"] = _siglip.Encoder(
                depth=self.depth,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                scan=self.scan,
                remat_policy=self.remat_policy,
                dtype_mm=self.dtype_mm,
                name="Transformer",
            )(x[:, 0], deterministic=not train)
            x = x[:, None]
        else:
            x, out["encoder"] = VideoEncoder(
                depth=self.depth,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                temporal_attention_interval=self.temporal_attention_interval,
                dtype_mm=self.dtype_mm,
                name="Transformer",
            )(x, image_mask, deterministic=not train)
        encoded = out["encoded"] = x[:, -1]

        if self.pool_type == "map":
            x = out["head_input"] = _siglip.MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype_mm=self.dtype_mm,
            )(encoded)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(encoded, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = encoded[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = encoded[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            x = encoded
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, (batch_size, h, w, -1))

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})
