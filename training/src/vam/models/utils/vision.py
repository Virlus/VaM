"""Image encoder construction and observation token helpers."""

import einops
import jax.numpy as jnp

from vam.models import model as _model
from vam.models import siglip as _siglip
from vam.models import siglip_video as _siglip_video
from vam.models import vam_config
from vam.models.utils.position_embedding import posemb_sincos_1d_identity_current


def _make_image_encoder(
    *,
    num_classes: int,
    dtype_mm: str,
    encoder_type: vam_config.ImageEncoderType,
    temporal_attention_interval: int,
):
    encoder_kwargs = {
        "num_classes": num_classes,
        "variant": "So400m/14",
        "pool_type": "none",
        "scan": True,
        "dtype_mm": dtype_mm,
    }
    if encoder_type == "siglip":
        return _siglip.Module(**encoder_kwargs)
    if encoder_type == "siglip_video":
        return _siglip_video.Module(
            **encoder_kwargs,
            temporal_attention_interval=temporal_attention_interval,
        )
    raise ValueError(f"Unsupported image_encoder_type={encoder_type}")


def _siglip_init_image(image, *, encoder_type: vam_config.ImageEncoderType):
    if encoder_type == "siglip" and image.ndim == 5:
        return image.reshape((-1, *image.shape[-3:]))
    return image


def _embed_image_tokens(
    img,
    image,
    *,
    encoder_type: vam_config.ImageEncoderType,
    image_mask=None,
    train: bool = False,
):
    if encoder_type == "siglip_video":
        return img(image, image_mask, train=train)
    if image.ndim == 5:
        batch_size, history_len = image.shape[:2]
        image_tokens, out = img(image.reshape((-1, *image.shape[-3:])), train=train)
        return image_tokens.reshape((batch_size, history_len * image_tokens.shape[1], image_tokens.shape[-1])), out
    return img(image, train=train)


def _image_token_mask(
    image_mask,
    image_tokens,
    *,
    encoder_type: vam_config.ImageEncoderType,
):
    if encoder_type == "siglip_video":
        if image_mask.ndim == 2:
            image_mask = image_mask[:, -1]
        return einops.repeat(image_mask, "b -> b s", s=image_tokens.shape[1])
    if image_mask.ndim == 2:
        per_frame_tokens = image_tokens.shape[1] // image_mask.shape[1]
        return einops.repeat(image_mask, "b t -> b (t s)", s=per_frame_tokens)
    return einops.repeat(image_mask, "b -> b s", s=image_tokens.shape[1])


def _state_tokens(state_proj, obs: _model.Observation):
    state_tokens = state_proj(obs.state)
    if state_tokens.ndim == 2:
        state_tokens = state_tokens[:, None, :]
        state_mask = jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_)
    else:
        state_tokens = state_tokens + posemb_sincos_1d_identity_current(state_tokens.shape[1], state_tokens.shape[-1])[
            :, :, 0, :
        ].astype(state_tokens.dtype)
        state_mask = (
            jnp.asarray(obs.state_mask)
            if obs.state_mask is not None
            else jnp.ones(obs.state.shape[:-1], dtype=jnp.bool_)
        )
    ar_mask = [True] + ([False] * (state_tokens.shape[1] - 1))
    return state_tokens, state_mask, ar_mask
