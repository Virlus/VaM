"""Checkpoint and temporal-mask contracts across the extracted model utilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vam.models import siglip
from vam.models import siglip_video


def tiny_encoder(module):
    return module.Module(
        variant="mu/2",
        num_classes=8,
        pool_type="none",
        scan=False,
        temporal_attention_interval=1,
        head_zeroinit=False,
    )


def test_video_single_frame_loads_image_encoder_parameters():
    """Single-frame video and image modules must share the existing checkpoint tree."""
    image = jax.random.normal(jax.random.key(1), (1, 4, 4, 3))
    standard = siglip.Module(variant="mu/2", num_classes=8, pool_type="none", scan=False, head_zeroinit=False)
    variables = standard.init(jax.random.key(0), image)
    video = tiny_encoder(siglip_video)
    expected, _ = standard.apply(variables, image)
    actual, _ = video.apply(variables, image[:, None], jnp.ones((1, 1), dtype=bool))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("pool_type", ["none", "gap"])
def test_video_padding_does_not_change_current_observation(pool_type):
    """Invalid history frames cannot influence the current frame through temporal attention."""
    encoder = tiny_encoder(siglip_video).clone(pool_type=pool_type)
    images = jax.random.normal(jax.random.key(1), (1, 3, 4, 4, 3))
    mask = jnp.array([[False, True, True]])
    variables = encoder.init(jax.random.key(0), images, mask)
    expected, _ = encoder.apply(variables, images, mask)
    actual, _ = encoder.apply(variables, images.at[:, 0].set(1000.0), mask)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
