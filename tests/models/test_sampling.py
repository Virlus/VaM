"""CPU comparisons against samplers from 0028766c, with no pretrained assets."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_vam_release import small_model

from vam.models.utils.rtc import get_prefix_weights
from vam.shared import nnx_utils

RTC = {
    "inference_delay": 1,
    "prefix_attention_horizon": 1,
    "max_guidance_weight": 2.0,
}


@pytest.fixture(scope="module", params=[1, 2])
def sampler(request):
    with pytest.MonkeyPatch.context() as patch:
        cfg, model = small_model(patch, request.param, td=False)
        model.pi06 = True
        obs = dataclasses.replace(
            cfg.fake_obs(),
            tokenized_advantage=jnp.ones((1, 1), dtype=jnp.int32),
            tokenized_advantage_mask=jnp.ones((1, 1), dtype=bool),
        )
        yield request.param, cfg, model, obs


@pytest.mark.parametrize("cfg_weight", [1.0, 2.0])
@pytest.mark.parametrize("mode", ["normal", "ones", "zeros", "linear", "exp"])
def test_previous_sampler_outputs(sampler, cfg_weight, mode):
    """Golden outputs cover ordinary/RTC, history, CFG and all taper schedules."""
    history, _, model, obs = sampler
    kwargs = {"num_steps": 3, "cfg_guidance_weight": cfg_weight}
    if mode != "normal":
        kwargs.update(
            enable_rtc=True,
            prefix_actions=jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 5,
            prefix_attention_schedule=mode,
            **RTC,
        )
    actual = model.sample_actions(jax.random.key(5), obs, **kwargs)
    with np.load(Path(__file__).parent / "fixtures/sampling_before_unification.npz") as baseline:
        expected = baseline[f"{history}/{cfg_weight}/{mode}"]
    # CPU kernels differ slightly between deployment and development machines.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_jit_switch_noise_and_disabled_guidance(sampler):
    """A shared compiled entry accepts both modes and repeatable caller-supplied noise."""
    _, _, model, obs = sampler
    sample = nnx_utils.module_jit(
        model.sample_actions,
        static_argnames=("enable_rtc", "cfg_guidance_weight", "prefix_attention_schedule"),
    )
    kwargs = {"num_steps": 3, "noise": jnp.ones((1, 2, 4)), "cfg_guidance_weight": 2.0}
    first = sample(jax.random.key(1), obs, enable_rtc=False, **kwargs)
    rtc = sample(
        jax.random.key(2),
        obs,
        enable_rtc=True,
        prefix_actions=jnp.zeros((2, 4)),
        prefix_attention_schedule="linear",
        **kwargs,
        **RTC,
    )
    assert not np.allclose(first, rtc)
    after = sample(jax.random.key(3), obs, enable_rtc=False, **kwargs)
    np.testing.assert_array_equal(first, after)
    disabled = sample(
        jax.random.key(4),
        obs,
        enable_rtc=True,
        prefix_actions=jnp.zeros((2, 4)),
        **kwargs,
        **{**RTC, "max_guidance_weight": 0.0},
    )
    np.testing.assert_allclose(disabled, first, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"prefix_actions": np.zeros((2, 4))},
        {"prefix_actions": np.zeros((1, 4)), **RTC},
        {"prefix_actions": np.zeros((2, 5)), **RTC},
    ],
)
def test_incomplete_or_malformed_prefix_fails_clearly(sampler, kwargs):
    _, _, model, obs = sampler
    with pytest.raises(ValueError, match="prefix_actions|RTC requires"):
        model.sample_actions(jax.random.key(0), obs, enable_rtc=True, **kwargs)


def test_ordinary_sampling_keeps_batch_support(sampler):
    _, cfg, model, _ = sampler
    obs = cfg.fake_obs(batch_size=2)
    result = model.sample_actions(jax.random.key(0), obs, num_steps=2)
    assert result.shape == (2, 2, 4)
    with pytest.raises(ValueError, match="batch_size=1"):
        model.sample_actions(jax.random.key(0), obs, enable_rtc=True)


@pytest.mark.parametrize(
    ("schedule", "expected"),
    [
        ("ones", [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
        ("zeros", [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("linear", [1, 1, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0]),
    ],
)
def test_prefix_schedule_boundaries(schedule, expected):
    np.testing.assert_allclose(get_prefix_weights(2, 6, 10, schedule), expected, atol=1e-7)
    np.testing.assert_array_equal(get_prefix_weights(3, 0, 10, schedule), np.zeros(10))


def test_unknown_schedule_fails():
    with pytest.raises(ValueError, match="Invalid schedule"):
        get_prefix_weights(0, 2, 4, "invalid")
