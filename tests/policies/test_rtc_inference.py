"""Policy dispatch, first-frame fallback and joint-prefix rebasing without hardware."""

# These tests exercise the deployed adapter reset/cache contract.
# ruff: noqa: SLF001

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

from vam import transforms
from vam.policies.policy import Policy

MASK = transforms.make_bool_mask(6, -1, 6, -1)
RTC = {"inference_delay": 0, "prefix_attention_horizon": 2, "max_guidance_weight": 2.0}


class EchoSampler(nnx.Module):
    def sample_actions(
        self, rng, obs, *, enable_rtc=False, prefix_actions=None, prefix_attention_schedule="exp", noise=None, **kwargs
    ):
        if enable_rtc:
            return prefix_actions[None]
        return jnp.zeros((1, 3, 14)) if noise is None else noise


def input_transform(data):
    result = {"state": np.array(data["observation/state"]), "image": {}, "image_mask": {}}
    if "actions" in data:
        result["actions"] = np.array(data["actions"][0])
    return transforms.DeltaActions(MASK)(result)


def output_transform(data):
    return transforms.AbsoluteActions(MASK)({k: np.array(v) for k, v in data.items()})


@pytest.fixture
def policy():
    return Policy(EchoSampler(), transforms=[input_transform], output_transforms=[output_transform])


def test_first_frame_seeds_prefix_and_later_frames_rebase_joints(policy):
    initial = {"observation/state": np.ones(14)}
    noise = np.arange(42, dtype=np.float32).reshape(3, 14) / 10
    first = policy.infer(initial, noise=noise, enable_rtc=True, time_base=0, rtc_config=RTC)
    expected = noise + np.where(MASK, initial["observation/state"], 0)
    np.testing.assert_allclose(first["actions"], expected)
    second = policy.infer(
        {"observation/state": np.ones(14) * 2},
        enable_rtc=True,
        time_base=1,
        rtc_config={**RTC, "prefix_attention_schedule": "linear"},
    )
    # One surviving absolute target; padding returns current joints and zero grippers.
    np.testing.assert_allclose(second["actions"][0], first["actions"][2])
    np.testing.assert_allclose(second["actions"][1:], np.tile(np.where(MASK, 2, 0), (2, 1)))
    assert "prefix_actions" in second
    np.testing.assert_array_equal(policy._prev_actions, second["actions"])
    policy._prev_actions = None
    reset = policy.infer(initial, enable_rtc=True, time_base=0, rtc_config=RTC)
    np.testing.assert_array_equal(reset["actions"], np.tile(np.where(MASK, 1, 0), (3, 1)))


def test_explicit_off_ignores_cached_prefix_and_rtc_parameters(policy):
    policy._prev_actions = np.ones((3, 14)) * 99
    output = policy.infer({"observation/state": np.zeros(14)}, enable_rtc=False, time_base=0, rtc_config=RTC)
    np.testing.assert_array_equal(output["actions"], np.zeros((3, 14)))
    assert "prefix_actions" not in output


@pytest.mark.parametrize("kwargs", [{}, {"time_base": 0}, {"time_base": 0, "rtc_config": {}}])
def test_explicit_on_requires_rtc_metadata(policy, kwargs):
    with pytest.raises(ValueError, match="requires|must contain"):
        policy.infer({}, enable_rtc=True, **kwargs)
