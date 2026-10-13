"""CPU contracts for the retained VAM model and the ALOHA training/inference boundary."""

import dataclasses

# These tests inspect internal training/loss contracts intentionally.
# ruff: noqa: SLF001
from unittest.mock import patch

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vam import transforms
from vam.models import gemma
from vam.models import vam
from vam.models import vam_config as config
from vam.models.utils import value_distribution
from vam.policies.async_aloha_policy import AsyncSequenceAlohaInputs
from vam.training import config as training_config


@pytest.mark.parametrize(("action_horizon", "value_horizon"), [(1, 1), (4, 2), (1, 4)])
def test_history_input_specs(action_horizon, value_horizon):
    model = config.VAMConfig(
        action_obs_encoder=config.ObsEncoderConfig(observation_horizon=action_horizon),
        value_obs_encoder=config.ObsEncoderConfig(observation_horizon=value_horizon),
    )
    obs, action, value = model.inputs_spec(batch_size=2)
    horizon = max(action_horizon, value_horizon)
    assert obs.state.shape == ((2, 32) if horizon == 1 else (2, horizon, 32))
    assert obs.image_masks["base_0_rgb"].shape == ((2,) if horizon == 1 else (2, horizon))
    assert action.shape == (2, 50, 32)
    assert value.shape == (2, 256)
    assert model.value_loss == "hl_gauss"
    assert dataclasses.replace(model, td_learning=True).value_loss == "distributional"
    assert not dataclasses.replace(model, action_expert=None).has_action_expert_branch
    assert not dataclasses.replace(model, value_expert=None).has_value_expert_branch


@pytest.mark.parametrize("kwargs", [{"kind": "paligemma_fast"}, {"observation_horizon": 0}])
def test_reject_unsupported_observation_encoder(kwargs):
    with pytest.raises(ValueError, match="Invalid obs encoder|observation_horizon"):
        config.ObsEncoderConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"n_bins": 1}, {"vmin": 1, "vmax": 0}, {"gaussian_sigma": 0}])
def test_reject_invalid_support(kwargs):
    with pytest.raises(ValueError, match="n_bins|vmin|gaussian_sigma"):
        config.ValueExpertConfig(**kwargs)


def raw_observation():
    return {
        "observation/state": np.ones(14, dtype=np.float32),
        **{
            f"observation/{key}": np.zeros((16, 16, 3), dtype=np.uint8)
            for key in ("base_image", "left_wrist_image", "right_wrist_image")
        },
    }


def test_next_observation_is_optional_only_for_inference():
    adapter = AsyncSequenceAlohaInputs(model_type=config.VAMConfig().model_type)
    raw = raw_observation()
    assert "next_state" not in adapter(raw)
    raw.update({key.replace("observation/", "observation/next_"): value for key, value in raw.copy().items()})
    raw["observation/next_state_is_pad"] = True
    td = adapter(raw)
    assert set(td["next_image"]) == set(td["image"])
    assert not td["next_state_mask"]
    del raw["observation/next_base_image"]
    with pytest.raises(KeyError):
        adapter(raw)


def test_registry_and_td_repack_contract(tmp_path):
    names = {c.name for c in training_config._CONFIGS}
    assert names == {
        "pi05_aloha_value_expert",
        "pi05_aloha_history_value_expert",
        "pi05_aloha_async_recap_dummy",
        "pi05_aloha_async_recap",
        "pi05_aloha_async_recap_split_action",
        "pi05_aloha_async_recap_split_value",
        "pi05_aloha_async_recap_streaming_validation",
    }
    model = config.VAMConfig(td_learning=True, value_obs_encoder=config.ObsEncoderConfig(observation_horizon=2))
    factory = training_config.LeRobotMyDataAlohaValueConfig(
        repo_id="example/offline",
        bimanual_robot=True,
        use_next_obs=True,
        include_advantage=False,
    )
    with patch.object(training_config.ModelTransformFactory, "__call__", return_value=transforms.Group()):
        resolved = factory.create(tmp_path, model)
    repack = resolved.repack_transforms.inputs[0].structure
    assert repack["observation/next_state"] == "next_state"
    assert "advantage" not in repack
    assert repack["terminate"] == "terminate"
    assert repack["observation/next_base_image_is_pad"] == "next_base_image_is_pad"
    assert repack["observation/base_image_is_pad"] == "base_image_is_pad"
    for name in ("pi05_aloha_value_expert", "pi05_aloha_history_value_expert"):
        recipe = training_config.get_config(name)
        assert recipe.train_type == {"bc", "value_mc_learning"}
        assert not recipe.model.td_learning


def test_terminal_td_stops_bootstrap_and_preserves_distribution():
    probs = jnp.array([[0.1, 0.2, 0.3, 0.4]], dtype=jnp.float32)
    kwargs = {"vmin": 0.0, "vmax": 1.0, "gaussian_sigma": 0.02, "dtype": jnp.float32}
    continued = value_distribution.distributional_td_target_distribution(
        probs,
        jnp.zeros((1, 1)),
        1.0,
        None,
        **kwargs,
    )
    np.testing.assert_allclose(continued, probs, atol=1e-6)
    terminal = value_distribution.distributional_td_target_distribution(
        probs,
        jnp.ones((1, 1)),
        1.0,
        jnp.ones((1, 1)),
        **kwargs,
    )
    np.testing.assert_allclose(terminal, [[0, 0, 0, 1]], atol=1e-5)


class TinyImageEncoder(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, image, image_mask=None, *, train=False):
        if image.ndim == 5:
            image = image[:, -1]
        pooled = jnp.mean(image, axis=(-3, -2))[:, None, :]
        return nn.Dense(self.num_classes)(pooled), {}


def small_model(monkeypatch, history, td):
    monkeypatch.setattr(vam, "_make_image_encoder", lambda **kw: TinyImageEncoder(kw["num_classes"]))
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda _: gemma.Config(
            width=16,
            depth=1,
            mlp_dim=32,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            vocab_size=128,
        ),
    )
    cfg = config.VAMConfig(
        action_obs_encoder=config.ObsEncoderConfig(
            variant="dummy",
            observation_horizon=history,
            history_tokens=config.HistoryTokenConfig(use_state_tokens=history > 1, use_value_tokens=history > 1),
        ),
        value_obs_encoder=config.ObsEncoderConfig(variant="dummy", observation_horizon=history),
        action_expert=config.ActionExpertConfig(variant="dummy", pi05=True),
        value_expert=config.ValueExpertConfig(n_bins=8),
        action_dim=4,
        action_horizon=2,
        max_token_len=4,
        td_learning=td,
        dtype="float32",
    )
    return cfg, cfg.create(jax.random.key(0))


@pytest.mark.parametrize(("history", "td"), [(1, False), (2, True)])
def test_model_forward_and_checkpoint_roundtrip(monkeypatch, history, td):
    cfg, model = small_model(monkeypatch, history, td)
    obs = cfg.fake_obs()
    probs = model.predict_value(obs)
    assert probs.shape == (1, 8)
    np.testing.assert_allclose(np.sum(probs, axis=-1), [1], atol=1e-6)
    actions = model.sample_actions(jax.random.key(1), obs, num_steps=2)
    assert actions.shape == (1, 2, 4)
    assert np.isfinite(actions).all()
    mc = model._mc_value_loss(obs, jnp.ones((1, 8)) / 8, jnp.zeros((1, 1)))
    assert mc.shape == (1, 1)
    assert np.isfinite(mc).all()
    if td:
        loss = model._td_value_loss(obs, obs, jnp.zeros((1, 2)), jnp.ones((1, 1)))
        assert loss.shape == (1, 1)
        assert np.isfinite(loss).all()
    graph, state = nnx.split(model)
    restored = nnx.merge(graph, state)
    np.testing.assert_allclose(restored.predict_value(obs), probs)
    loaded = cfg.load(state.to_pure_dict())
    np.testing.assert_array_equal(loaded.predict_value(obs), probs)
    assert isinstance(loaded, vam.VAM)
    assert not hasattr(model, "generate_subtask")
