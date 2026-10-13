"""Joint-space release regressions; no real checkpoint, dataset or robot is needed."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tyro

from vam import transforms
from vam.training import config as training_config
from vam.training import data_loader
from vam.training import sharding
from vam.training import value_inference


@pytest.mark.parametrize("name", ["pi05_aloha_value_expert", "pi05_aloha_history_value_expert"])
def test_cli_overrides_action_and_value_horizons_independently(name):
    baseline = training_config.get_config(name)
    updated = tyro.cli(
        training_config.TrainConfig,
        default=baseline,
        args=[
            "--model.action-obs-encoder.observation-horizon",
            "6",
            "--model.value-obs-encoder.observation-horizon",
            "3",
        ],
    )
    assert updated.model.action_observation_horizon == 6
    assert updated.model.value_observation_horizon == 3
    assert updated.model.observation_horizon == 6
    assert baseline.model.action_observation_horizon in {1, 4}
    assert baseline.model.value_observation_horizon in {1, 2}
    assert updated.data.observation_stride == baseline.data.observation_stride
    assert updated.model.action_obs_encoder.history_tokens == baseline.model.action_obs_encoder.history_tokens


@pytest.mark.parametrize("history", [1, 4])
def test_joint_delta_roundtrip_keeps_grippers_absolute(history):
    state = np.arange(history * 14, dtype=np.float32).reshape(history, 14)
    if history == 1:
        state = state[0]
    actions = np.arange(5 * 14, dtype=np.float32).reshape(5, 14) / 10
    mask = transforms.make_bool_mask(6, -1, 6, -1)
    original = {"state": state.copy(), "actions": actions.copy()}
    delta = transforms.DeltaActions(mask)(copy.deepcopy(original))
    current = state if history == 1 else state[-1]
    expected = actions - np.where(mask, current, 0)
    np.testing.assert_allclose(delta["actions"], expected)
    np.testing.assert_array_equal(delta["actions"][:, [6, 13]], actions[:, [6, 13]])
    restored = transforms.AbsoluteActions(mask)(delta)
    np.testing.assert_allclose(restored["actions"], actions, atol=5e-6)


def test_norm_pipeline_retains_joint_delta(tmp_path):
    recipe = training_config.get_config("pi05_aloha_value_expert")
    with patch.object(training_config.ModelTransformFactory, "__call__", return_value=transforms.Group()):
        data_config = recipe.data.create(tmp_path, recipe.model)
    chain = data_loader.build_lowdim_norm_transforms(data_config, recipe.model)
    assert any(isinstance(item, transforms.DeltaActions) for item in chain)
    state = np.arange(14, dtype=np.float32)
    actions = np.ones((100, 14), dtype=np.float32) * 4
    expected = actions - np.where(transforms.make_bool_mask(6, -1, 6, -1), state, 0)
    output = transforms.compose(chain)({"state": state, "actions": actions})
    np.testing.assert_allclose(output["actions"][..., :14], expected)


class TinyValueModel(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.zeros(3, dtype=jnp.float32))

    def predict_value(self, observation, actions, *, scale=1.0):
        return self.weight[None, :] + scale * observation + actions


@pytest.mark.parametrize("use_mesh", [False, True])
def test_annotation_loader_and_sharded_predict_keep_loaded_weights(monkeypatch, tmp_path, use_mesh):
    config = SimpleNamespace(model=SimpleNamespace(create=lambda rng: TinyValueModel()), freeze_filter=nnx.Nothing)
    params_path = tmp_path / "params"
    params_path.mkdir()

    def load(loader, reference):
        assert str(params_path) in repr(loader)
        assert reference["weight"].shape == (3,)
        return {"weight": jnp.array([0.1, 0.2, 0.7], dtype=jnp.float32)}

    monkeypatch.setattr(value_inference.training_utils, "load_and_validate_partial_params", load)
    mesh = sharding.make_mesh(1) if use_mesh else None
    model = value_inference.load_model_fsdp(config, jax.random.key(0), str(tmp_path), mesh)
    np.testing.assert_allclose(model.weight.value, [0.1, 0.2, 0.7])
    predict = value_inference.make_sharded_predict_value(
        model, sharding.make_mesh(1), predict_value_kwargs={"scale": 2.0}
    )
    actual = predict(jnp.ones((1, 3), dtype=jnp.float32), jnp.zeros((1, 3), dtype=jnp.float32))
    np.testing.assert_allclose(actual, [[2.1, 2.2, 2.7]])


def test_annotation_decodes_categorical_support_and_legacy_scalars():
    model_config = training_config.get_config("pi05_aloha_value_expert").model
    support = value_inference.get_value_decode_config(model_config)
    one_hot = np.eye(support.n_bins, dtype=np.float32)[[0, support.n_bins - 1]]
    np.testing.assert_allclose(value_inference.decode_value_predictions(one_hot, support), [support.vmin, support.vmax])
    scalar = value_inference.ValueDecodeConfig(discrete_value=False, n_bins=1, vmin=0, vmax=1)
    np.testing.assert_array_equal(value_inference.decode_value_predictions([[[1]], [[2]]], scalar), [1, 2])
    with pytest.raises(ValueError, match="scalar output"):
        value_inference.decode_value_predictions(np.array(1), scalar)
