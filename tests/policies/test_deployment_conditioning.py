"""Deployment must tokenize positive advantage even with an offline training recipe."""

import numpy as np
import pytest

from vam import transforms
from vam.models import model
from vam.models.vam_config import VAMConfig
from vam.policies import policy as policy_module
from vam.policies import policy_config
from vam.training import config


@pytest.mark.parametrize("name", ["pi05_aloha_value_expert", "pi05_aloha_history_value_expert"])
def test_deployment_uses_model_conditioning_without_changing_training_recipe(monkeypatch, tmp_path, name):
    recipe = config.get_config(name)
    assert recipe.data.include_advantage is False
    monkeypatch.setattr(model, "restore_params", lambda *a, **kw: {})
    monkeypatch.setattr(VAMConfig, "load", lambda *a, **kw: object())
    monkeypatch.setattr(policy_module, "Policy", lambda model, **kw: kw)
    policy = policy_config.create_trained_policy(recipe, tmp_path, norm_stats={})
    injector = next(t for t in policy["transforms"] if isinstance(t, transforms.InjectAdvantage))
    result = injector({"advantage": "Advantage:Positive"})
    assert "advantage" not in result
    assert np.issubdtype(result["tokenized_advantage"].dtype, np.integer)
    assert result["tokenized_advantage_mask"].any()
    assert recipe.data.include_advantage is False
