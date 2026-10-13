"""Checkpoint-local asset lookup must never silently choose between datasets."""

import json
from types import SimpleNamespace

import pytest

from policy.vam_reload_utils import resolve_checkpoint_norm_stats


def write_stats(root, asset_id, content="{}"):
    path = root / "assets" / asset_id / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_exact_asset_wins_even_with_other_datasets(tmp_path):
    expected = write_stats(tmp_path, "aloha/original")
    write_stats(tmp_path, "aloha/other")
    assert resolve_checkpoint_norm_stats(tmp_path, "aloha/original") == expected


@pytest.mark.parametrize("asset_id", ["aloha/original", None])
def test_only_checkpoint_asset_is_used_without_modifying_files(tmp_path, asset_id):
    expected = write_stats(tmp_path, "aloha/original_downsampled_by_3", "unique stats")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert resolve_checkpoint_norm_stats(tmp_path, asset_id) == expected
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("empty_assets", [False, True])
def test_missing_stats_fail_instead_of_using_other_checkpoints(tmp_path, empty_assets):
    checkpoint = tmp_path / "checkpoint"
    if empty_assets:
        (checkpoint / "assets").mkdir(parents=True)
    write_stats(tmp_path / "old_checkpoint", "aloha/original")
    with pytest.raises(FileNotFoundError, match="No norm_stats.json"):
        resolve_checkpoint_norm_stats(checkpoint, "aloha/original")


def test_multiple_unmatched_assets_fail_with_candidate_paths(tmp_path):
    write_stats(tmp_path, "aloha/one")
    write_stats(tmp_path, "aloha/two")
    with pytest.raises(ValueError, match="Ambiguous") as error:
        resolve_checkpoint_norm_stats(tmp_path, "aloha/missing")
    assert "aloha/one/norm_stats.json" in str(error.value)
    assert "aloha/two/norm_stats.json" in str(error.value)


def test_corrupt_exact_asset_is_not_replaced_by_another_dataset(tmp_path):
    expected = write_stats(tmp_path, "aloha/original", "not JSON")
    write_stats(tmp_path, "aloha/other")
    # Selection is based on identity. The VAM parser must report corruption.
    assert resolve_checkpoint_norm_stats(tmp_path, "aloha/original") == expected


def test_initial_load_and_reload_pass_each_checkpoints_own_stats(tmp_path, monkeypatch):
    """Exercise the deployment adapter and real parser without loading model weights."""
    pytest.importorskip("jax")
    from vam.policies import policy_config
    from vam.training import config

    from policy import vam_policy as adapter

    base, merged = tmp_path / "base", tmp_path / "merged"
    for root, asset_id, mean in (
        (base, "aloha/original", 1),
        (merged, "aloha/original_downsampled_by_3", 2),
    ):
        write_stats(
            root,
            asset_id,
            json.dumps(
                {
                    "norm_stats": {"state": {"mean": [mean], "std": [1]}},
                }
            ),
        )
    train_config = SimpleNamespace(
        data=SimpleNamespace(create_base_config=lambda *args: SimpleNamespace(asset_id="aloha/original")),
        assets_dirs=tmp_path,
        model=object(),
    )
    monkeypatch.setattr(config, "get_config", lambda name: train_config)
    calls = []

    def create_policy(cfg, checkpoint_dir, **kwargs):
        assert cfg is train_config
        calls.append((checkpoint_dir, kwargs["norm_stats"]["state"].mean.tolist()))
        return SimpleNamespace()

    monkeypatch.setattr(policy_config, "create_trained_policy", create_policy)
    monkeypatch.setattr(adapter, "clear_accelerator_caches", lambda: None)
    policy = object.__new__(adapter.VAMPolicy)
    policy.pi_config_name = "test"
    policy.device = "cpu"
    policy.cfg_guidance_weight = 1.0
    policy._policy = policy.load_vam_policy("test", str(base))
    policy.reload_checkpoint(str(merged), policy_id=1, checkpoint_id="merged")
    assert [(str(path), mean) for path, mean in calls] == [
        (str(base), [1]),
        (str(merged), [2]),
    ]
    assert policy.checkpoint_id == "merged"
