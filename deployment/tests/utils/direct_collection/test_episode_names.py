"""Sequential names retain UUID identity and reserve interrupted episode numbers."""

import uuid
from pathlib import Path
from unittest.mock import Mock

import pytest

from utils.lerobot_direct import DirectSession, validate_direct_episode
from utils.lerobot_episode_names import (
    bundle_directory_name,
    episode_name,
    next_episode_index,
)
from utils.lerobot_online.protocol import atomic_json
from utils.lerobot_progress import run_rollout_history


@pytest.mark.parametrize("index,name", [(0, "episode_000"), (20, "episode_020"), (1000, "episode_1000")])
def test_three_digit_names(index, name):
    assert episode_name(index) == name


@pytest.mark.parametrize("index", [-1, None, True, 1.5])
def test_invalid_index_rejected(index):
    with pytest.raises(ValueError):
        episode_name(index)


def test_numbered_real_output_contains_both_roles_and_uuid(recorded):
    manifest = validate_direct_episode(recorded, decode=True)
    assert recorded.name == manifest["episode_name"] == "episode_000"
    assert str(uuid.UUID(manifest["episode_uid"])) == manifest["episode_uid"]
    assert (recorded / "action").is_dir() and (recorded / "value").is_dir()


def test_directory_cannot_escape_or_disagree_with_index():
    manifest = {"episode_uid": str(uuid.uuid4()), "rollout_index": 1}
    assert bundle_directory_name(manifest) == manifest["episode_uid"]
    for name in ("../episode_001", "episode_002", "/episode_001", "episode_000001"):
        with pytest.raises(ValueError):
            bundle_directory_name({**manifest, "episode_name": name})


def test_restart_continues_after_legacy_data_and_reserved_cache(config, policy):
    root = Path(config["output_dir"])
    for index in range(2):
        uid = str(uuid.uuid4())
        atomic_json(
            root / "ready" / uid / "manifest.json",
            {
                "episode_uid": uid,
                "run_name": config["run_name"],
                "result": "success",
                "start_ns": index,
            },
        )
    assert next_episode_index(root, config["run_name"]) == 2
    first = DirectSession(config, executor=Mock())
    assert first.start(policy, 10, 2)
    assert first.active.manifest["episode_name"] == "episode_002"
    first.close()  # Interrupted cache stays, so its number cannot be reused.
    assert next_episode_index(root, config["run_name"]) == 3
    second = DirectSession(config, executor=Mock())
    try:
        assert second.start(policy, 20, 0)  # A stale caller cannot overwrite history.
        assert second.active.manifest["episode_name"] == "episode_003"
    finally:
        second.close()


def test_existing_numbered_directory_is_never_reused(tmp_path):
    (tmp_path / "ready/episode_1000").mkdir(parents=True)
    assert next_episode_index(tmp_path, "run") == 1001


def test_discard_reuses_number_with_new_identity_and_survives_restart(config, policy):
    """Repeated D removes raw data without consuming a number or conversion slot."""
    config["progress_dir"] = str(Path(config["output_dir"]) / ".runtime/test")
    session = DirectSession(config, executor=Mock())
    identities = set()
    try:
        for _ in range(3):
            assert session.start(policy, 10, 0)
            cache = session.active
            identities.add(cache.manifest["episode_uid"])
            assert cache.manifest["episode_name"] == "episode_000"
            assert session._last_status["rollout_index"] == 0
            session.stop("discard", 20)
            assert not cache.path.exists()
            assert not session.futures
        assert len(identities) == 3
        assert next_episode_index(config["output_dir"], config["run_name"]) == 0
    finally:
        session.close()
    restarted = DirectSession(config, executor=Mock())
    try:
        assert restarted.start(policy, 30)
        assert restarted.active.manifest["episode_name"] == "episode_000"
        restarted.stop("success", 40)
        assert restarted.start(policy, 50)
        assert restarted.active.manifest["episode_name"] == "episode_001"
        restarted.stop("discard", 60)
        assert restarted.start(policy, 70, 1)
        assert restarted.active.manifest["episode_name"] == "episode_001"
        restarted.stop("discard", 80)
    finally:
        restarted.close()


def test_processing_label_keeps_numbered_identity_after_a_gap(tmp_path):
    uid = str(uuid.uuid4())
    atomic_json(
        tmp_path / "capture" / f"{uid}_temp" / "manifest.json",
        {
            "episode_uid": uid,
            "run_name": "run",
            "result": "success",
            "start_ns": 10,
            "normal_end": True,
            "episode_name": "episode_005",
            "rollout_index": 5,
        },
    )
    assert run_rollout_history(tmp_path, "run") == {uid: 5}
    assert next_episode_index(tmp_path, "run") == 6
