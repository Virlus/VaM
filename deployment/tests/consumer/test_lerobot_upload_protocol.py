"""Existing uploader dispatch must bypass legacy metadata rewriting in bundle mode."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from consumer.data_upload_manager import DataUploadManager
from consumer.policy_connector import PolicyConnector


def manager(tmp_path):
    return DataUploadManager(
        data_dir=str(tmp_path / "ready"),
        remote_path="/remote/inbox/run",
        host="example.test",
        port=22,
        username="test",
        poll_interval=1,
        ssh_key_path=None,
        upload_config={
            "payload_format": "lerobot_direct_v1",
        },
    )


def test_direct_live_recorder_is_available_without_nedf():
    import sys

    connector = SimpleNamespace(
        infer_rollout_config={"enabled": True, "format": "lerobot"},
        rollout_python_path=sys.executable,
    )
    assert PolicyConnector._should_start_infer_rollout(connector)
    connector.infer_rollout_config["enabled"] = False
    assert not PolicyConnector._should_start_infer_rollout(connector)


def test_direct_upload_dispatch_skips_legacy_metadata(tmp_path, monkeypatch):
    from utils import lerobot_direct_upload

    upload = manager(tmp_path)
    upload.upload_config["payload_format"] = "lerobot_direct_v1"
    ready = tmp_path / "ready"
    (ready / "episode").mkdir(parents=True)
    (ready / "episode/manifest.json").write_text("{}")
    (ready / "partial").mkdir()
    send = Mock()
    monkeypatch.setattr(lerobot_direct_upload, "upload_direct", send)
    upload._prepare_episode_policy_metadata = Mock(side_effect=AssertionError("legacy rewrite"))
    upload._poll_and_upload()
    send.assert_called_once_with(upload, ready / "episode")
    upload._prepare_episode_policy_metadata.assert_not_called()


def test_collect_upload_dispatch(tmp_path, monkeypatch):
    from utils import lerobot_collect_upload

    upload = manager(tmp_path)
    upload.upload_config["payload_format"] = "lerobot_collect_v1"
    poll = Mock()
    monkeypatch.setattr(lerobot_collect_upload, "poll_collect", poll)
    upload._poll_and_upload()
    poll.assert_called_once_with(upload)


def test_upload_rejects_unknown_format_and_nonready_directory(tmp_path):
    upload = manager(tmp_path)
    upload.data_dir = str(tmp_path / "capture")
    with pytest.raises(ValueError, match="ready"):
        upload._poll_and_upload()
    upload.upload_config["payload_format"] = "unknown"
    with pytest.raises(ValueError, match="Unsupported"):
        upload._poll_and_upload()
