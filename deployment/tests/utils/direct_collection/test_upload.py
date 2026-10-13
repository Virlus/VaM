"""Upload retry, last-marker publication and immutable source bundle."""

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utils.lerobot_direct_upload import upload_direct
from utils.lerobot_online.protocol import atomic_json, read_json


def test_upload_retry_and_no_overwrite(recorded, tmp_path):
    remote = tmp_path / "remote"
    sftp = SimpleNamespace(
        open=open,
        rename=lambda a, b: Path(a).rename(b),
        close=Mock(),
    )
    ssh = SimpleNamespace(open_sftp=lambda: sftp, close=Mock())
    copy = Mock(side_effect=lambda ftp, source, target: bool(shutil.copyfile(source, target)))
    manager = SimpleNamespace(
        remote_path=str(remote),
        _get_ssh_client=lambda **kw: ssh,
        _ensure_remote_dirs_recursive=lambda ftp, path: Path(path).mkdir(parents=True, exist_ok=True),
        _rollout_upload_method=lambda: "sftp",
        _upload_file_via_sftp_from_scratch=copy,
    )
    original = (recorded / "manifest.json").read_bytes()
    uid = read_json(recorded / "manifest.json")["episode_uid"]
    copy.side_effect = OSError("network interrupted")
    with pytest.raises(OSError, match="interrupted"):
        upload_direct(manager, recorded)
    marker = remote / recorded.name / "complete.txt"
    assert not marker.exists()
    copy.side_effect = lambda ftp, source, target: bool(shutil.copyfile(source, target))
    assert upload_direct(manager, recorded)
    assert marker.exists()
    committed = read_json(marker)
    count = copy.call_count
    assert upload_direct(manager, recorded)
    assert copy.call_count == count
    assert (recorded / "manifest.json").read_bytes() == original
    committed["manifest_sha256"] = "bad"
    atomic_json(marker, committed)
    with pytest.raises(ValueError, match="Conflicting"):
        upload_direct(manager, recorded)
    assert read_json(recorded.parent.parent / "state" / "uploads" / f"{uid}.json")["status"] == "UPLOADED"


def test_rsync_failure_falls_back_without_early_marker(recorded, tmp_path):
    remote = tmp_path / "rsync_remote"
    sftp = SimpleNamespace(open=open, rename=lambda a, b: Path(a).rename(b), close=Mock())
    ssh = SimpleNamespace(open_sftp=lambda: sftp, close=Mock())
    manager = SimpleNamespace(
        remote_path=str(remote),
        _get_ssh_client=lambda **kw: ssh,
        _ensure_remote_dirs_recursive=lambda ftp, path: Path(path).mkdir(parents=True, exist_ok=True),
        _rollout_upload_method=lambda: "rsync",
        _build_rollout_rsync_command=lambda source, target: ["rsync", source, target],
        _run_external_transfer=Mock(side_effect=OSError("rsync unavailable")),
        _rollout_rsync_fallback_enabled=lambda: False,
        _upload_file_via_sftp_from_scratch=lambda ftp, source, target: bool(shutil.copyfile(source, target)),
    )
    with pytest.raises(OSError, match="rsync unavailable"):
        upload_direct(manager, recorded)
    assert not (remote / recorded.name / "complete.txt").exists()
    manager._rollout_rsync_fallback_enabled = lambda: True
    assert upload_direct(manager, recorded)
    command = manager._run_external_transfer.call_args.args[0]
    assert "--checksum" in command and "--from0" in command


def test_legacy_uuid_bundle_remains_uploadable(recorded, tmp_path):
    """Old completed bundles must not be renamed or uploaded a second time."""
    manifest = read_json(recorded / "manifest.json")
    manifest.pop("episode_name")
    atomic_json(recorded / "manifest.json", manifest)
    legacy = recorded.with_name(manifest["episode_uid"])
    recorded.rename(legacy)
    test_upload_retry_and_no_overwrite(legacy, tmp_path)
