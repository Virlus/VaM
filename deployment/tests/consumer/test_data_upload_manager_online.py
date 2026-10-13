import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "data_upload_manager_under_test",
    PROJECT_ROOT / "consumer" / "data_upload_manager.py",
)
data_upload_manager = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(data_upload_manager)
DataUploadManager = data_upload_manager.DataUploadManager


class FakeSFTP:
    def __init__(self, dirs: set[str]) -> None:
        self.dirs = set(dirs)

    def stat(self, path: str):
        if path not in self.dirs:
            raise FileNotFoundError(2, "No such file")
        return object()

    def mkdir(self, path: str) -> None:
        parent = str(Path(path).parent)
        if parent == ".":
            parent = ""
        if parent and parent not in self.dirs:
            raise FileNotFoundError(2, "No such file")
        self.dirs.add(path)


def _capture_prefixed_logs(manager: DataUploadManager) -> list[dict]:
    logs = []

    def capture(tag, message, **kwargs):
        logs.append({"tag": tag, "message": message, "level": kwargs.get("level", "info")})

    manager._prefixed_log = capture
    return logs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f)


def _manager(tmp_path: Path) -> DataUploadManager:
    data_dir = tmp_path / "data"
    policy_dir = tmp_path / "policy"
    checkpoint_root = tmp_path / "checkpoints"
    data_dir.mkdir()
    policy_dir.mkdir()
    checkpoint_root.mkdir()
    return DataUploadManager(
        data_dir=str(data_dir),
        remote_path="/remote/data",
        host="example.com",
        port=22,
        username="user",
        poll_interval=1.0,
        ssh_key_path=None,
        online_training_config={
            "enabled": True,
            "policy_directory": str(policy_dir),
            "local_checkpoint_root": str(checkpoint_root),
        },
    )


def _complete_checkpoint(
    root: Path,
    name: str,
    manifest: dict | None = None,
    *,
    archived: bool = True,
) -> Path:
    checkpoint_dir = root / "archive" / name if archived else root / name
    (checkpoint_dir / "params").mkdir(parents=True)
    (checkpoint_dir / "complete.txt").write_text("downloaded\n")
    if manifest is not None:
        _write_json(checkpoint_dir / "manifest.json", manifest)
    return checkpoint_dir


def _episode(data_dir: Path, name: str, policy_info: dict | None = None) -> Path:
    episode_dir = data_dir / name
    episode_dir.mkdir()
    (episode_dir / "data.txt").write_text("rollout\n")
    if policy_info is not None:
        _write_json(episode_dir / "episode_policy.json", policy_info)
    return episode_dir


def test_get_ssh_client_does_not_log_error_when_retry_succeeds(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    manager.upload_config.update({"sftp_connect_attempts": 2, "sftp_connect_retry_delay": 0})
    logs = _capture_prefixed_logs(manager)
    connect_results = [OSError("temporary reset"), None]
    created_clients = []

    class FakeSSHClient:
        def __init__(self) -> None:
            self.log_channel = None
            created_clients.append(self)

        def set_log_channel(self, log_channel: str) -> None:
            self.log_channel = log_channel

        def set_missing_host_key_policy(self, policy) -> None:
            pass

        def connect(self, **kwargs) -> None:
            result = connect_results.pop(0)
            if result is not None:
                raise result

        def close(self) -> None:
            pass

    monkeypatch.setattr(data_upload_manager, "SSHClient", FakeSSHClient)

    client = manager._get_ssh_client(context="UPLOAD")

    assert client is created_clients[1]
    assert all(client.log_channel.startswith("paramiko.upload.suppressed.") for client in created_clients)
    assert not any("SFTP connect failed" in log["message"] for log in logs)
    assert [log["level"] for log in logs if "SFTP connect succeeded after retry" in log["message"]] == ["info"]
    assert not any(log["level"] == "error" for log in logs)


def test_get_ssh_client_logs_error_after_retries_exhausted(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    manager.upload_config.update({"sftp_connect_attempts": 2, "sftp_connect_retry_delay": 0})
    logs = _capture_prefixed_logs(manager)
    connect_results = [OSError("reset 1"), OSError("reset 2")]

    class FakeSSHClient:
        def set_log_channel(self, log_channel: str) -> None:
            self.log_channel = log_channel

        def set_missing_host_key_policy(self, policy) -> None:
            pass

        def connect(self, **kwargs) -> None:
            raise connect_results.pop(0)

        def close(self) -> None:
            pass

    monkeypatch.setattr(data_upload_manager, "SSHClient", FakeSSHClient)

    try:
        manager._get_ssh_client(context="UPLOAD")
        assert False, "expected final SSH connect failure"
    except OSError as exc:
        assert "reset 2" in str(exc)

    assert not any("SFTP connect failed; retrying" in log["message"] for log in logs)
    assert [log["level"] for log in logs if "SFTP connect failed after 2 attempts" in log["message"]] == ["error"]


def test_latest_registry_same_step_downloads_when_archive_missing(tmp_path):
    manager = _manager(tmp_path)
    latest_path = manager._local_latest_checkpoint_path()
    _write_json(
        latest_path,
        {
            "policy_id": 1,
            "checkpoint_id": "ckpt_000001",
            "checkpoint_name": "ckpt_000001",
            "checkpoint_path": str(manager.local_checkpoint_root / "archive" / "ckpt_000001"),
            "params_path": str(manager.local_checkpoint_root / "archive" / "ckpt_000001" / "params"),
            "train_step": 200,
        },
    )
    remote_latest = {
        "policy_id": 1,
        "checkpoint_id": "ckpt_000001",
        "checkpoint_name": "ckpt_000001",
        "checkpoint_path": "/remote/checkpoints/ckpt_000001",
        "train_step": 200,
    }
    checkpoint_ref = {
        **remote_latest,
        "remote_latest_info": dict(remote_latest),
        "remote_checkpoint_path": "/remote/checkpoints/ckpt_000001",
        "remote_params_path": "/remote/checkpoints/ckpt_000001/params",
    }
    downloaded = []

    manager._remote_exists = lambda sftp, remote_path: True
    manager._read_remote_json_safe = lambda sftp, remote_path: dict(remote_latest)
    manager._build_remote_checkpoint_ref_from_latest = lambda sftp, latest_path_arg, latest_info: dict(checkpoint_ref)
    manager._download_checkpoint_if_needed = lambda sftp, ref: downloaded.append(ref["checkpoint_name"]) or False

    manager._poll_latest_checkpoint_registry(object(), "/remote/latest.json")

    assert downloaded == ["ckpt_000001"]
    latest_info = json.loads(latest_path.read_text())
    assert latest_info["download_status"] == "pending_download"
    assert latest_info["checkpoint_path"] == "/remote/checkpoints/ckpt_000001"


def test_latest_registry_writes_pending_before_download_without_current_policy_switch(tmp_path):
    manager = _manager(tmp_path)
    _write_json(
        manager.current_policy_path,
        {
            "policy_id": 0,
            "checkpoint_id": "base_39999",
            "checkpoint_dir": "/local/base_39999",
            "params_dir": "/local/base_39999/params",
        },
    )
    remote_latest = {
        "policy_id": 1,
        "checkpoint_id": "ckpt_000001",
        "checkpoint_name": "ckpt_000001",
        "checkpoint_path": "/remote/checkpoints/ckpt_000001",
        "train_step": 200,
    }
    checkpoint_ref = {
        **remote_latest,
        "remote_latest_info": dict(remote_latest),
        "remote_checkpoint_path": "/remote/checkpoints/ckpt_000001",
        "remote_params_path": "/remote/checkpoints/ckpt_000001/params",
    }
    observed_during_download = {}

    def fake_download(sftp, ref):
        observed_during_download["latest"] = json.loads(manager._local_latest_checkpoint_path().read_text())
        observed_during_download["current"] = json.loads(manager.current_policy_path.read_text())
        archive_dir = manager._checkpoint_archive_dir(ref)
        assert archive_dir is not None
        (archive_dir / "params").mkdir(parents=True)
        (archive_dir / "complete.txt").write_text("downloaded\n")
        return True

    manager._remote_exists = lambda sftp, remote_path: True
    manager._read_remote_json_safe = lambda sftp, remote_path: dict(remote_latest)
    manager._build_remote_checkpoint_ref_from_latest = lambda sftp, latest_path_arg, latest_info: dict(checkpoint_ref)
    manager._download_checkpoint_if_needed = fake_download

    manager._poll_latest_checkpoint_registry(object(), "/remote/latest.json")

    assert observed_during_download["latest"]["download_status"] == "pending_download"
    assert observed_during_download["latest"]["policy_id"] == 1
    assert observed_during_download["current"]["policy_id"] == 0
    latest_info = json.loads(manager._local_latest_checkpoint_path().read_text())
    assert latest_info["policy_id"] == 1
    assert latest_info["checkpoint_path"].endswith("/archive/ckpt_000001")
    assert "downloaded_at" in latest_info


def test_ensure_remote_dirs_recursive_creates_remote_path_before_episode(tmp_path):
    manager = _manager(tmp_path)
    fake_sftp = FakeSFTP({"/", "/remote"})

    manager._ensure_remote_dirs_recursive(fake_sftp, manager.remote_path)
    manager._ensure_remote_dir(fake_sftp, "/remote/data/episode_0001")

    assert "/remote/data" in fake_sftp.dirs
    assert "/remote/data/episode_0001" in fake_sftp.dirs
