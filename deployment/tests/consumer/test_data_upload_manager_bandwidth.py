"""Bandwidth budgets apply before I/O, including retries and resumed payloads."""

import importlib.util
import io
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.transfer_rate_limiter import TransferRateLimiter


def _manager(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "data_upload_manager_bandwidth_under_test",
        PROJECT_ROOT / "consumer" / "data_upload_manager.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.DataUploadManager(
        data_dir=str(tmp_path / "data"),
        remote_path="/remote/data",
        host="example.com",
        port=22,
        username="user",
        poll_interval=1.0,
        ssh_key_path=None,
        online_training_config={
            "enabled": True,
            "policy_directory": str(tmp_path / "policy"),
            "local_checkpoint_root": str(tmp_path / "checkpoints"),
        },
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def wait(self, delay):
        self.waits.append(delay)
        self.now += delay
        return False


def test_budget_persists_without_idle_burst_credit(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("utils.transfer_rate_limiter.time.monotonic", lambda: clock.now)
    limiter = TransferRateLimiter(100, clock)
    limiter.acquire(20)
    limiter.acquire(30)
    clock.now += 10
    limiter.acquire(10)
    assert clock.waits == pytest.approx([0.2, 0.3, 0.1])
    assert limiter.chunk_size(10000) == 5


def test_chunk_bounds_and_stop():
    stop = threading.Event()
    limiter = TransferRateLimiter(100 * 1024**2, stop)
    assert limiter.chunk_size(8 * 1024**2) == 256 * 1024
    assert limiter.chunk_size(100) == 100
    with pytest.raises(ValueError):
        limiter.chunk_size(0)
    with pytest.raises(ValueError):
        limiter.acquire(-1)
    stop.set()
    with pytest.raises(RuntimeError, match="stopping"):
        limiter.acquire(1024)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_rates(rate, tmp_path):
    with pytest.raises(ValueError):
        TransferRateLimiter(rate, threading.Event())
    manager = _manager(tmp_path)
    manager.upload_config["upload_bandwidth_mbps"] = rate
    with pytest.raises(ValueError):
        manager._bandwidth_limit("upload", 15)


def test_rsync_limits_and_config_precedence(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_rsync_executable", lambda: "/usr/bin/rsync")
    monkeypatch.setattr(manager, "_rollout_rsync_executable", lambda: "/usr/bin/rsync")
    manager.upload_config["rsync_extra_args"] = ["--bwlimit=0"]
    upload = manager._build_rollout_rsync_command(str(tmp_path), "/remote/data")
    download = manager._build_rsync_command("/remote/checkpoint", tmp_path / "ckpt")
    assert upload[-3] == "--bwlimit=48828"
    assert download[-3] == "--bwlimit=68359"
    manager.online_training_config["upload_bandwidth_mbps"] = 2
    assert manager._bandwidth_limit("upload", 15) == 2
    manager.upload_config["upload_bandwidth_mbps"] = 1
    assert manager._bandwidth_limit("upload", 15) == 1


class MemorySFTP:
    def __init__(self, data, reservations):
        self.data = data
        self.reservations = reservations

    def stat(self, path):
        return SimpleNamespace(st_size=len(self.data), st_mode=0)

    def open(self, path, mode):
        owner = self

        class RemoteFile(io.BytesIO):
            def read(self, size=-1):
                assert owner.reservations.pop(0) == size
                return super().read(size)

            def write(self, data):
                assert owner.reservations.pop(0) == len(data)
                return super().write(data)

            def close(self):
                owner.data = self.getvalue()
                super().close()

        result = RemoteFile(b"" if mode == "wb" else self.data)
        if mode == "ab":
            result.seek(0, 2)
        return result


@pytest.mark.parametrize("resume", [False, True])
def test_sftp_upload_paces_only_remaining_bytes(tmp_path, monkeypatch, resume):
    manager = _manager(tmp_path)
    reservations = []
    sizes = []

    def acquire(size):
        reservations.append(size)
        sizes.append(size)

    monkeypatch.setattr(manager._upload_limiter, "acquire", acquire)
    source = tmp_path / "payload"
    payload = b"x" * (600 * 1024)
    source.write_bytes(payload)
    offset = 100 * 1024 if resume else 0
    sftp = MemorySFTP(payload[:offset], reservations)
    method = manager._upload_file_via_sftp_resumable if resume else manager._upload_file_via_sftp_from_scratch
    assert method(sftp, str(source), "/remote/payload")
    assert sftp.data == payload
    assert sum(sizes) == len(payload) - offset
    assert max(sizes) <= 256 * 1024
    assert not reservations


@pytest.mark.parametrize("known_size", [False, True])
def test_sftp_download_paces_before_reads_and_resumes(tmp_path, monkeypatch, known_size):
    manager = _manager(tmp_path)
    reservations = []
    monkeypatch.setattr(manager._download_limiter, "acquire", reservations.append)
    payload = b"a" * (600 * 1024)
    target = tmp_path / "payload"
    target.with_name("payload.partial").write_bytes(payload[: 100 * 1024])
    sftp = MemorySFTP(payload, reservations)
    manager._download_remote_file(sftp, "/remote/payload", target, len(payload) if known_size else None)
    assert target.read_bytes() == payload
    assert not target.with_name("payload.partial").exists()
    assert not reservations


def test_legacy_mib_config_converts_to_decimal_mb(tmp_path):
    manager = _manager(tmp_path)
    manager.online_training_config["upload_bandwidth_mib_s"] = 15
    assert manager._bandwidth_limit("upload", 50) == 15.72864
    manager.upload_config["upload_bandwidth_mbps"] = 50
    assert manager._bandwidth_limit("upload", 50) == 50
    assert manager.get_status()["upload_bandwidth_mbps"] == 50
    assert manager._upload_limiter.bytes_per_second == 50_000_000
    assert manager._download_limiter.bytes_per_second == 70_000_000
