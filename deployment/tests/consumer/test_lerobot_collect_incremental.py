"""Real two-episode datasets, immutable snapshots, and interrupted remote commits."""

import copy
import json
import os
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from collect_samples import make_streams

from utils import lerobot_recording as recording
from utils.lerobot_collect_publish import check_offline_streams, publish_episode
from utils.lerobot_collect_upload import poll_collect
from utils.lerobot_online.protocol import read_json


@pytest.fixture
def published(collect_config):
    config = collect_config
    config["publication"] = {"repo_ids": {"action": "aloha/test", "value": "aloha/test_downsampled"}}
    config["outputs"]["action"]["frame_stride"] = 3
    root = Path(config["output_dir"])
    root.mkdir()
    data, timestamps = make_streams(config, 120)
    sources = []
    for number in range(2):
        cache = recording.EpisodeCache(root / f"episode_{number:06d}_temp", config)
        for device, values in data.items():
            for ts, value in zip(timestamps[device], values):
                cache.record(device, round((ts + number * 10) * 1e9), value)
        cache.close(normal_end=True)
        sources.append(Path(recording.finalize_episode(cache.path)))
    return config, sources


class SFTP:
    open = staticmethod(open)
    stat = staticmethod(os.stat)
    mkdir = staticmethod(os.mkdir)
    rmdir = staticmethod(os.rmdir)
    listdir = staticmethod(os.listdir)
    posix_rename = staticmethod(os.replace)
    close = Mock()


def uploader(config, tmp_path):
    sftp = SFTP()
    ssh = SimpleNamespace(open_sftp=lambda: sftp, close=Mock())

    def transfer(sftp, source, destination):
        shutil.copyfile(source, destination)
        return True

    manager = SimpleNamespace(
        data_dir=str(Path(config["output_dir"]) / "ready"),
        remote_path=str(tmp_path / "remote"),
        _stop_event=threading.Event(),
        _get_ssh_client=lambda **kw: ssh,
        _ensure_remote_dirs_recursive=lambda ftp, path: Path(path).mkdir(parents=True, exist_ok=True),
        _rollout_upload_method=lambda: "sftp",
        _resume_rollout_upload_enabled=lambda: True,
        _upload_file_via_sftp_from_scratch=Mock(side_effect=transfer),
        _upload_file_via_sftp_resumable=Mock(side_effect=transfer),
        _upload_log=Mock(),
    )
    return manager, sftp


def check_dataset(root):
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    info = read_json(root / "meta/info.json")
    assert info["total_episodes"] == 2
    assert info["total_videos"] == 12
    lengths = [json.loads(line)["length"] for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    assert sum(lengths) == info["total_frames"]
    stats = [json.loads(line) for line in (root / "meta/episodes_stats.jsonl").read_text().splitlines()]
    for index, table_path in enumerate(sorted((root / "data").rglob("*.parquet"))):
        table = pq.read_table(table_path)
        assert set(table["episode_index"].to_pylist()) == {index}
        offset = sum(lengths[:index])
        assert table["index"].to_pylist() == list(range(offset, offset + lengths[index]))
        assert stats[index]["stats"]["episode_index"]["mean"] == [index]
        assert stats[index]["stats"]["index"]["min"] == [offset]
    dataset = LeRobotDataset("collect/test", root=root, video_backend="pyav")
    assert len(dataset) == sum(lengths)
    for index in (0, lengths[0] - 1, lengths[0], len(dataset) - 1):
        assert dataset[index]["base_image"].shape == (3, 16, 16)
    return lengths


def test_two_episodes_upload_retry_and_metadata(published, tmp_path):
    """Interrupted metadata publication retries before episode 2; repeat polls never regress."""
    config, sources = published
    root = Path(config["output_dir"])
    first = root / "ready/episode_000000/action/meta/info.json"
    frozen = first.read_bytes()
    assert read_json(first)["total_episodes"] == 1
    for repo in config["publication"]["repo_ids"].values():
        check_dataset(root / "datasets" / repo)
    assert not list((root / "ready").rglob("cache"))
    manager, sftp = uploader(config, tmp_path)
    rename = sftp.posix_rename
    failed = False

    def fail_once(source, dest):
        nonlocal failed
        if dest.endswith("/meta/info.json") and not failed:
            failed = True
            raise OSError("network interrupted")
        rename(source, dest)

    sftp.posix_rename = fail_once
    with pytest.raises(OSError, match="interrupted"):
        poll_collect(manager)
    state_file = next((Path(manager.remote_path) / ".collect_uploads").glob("*.json"))
    assert read_json(state_file)["receipts"] == []
    assert not (Path(manager.remote_path) / ".collect-upload.lock").exists()
    poll_collect(manager)
    state = read_json(state_file)
    assert len(state["receipts"]) == 2 and state["pending"] is None
    for repo in config["publication"]["repo_ids"].values():
        check_dataset(Path(manager.remote_path) / repo)
    calls = manager._upload_file_via_sftp_from_scratch.call_count
    poll_collect(manager)
    assert manager._upload_file_via_sftp_from_scratch.call_count == calls
    assert first.read_bytes() == frozen
    # A late local retry of episode 0 must not regress cumulative metadata to 1.
    assert publish_episode(sources[0], config, read_json(sources[0] / "manifest.json")).endswith("episode_000000")
    check_dataset(root / "datasets" / config["publication"]["repo_ids"]["action"])


def test_refuses_existing_repo_and_stops_order(published, tmp_path):
    config, _ = published
    manager, _ = uploader(config, tmp_path)
    occupied = Path(manager.remote_path) / config["publication"]["repo_ids"]["action"]
    occupied.mkdir(parents=True)
    (occupied / "old.txt").write_text("existing dataset")
    with pytest.raises(ValueError, match="unowned"):
        poll_collect(manager)
    assert (occupied / "old.txt").read_text() == "existing dataset"
    manager._upload_file_via_sftp_from_scratch.assert_not_called()


def test_rsync_fallback(published, tmp_path):
    config, _ = published
    manager, _ = uploader(config, tmp_path)
    manager._rollout_upload_method = lambda: "rsync"
    manager._rollout_rsync_fallback_enabled = lambda: True
    manager._build_rollout_rsync_command = lambda *args: ["rsync"]
    manager._run_external_transfer = Mock(side_effect=OSError("missing rsync"))
    poll_collect(manager)
    assert manager._run_external_transfer.call_count == 2
    check_dataset(Path(manager.remote_path) / config["publication"]["repo_ids"]["value"])


@pytest.mark.parametrize("problem", ["missing", "short", "gap", "frozen"])
def test_legacy_checks(collect_config, problem):
    """Only the four legacy checks are applied; bad raw streams never reach ready."""
    config = copy.deepcopy(collect_config)
    config["anomaly_detection"] = {"min_frames": 70, "max_timestamp_gap": 1.0}
    data, ts = make_streams(config, 300)
    check_offline_streams(data, ts, config)
    device = config["inputs"]["leader_joint_pos"]
    if problem == "missing":
        data.pop(device)
    elif problem == "short":
        data[device] = data[device][:2]
    elif problem == "gap":
        ts[device][-1] += 2
    else:
        data[device][:] = 0
    with pytest.raises(ValueError):
        check_offline_streams(data, ts, config)


def test_main_config_and_stride():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="collect",
            overrides=["lerobot.action_stride=3", "run_name=test_batch"],
        )
        config = recording.validate_config(OmegaConf.to_container(cfg.consumer, resolve=True))
        assert config["outputs"]["action"]["frame_stride"] == 3
        assert Path(cfg.augmental_data_upload.data_dir).resolve() == (Path(config["output_dir"]) / "ready")
        assert config["publication"]["repo_ids"]["action"] == "test_batch"
    for invalid in (0, -1, 1.5, True):
        config["outputs"]["action"]["frame_stride"] = invalid
        with pytest.raises(ValueError):
            recording.validate_config(config)


def test_local_materialization_recovery(published, monkeypatch):
    """Ready remains authoritative if local materialization was interrupted."""
    from utils import lerobot_collect_publish as publish

    config, sources = published
    root = Path(config["output_dir"])
    shutil.rmtree(root / "datasets")
    (root / ".materialized.json").unlink()
    original = publish.materialize_local
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("local disk interrupted")
        return original(*args)

    monkeypatch.setattr(publish, "materialize_local", interrupted)
    manifest = read_json(sources[0] / "manifest.json")
    with pytest.raises(OSError, match="disk interrupted"):
        publish.publish_episode(sources[0], config, manifest)
    assert read_json(root / ".materialized.json")["sequence"] == 0
    monkeypatch.setattr(publish, "materialize_local", original)
    publish.publish_episode(sources[0], config, manifest)
    assert read_json(root / ".materialized.json")["sequence"] == 1
    check_dataset(root / "datasets" / config["publication"]["repo_ids"]["action"])


def test_snapshot_schema_conflict(published):
    """A different task/schema cannot silently extend an existing dataset."""
    from utils.lerobot_collect_publish import _append_role

    config, sources = published
    root = Path(config["output_dir"])
    source = sources[0] / "action"
    previous = root / "ready/episode_000001/action"
    info_file = source / "meta/info.json"
    info = read_json(info_file)
    info["fps"] = 15
    info_file.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="changed LeRobot fps"):
        _append_role(source, root / "bad", previous, 2)
    info["fps"] = 30
    info_file.write_text(json.dumps(info))
    (source / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"other"}\n')
    with pytest.raises(ValueError, match="Task changed"):
        _append_role(source, root / "bad", previous, 2)


def test_rsync_success_and_changed_completed_manifest(published, tmp_path):
    config, _ = published
    manager, _ = uploader(config, tmp_path)
    manager._rollout_upload_method = lambda: "rsync"
    manager._build_rollout_rsync_command = lambda source, target: [source, target]
    manager._run_external_transfer = lambda command, label: shutil.copytree(command[0], command[1], dirs_exist_ok=True)
    poll_collect(manager)
    manager._upload_file_via_sftp_from_scratch.assert_not_called()
    first = Path(manager.data_dir) / "episode_000000/manifest.json"
    manifest = read_json(first)
    manifest["note"] = "changed"
    first.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Previously uploaded episode changed"):
        poll_collect(manager)


def test_incomplete_payload_and_missing_predecessor(published, tmp_path):
    config, _ = published
    manager, _ = uploader(config, tmp_path)
    first = Path(manager.data_dir) / "episode_000000"
    payload = next((first / "action/data").rglob("*.parquet"))
    original = payload.read_bytes()
    payload.write_bytes(b"broken")
    with pytest.raises(ValueError, match="inventory changed"):
        poll_collect(manager)
    payload.write_bytes(original)
    first.rename(first.with_name(".hidden_first"))
    with pytest.raises(ValueError, match="Missing predecessor"):
        poll_collect(manager)
    manager._upload_file_via_sftp_from_scratch.assert_not_called()


def test_manager_dispatch_and_stop(published, tmp_path, monkeypatch):
    from consumer.data_upload_manager import DataUploadManager
    from utils import lerobot_collect_upload

    config, _ = published
    manager, _ = uploader(config, tmp_path)
    manager._stop_event.set()
    poll_collect(manager)
    manager._upload_file_via_sftp_from_scratch.assert_not_called()
    send = Mock()
    monkeypatch.setattr(lerobot_collect_upload, "poll_collect", send)
    manager.upload_config = {"payload_format": "lerobot_collect_v1"}
    DataUploadManager._poll_and_upload(manager)
    send.assert_called_once_with(manager)


@pytest.mark.parametrize(
    "repos",
    [
        {"action": "../bad", "value": "good"},
        {"action": "/bad", "value": "good"},
        {"action": "same", "value": "same"},
        {"action": "parent", "value": "parent/child"},
        {"action": ".collect_uploads/bad", "value": "good"},
    ],
)
def test_invalid_repo_paths(repos):
    from utils.lerobot_collect_publish import validate_publication

    with pytest.raises(ValueError):
        validate_publication({"repo_ids": repos})


def test_ready_only_layout_and_restart(published):
    """Completed sources disappear; ready preserves diagnostics and restart numbering."""
    config, bundles = published
    root = Path(config["output_dir"])
    assert not list(root.glob("episode_*"))
    for bundle in bundles:
        manifest = read_json(bundle / "manifest.json")
        assert manifest["recording"]["status"] == "complete"
        assert manifest["recording"]["source_episode_id"] == manifest["source_episode_id"]
        assert manifest["source_episode_name"].startswith("episode_")
    session = recording.RecordingSession(config)
    try:
        cache = session.start_episode()
        assert cache.path.name == "episode_000002_temp"
        session.end_episode(discard=True)
    finally:
        session.close()


def test_committed_cleanup_recovery_without_cache(published):
    """A committed episode can finish cleanup even if raw cache was already removed."""
    config, bundles = published
    manifest = read_json(bundles[0] / "manifest.json")["recording"]
    retained = Path(config["output_dir"]) / "episode_000000_temp"
    retained.mkdir()
    (retained / "manifest.json").write_text(json.dumps(manifest))
    assert recording.finalize_episode(retained) == str(bundles[0])
    assert not retained.exists()
    assert len(list((Path(config["output_dir"]) / "ready").glob("episode_*"))) == 2


def test_start_index_directories_and_contiguous_dataset(published, tmp_path):
    """Directory offset does not create holes in LeRobot episode_index or upload sequence."""
    config, bundles = published
    config = copy.deepcopy(config)
    config["start_index"] = 10
    config["output_dir"] = str(tmp_path / "offset_batch")
    Path(config["output_dir"]).mkdir()
    source = bundles[0]
    recording_manifest = read_json(source / "manifest.json")["recording"]
    first = Path(publish_episode(source, config, recording_manifest))
    assert first.name == "episode_000010"
    assert read_json(first / "manifest.json")["sequence"] == 0
    assert read_json(first / "action/meta/info.json")["total_episodes"] == 1
    import uuid

    second_manifest = copy.deepcopy(recording_manifest)
    second_manifest["source_episode_id"] = str(uuid.uuid4())
    second = Path(publish_episode(source, config, second_manifest))
    assert second.name == "episode_000011"
    assert read_json(second / "manifest.json")["sequence"] == 1
    manager, _ = uploader(config, tmp_path)
    poll_collect(manager)
    check_dataset(Path(manager.remote_path) / config["publication"]["repo_ids"]["action"])
    session = recording.RecordingSession(config)
    try:
        assert session.start_episode().path.name == "episode_000012_temp"
        session.end_episode(discard=True)
    finally:
        session.close()
    config["start_index"] = 11
    second_manifest["source_episode_id"] = str(uuid.uuid4())
    with pytest.raises(ValueError, match="start_index changed"):
        publish_episode(source, config, second_manifest)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_start_index(collect_config, value):
    collect_config["start_index"] = value
    with pytest.raises(ValueError, match="start_index"):
        recording.validate_config(collect_config)


@pytest.mark.parametrize("first_fails", [False, True])
def test_parallel_conversion_publishes_in_submission_order(collect_config, first_fails):
    """A later conversion cannot publish first; failures do not deadlock drain."""
    from test_lerobot_recording import Executor

    config = collect_config
    config["capture"].update(conversion_workers=2, max_pending_episodes=2)
    config["publication"] = {"repo_ids": {"action": "test", "value": "test_value"}}
    executor = Executor()
    session = recording.RecordingSession(config, executor=executor)
    data, timestamps = make_streams(config, 120)
    paths = []
    try:
        for number in range(2):
            cache = session.start_episode()
            paths.append(cache.path)
            for device, values in data.items():
                for ts, value in zip(timestamps[device], values):
                    cache.record(device, round((ts + number * 10) * 1e9), value)
            session.end_episode()
        assert not session.can_start()
        future, fn, path = executor.jobs[1]
        future.set_result(fn(path))
        assert not list(session.root.glob("ready/episode_*"))
        assert not session.futures[1].done()
        future, fn, path = executor.jobs[0]
        if first_fails:
            future.set_exception(RuntimeError("encoding failed"))
        else:
            future.set_result(fn(path))
    finally:
        session.close()
    bundles = sorted(session.root.glob("ready/episode_*/manifest.json"))
    names = [read_json(p)["source_episode_name"] for p in bundles]
    expected = paths[1:] if first_fails else paths
    assert names == [p.name.removesuffix("_temp") for p in expected]
    assert len(session.errors) == int(first_fails)
    assert all(not p.exists() for p in expected)
    if first_fails:
        assert paths[0].exists()


@pytest.mark.parametrize("key", ["conversion_workers", "max_pending_episodes"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_background_limits(collect_config, key, value):
    collect_config["capture"][key] = value
    with pytest.raises(ValueError, match=key):
        recording.validate_config(collect_config)
