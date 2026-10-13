"""Recording, disk integrity, idempotent publication and real LeRobot integration."""

import copy
import json
import queue
import subprocess
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from collect_samples import make_streams

from utils import lerobot_recording as recording

ROOT = Path(__file__).resolve().parents[2]


class Executor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, path):
        future = Future()
        self.jobs.append((future, fn, path))
        return future

    def shutdown(self, wait=True):
        for future, _, _ in self.jobs:
            if not future.done():
                future.set_result(None)


def fill_cache(cache, config, count=80):
    data, ts = make_streams(config, count)
    for device, array in data.items():
        for timestamp, row in zip(ts[device], array):
            cache.record(device, round(timestamp * 1e9), row)
    return data, ts


def test_observed_gaps_ignore_duplicates_and_survive_sealing(collect_config):
    """Source gaps measure observed updates, independently of poll cadence."""
    session = recording.RecordingSession(collect_config, executor=Executor())
    try:
        cache = session.start_episode()
        device = collect_config["inputs"]["leader_joint_pos"]
        for timestamp in [1_000_000_000, 1_000_000_000, 1_010_000_000, 1_030_000_000]:
            cache.record(device, timestamp, np.zeros(14))
        session.end_episode()
        stats = json.loads((cache.path / "manifest.json").read_text())["observed_timestamp_gaps"][device]
        assert stats == {
            "count": 2,
            "total_ns": 30_000_000,
            "max_ns": 20_000_000,
            "backwards": 0,
        }
    finally:
        session.close()


def test_config_and_import_without_models(tmp_path):
    """Hydra composes one consumer level; CLI inspection has no hardware/model import."""
    from hydra import compose, initialize_config_dir

    from consumer.lerobot_data_saver import resolved_consumer_config

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="collect",
            overrides=[
                "camera.ego_id=7",
                "consumer.task_name=test task",
                f"consumer.output_dir={tmp_path}",
                "consumer.conversion.window_size=23",
            ],
        )
        config = resolved_consumer_config(cfg)
    assert "consumer" not in config
    assert config["features"]["actions"]["shape"] == [23, 14]
    assert config["inputs"]["img_ego"] == "OpenCVCameraDevice_7"
    assert config["task_name"] == "test task"
    assert config["output_dir"] == str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import consumer; import consumer.lerobot_data_saver; import sys; assert 'policy' not in sys.modules; assert 'nmx_nedf_api' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "type", "nedf"),
        (None, "task_name", ""),
        ("conversion", "data_mode", "online"),
        ("conversion", "reward_range", "signed"),
        ("conversion", "window_size", 0),
        ("conversion", "fps", 0),
        ("conversion", "max_time_diff_ms", -1),
        ("capture", "queue_size", 0),
        ("capture", "poll_interval", 0),
        ("capture", "max_pending_episodes", 0),
        ("writer", "image_writer_threads", 0),
    ],
)
def test_invalid_config(collect_config, section, key, value):
    """Reject unsupported recipes before opening SHM or creating a recording."""
    (collect_config if section is None else collect_config[section])[key] = value
    with pytest.raises(ValueError):
        recording.validate_config(collect_config)


def test_config_schema_validation(collect_config):
    mutations = [
        lambda c: c["inputs"].pop("img_ego"),
        lambda c: c["outputs"].pop("value"),
        lambda c: c["outputs"]["value"].update(stride_basis="sample_row"),
        lambda c: c["outputs"]["action"].update(frame_stride=0),
        lambda c: c["conversion"]["latency_s"].update(img_ego=float("nan")),
        lambda c: c["features"].pop("state"),
        lambda c: c["features"]["actions"].update(shape=[99, 14]),
        lambda c: c["features"]["reward"].update(shape=[99]),
        lambda c: c["features"]["state"].update(shape=[7]),
        lambda c: c["features"]["value"].update(dtype="float64"),
        lambda c: c["features"]["base_image"].update(dtype="image"),
        lambda c: c["features"]["next_base_image"].update(shape=[32, 16, 3]),
        lambda c: c["writer"]["video"].update(vcodec="invalid"),
    ]
    for mutation in mutations:
        config = copy.deepcopy(collect_config)
        mutation(config)
        with pytest.raises(ValueError):
            recording.validate_config(config)


def test_cache_copy_dedup_and_integrity(collect_config, tmp_path):
    """SHM mutation cannot change queued data; duplicate timestamps do not add samples."""
    cache = recording.EpisodeCache(tmp_path / "episode_000000_temp", collect_config)
    device = collect_config["inputs"]["leader_joint_pos"]
    value = np.ones(14)
    assert cache.record(device, 10, value)
    value[:] = 9
    assert not cache.record(device, 10, value)
    assert not cache.record(device, 0, value)
    with pytest.raises(KeyError):
        cache.record("unknown", 11, value)
    cache.close(normal_end=True)
    cache.close()
    with pytest.raises(RuntimeError):
        cache.record(device, 12, value)
    data, ts = recording.load_cached_streams(cache.path, cache.manifest)
    np.testing.assert_array_equal(data[device], np.ones((1, 14)))
    assert ts[device][0] == 1e-8
    stream = cache.manifest["streams"][device]["file"]
    (cache.path / "cache" / f"{stream}.timestamps").write_bytes(b"")
    with pytest.raises(ValueError, match="timestamps"):
        recording.load_cached_streams(cache.path, cache.manifest)
    (cache.path / "cache" / f"{stream}.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="Incomplete cache"):
        recording.load_cached_streams(cache.path, cache.manifest)


def test_cache_failure_discard_and_drop(collect_config, tmp_path, monkeypatch):
    """Writer errors drain queues; discarded or incomplete records cannot be finalized."""
    cache = recording.EpisodeCache(tmp_path / "bad_temp", collect_config)
    device = collect_config["inputs"]["leader_joint_pos"]
    cache.record(device, 1, np.zeros(14))
    cache.record(device, 2, np.zeros(7))
    cache.record(device, 3, np.zeros(14))
    cache.close(normal_end=True)
    assert cache.error
    assert not cache.manifest["normal_end"]
    with pytest.raises(ValueError, match="normally ended"):
        recording.finalize_episode(cache.path)
    cache = recording.EpisodeCache(tmp_path / "discard_temp", collect_config)
    original = cache.queue.put

    def full(*args, **kwargs):
        raise queue.Full

    monkeypatch.setattr(cache.queue, "put", full)
    assert not cache.record(device, 1, np.zeros(14))
    assert cache.dropped[device] == 1
    monkeypatch.setattr(cache.queue, "put", original)
    cache.close(discard=True)
    assert not cache.path.exists()


def test_session_backlog_numbering_and_lock(collect_config):
    """Two sealed episodes apply backpressure; existing directories are never overwritten."""
    executor = Executor()
    session = recording.RecordingSession(collect_config, executor=executor)
    with pytest.raises(RuntimeError, match="owns"):
        recording.RecordingSession(collect_config)
    (session.root / "episode_000000").mkdir()
    first = session.start_episode()
    assert first.path.name == "episode_000001_temp"
    with pytest.raises(RuntimeError):
        session.start_episode()
    session.end_episode()
    session.start_episode()
    session.end_episode()
    assert not session.can_start()
    executor.jobs[0][0].set_exception(RuntimeError("injected encoding failure"))
    assert session.can_start()
    assert session.errors
    cache = session.start_episode()
    session.end_episode(discard=True)
    assert not cache.path.exists()
    cache = session.start_episode()
    session.close()
    assert json.loads((cache.path / "manifest.json").read_text())["status"] == "interrupted"


def test_finalize_pending_only_sealed(collect_config):
    executor = Executor()
    session = recording.RecordingSession(collect_config, executor=executor)
    cache = session.start_episode()
    cache.close(normal_end=True)
    session.active = None
    interrupted = session.start_episode()
    session.end_episode(interrupted=True)
    session.finalize_pending()
    assert [p for _, _, p in executor.jobs] == [str(cache.path)]
    assert interrupted.path.exists()
    session.close()


def test_failure_retry_and_row_selection(collect_config, tmp_path, monkeypatch):
    """A partial value failure leaves no published pair; retry preserves source identity."""
    cache = recording.EpisodeCache(tmp_path / "episode_000000_temp", collect_config)
    fill_cache(cache, collect_config, count=240)
    cache.close(normal_end=True)
    seen = {}

    def fail_value(root, frames, indices, config):
        root.mkdir()
        if root.name == "value":
            raise RuntimeError("encoder failed")

    monkeypatch.setattr(recording, "write_lerobot_output", fail_value)
    with pytest.raises(RuntimeError, match="encoder failed"):
        recording.finalize_episode(cache.path)
    assert cache.path.exists() and (cache.path / "cache").exists()
    assert not (tmp_path / "episode_000000").exists()

    def save(root, frames, indices, config):
        root.mkdir()
        seen[root.name] = (indices, [dict(frames[i]) for i in indices])

    monkeypatch.setattr(recording, "write_lerobot_output", save)
    final = Path(recording.finalize_episode(cache.path))
    manifest = json.loads((final / "manifest.json").read_text())
    assert manifest["source_episode_id"] == cache.manifest["source_episode_id"]
    assert manifest["status"] == "complete"
    assert not (final / "cache").exists()
    action_indices = seen["action"][0]
    # The fixture compensates the camera delay before selecting raw action indices.
    assert seen["value"][0] == [0, 30, 60, 90, 120, 150, 180, 210, 237]
    action_by_index = dict(zip(action_indices, seen["action"][1]))
    for source_index, value_frame in zip(*seen["value"]):
        for key in collect_config["features"]:
            np.testing.assert_array_equal(value_frame[key], action_by_index[source_index][key])
    existing = recording.EpisodeCache(tmp_path / "episode_000000_temp", collect_config)
    existing.close(normal_end=True)
    with pytest.raises(FileExistsError):
        recording.finalize_episode(existing.path)


def test_real_lerobot_roundtrip(collect_config, tmp_path):
    """Real parquet + six videos in both repos can be read after atomic directory rename."""
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    cache = recording.EpisodeCache(tmp_path / "episode_000000_temp", collect_config)
    fill_cache(cache, collect_config, count=120)
    cache.close(normal_end=True)
    final = Path(recording.finalize_episode(cache.path))
    datasets = {}
    for name in ("action", "value"):
        root = final / name
        info = json.loads((root / "meta" / "info.json").read_text())
        assert info["fps"] == 30 and info["total_episodes"] == 1
        assert len(list((root / "videos").rglob("*.mp4"))) == 6
        table = pq.read_table(next((root / "data").rglob("*.parquet")))
        assert table["frame_index"].to_pylist() == list(range(table.num_rows))
        assert set(table["episode_index"].to_pylist()) == {0}
        assert "value_prediction" not in table.column_names
        if name == "action":
            assert info["features"]["advantage"] == {
                "dtype": "float32",
                "shape": [1],
                "names": ["advantage"],
            }
            assert table["advantage"].to_numpy().dtype == np.dtype("float32")
            np.testing.assert_array_equal(table["advantage"].to_numpy(), np.ones(table.num_rows))
            stats = json.loads((root / "meta/episodes_stats.jsonl").read_text())
            for statistic in ("min", "max", "mean"):
                np.testing.assert_array_equal(stats["stats"]["advantage"][statistic], [1.0])
        else:
            assert "advantage" not in table.column_names
            assert "advantage" not in info["features"]
        datasets[name] = LeRobotDataset(f"collect/test_{name}", root=root, video_backend="pyav")
        sample = datasets[name][0]
        assert sample["actions"].shape == (100, 14)
        assert sample["base_image"].shape == (3, 16, 16)
        if name == "action":
            assert sample["advantage"].item() == 1.0
        assert datasets[name][-1]["terminate"].item() == 1
    assert len(datasets["value"]) == 5
    np.testing.assert_array_equal(datasets["action"][9]["actions"], datasets["value"][1]["actions"])

    # The operator's local/cloud check must accept identical transferred payloads
    # and reject a missing completion marker or changed source manifest.
    import runpy
    import shutil

    verify_collection = runpy.run_path(str(Path(__file__).parents[2] / "scripts/verify_collect_lerobot.py"))[
        "verify_collection"
    ]

    expected = verify_collection(tmp_path)
    remote = tmp_path / "remote"
    uploaded = remote / final.name
    shutil.copytree(final, uploaded)
    with pytest.raises(ValueError, match="Upload not complete"):
        verify_collection(remote, require_complete=True, expected=expected)
    (uploaded / "complete.txt").write_text("1")
    assert verify_collection(remote, require_complete=True, expected=expected) == expected
    manifest_path = uploaded / "manifest.json"
    altered = json.loads(manifest_path.read_text())
    altered["task"] = "changed after upload"
    manifest_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="reports differ"):
        verify_collection(remote, require_complete=True, expected=expected)


def test_runtime_dependency_check(monkeypatch):
    """Reject the datasets 4 Column API before connecting to any hardware."""
    monkeypatch.setattr(recording, "version", lambda name: "4.6.1")
    with pytest.raises(RuntimeError, match="datasets<4"):
        recording.check_runtime()
    monkeypatch.setattr(recording, "version", lambda name: "3.6.0")
    recording.check_runtime()


def test_finalize_pending_cli_spawn(collect_config, tmp_path):
    """CLI recovery starts a real worker, uses the sealed config, and ignores interrupted data."""
    output = Path(collect_config["output_dir"])
    output.mkdir()
    cache = recording.EpisodeCache(output / "episode_000000_temp", collect_config)
    fill_cache(cache, collect_config, count=12)
    cache.close(normal_end=True)
    interrupted = recording.EpisodeCache(output / "episode_000001_temp", collect_config)
    interrupted.close()
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "consumer/lerobot_data_saver.py"),
            "--config-name",
            "collect",
            "--finalize-pending",
            f"consumer.output_dir={output}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "episode_000000/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["config"]["features"]["base_image"]["shape"] == [16, 16, 3]
    assert interrupted.path.exists()
    assert not (output / "episode_000001").exists()


def test_incomplete_video_retains_source(collect_config, tmp_path, monkeypatch):
    """A nonempty MP4 with missing frames must not be published as a valid dataset."""
    from lerobot.common.datasets import video_utils

    original = video_utils.encode_video_frames

    def truncate(image_dir, *args, **kwargs):
        max(image_dir.glob("*.png")).unlink()
        return original(image_dir, *args, **kwargs)

    monkeypatch.setattr(video_utils, "encode_video_frames", truncate)
    cache = recording.EpisodeCache(tmp_path / "episode_000000_temp", collect_config)
    fill_cache(cache, collect_config, count=12)
    cache.close(normal_end=True)
    with pytest.raises(ValueError, match="video frame count"):
        recording.finalize_episode(cache.path)
    assert (cache.path / "cache").is_dir()
    assert json.loads((cache.path / "manifest.json").read_text())["status"] == "failed"
    assert not (tmp_path / "episode_000000").exists()
