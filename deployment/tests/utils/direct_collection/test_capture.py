"""Real ring layouts and saver lifecycle; no robot hardware or NEDF SDK."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from consumer.async_infer_data_saver_lerobot import DirectLeRobotSaver
from utils.lerobot_direct import (
    DirectSession,
    finalize_direct_episode,
    finalize_with_progress,
)
from utils.lerobot_online.protocol import atomic_json, read_json
from utils.lerobot_shm import RingStream
from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    pack_buffer_header,
    pack_device_header,
    pack_frame_header,
)


def test_recovered_conversion_uses_run_history_index(config, tmp_path, monkeypatch):
    """A cache's old session index must not appear as the run-level index."""
    from concurrent.futures import Future

    from utils import lerobot_direct

    path = Path(config["output_dir"]) / "capture" / "old_temp"
    atomic_json(
        path / "manifest.json",
        {
            "episode_uid": "old",
            "rollout_index": 0,
            "normal_end": True,
        },
    )
    config["history_rollout_indices"] = {"old": 5}
    config["progress_dir"] = str(tmp_path / "progress")
    future = Future()
    executor = Mock()
    executor.submit.return_value = future
    session = DirectSession(config, executor=executor)
    try:
        session.reap()
        args = executor.submit.call_args.args
        assert args == (finalize_with_progress, str(path), config["progress_dir"], 5)

        def check_processing(source, **kwargs):
            status = read_json(Path(config["progress_dir"]) / "worker.json")
            assert status["episode"] == "5" and status["state"] == "processing"
            assert status["stage"] == "loading"
            return "ready"

        monkeypatch.setattr(lerobot_direct, "finalize_direct_episode", check_processing)
        future.set_result(args[0](*args[1:]))
        assert read_json(path / "manifest.json")["rollout_index"] == 0
    finally:
        if not future.done():
            future.set_result(None)
        session.close()


def test_new_session_displays_latest_completed_episode_of_same_run(config, tmp_path):
    import uuid

    config["progress_dir"] = str(tmp_path / "progress")
    for index, run_name in [(7, config["run_name"]), (9, "another-run")]:
        name = f"episode_{index:03d}"
        atomic_json(
            Path(config["output_dir"]) / "ready" / name / "manifest.json",
            {
                "episode_uid": str(uuid.uuid4()),
                "episode_name": name,
                "rollout_index": index,
                "run_name": run_name,
                "result": "success",
                "start_ns": index,
            },
        )
    session = DirectSession(config)
    try:
        assert session.last_rollout_index == 7
        status = read_json(Path(config["progress_dir"]) / "worker.json")
        assert status["episode"] == "7" and status["state"] == "finished"
    finally:
        session.close()


class Ring:
    def __init__(self, capacity=8, latest=False):
        self.capacity, self.latest = capacity, latest
        self.frame_size = FRAME_HEADER_SIZE + 4
        self.buf = bytearray(DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + capacity * self.frame_size)
        self.buf[:DEVICE_HEADER_SIZE] = pack_device_header("test", 0, 100, "float32", (1,), self.frame_size)
        self.n = 0
        self.header(0, 0)

    def header(self, count, cursor):
        self.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = pack_buffer_header(
            self.capacity, count, cursor
        )

    def write(self, timestamp):
        index = self.n % self.capacity
        offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + index * self.frame_size
        self.buf[offset : offset + FRAME_HEADER_SIZE] = pack_frame_header(timestamp)
        self.buf[offset + FRAME_HEADER_SIZE : offset + self.frame_size] = np.array(
            [timestamp], dtype=np.float32
        ).tobytes()
        self.n += 1
        self.header(min(self.n, self.capacity), index if self.latest else self.n % self.capacity)

    def close(self):
        pass


@pytest.mark.parametrize("latest", [False, True])
def test_ring_wrap_dedup_and_overrun(latest):
    channel = Ring(latest=latest)
    reader = RingStream("test", channel=channel, index_is_latest=latest)
    assert reader.read_new() == []
    for ts in range(1, 5):
        channel.write(ts)
    assert [ts for ts, _ in reader.read_new()] == [1, 2, 3, 4]
    assert reader.read_new() == []
    channel.write(4)
    assert reader.read_new() == []
    for ts in range(5, 10):
        channel.write(ts)
        assert [t for t, _ in reader.read_new()] == [ts]
    for ts in range(10, 18):
        channel.write(ts)
    with pytest.raises(RuntimeError, match="overrun"):
        reader.read_new()
    reader.reset(15)
    assert [t for t, _ in reader.read_new()] == [15, 16, 17]
    channel.write(1)
    with pytest.raises(RuntimeError, match="regressed"):
        reader.read_new()
    reader.close()


def test_signal_ack_seal_discard_and_interruption(config, policy, samples):
    executor = Mock()
    executor.submit.return_value.done.return_value = False
    session = DirectSession(config, executor=executor)
    readers = {name: Mock(read_new=Mock(return_value=[])) for name in config["inputs"].values()}
    saver = DirectLeRobotSaver(config, session=session, readers=readers)
    saver.connect()
    assert saver.signals["rollout_ready"].read_text() == "ready"
    start = {"request_id": "start-1", "policy": policy, "start_ns": samples[2]}
    atomic_json(saver.signals["start_recording"], start)
    saver.tick()
    assert read_json(str(saver.signals["start_recording"]) + ".ack")["state"] == "recording"
    active = session.active
    atomic_json(saver.signals["start_recording"], start)
    saver.tick()
    assert session.active is active
    for name in samples[0]:
        readers[name].read_new.return_value = list(zip(samples[1][name], samples[0][name], strict=True))
    atomic_json(
        saver.signals["stop_recording"],
        {"request_id": "stop-1", "result": "failure", "stop_ns": samples[3]},
    )
    saver.tick()
    manifest = read_json(active.path / "manifest.json")
    assert manifest["result"] == "failure" and manifest["success"] is False
    assert manifest["normal_end"]
    assert session.last_result == "failure"
    assert "result=FAILURE" in session.event["message"]
    assert session.event["level"] == "success"  # Failed task, successful recording.
    executor.submit.assert_called_once_with(finalize_with_progress, str(active.path), None, None)
    assert read_json(str(saver.signals["stop_recording"]) + ".ack")["state"] == "sealed"
    assert session.start(policy, samples[2])
    discarded = session.active.path
    session.stop("discard", samples[3])
    assert session.last_result == "discard" and session.last_rollout_index == 1
    assert "no conversion" in session.event["message"]
    executor.submit.assert_called_once()  # Discard never creates a conversion job.
    assert not discarded.exists()
    assert session.start(policy, samples[2])
    interrupted = session.active.path
    executor.submit.return_value.done.return_value = True
    session.close()
    assert read_json(interrupted / "manifest.json")["normal_end"] is False


def test_failed_finalizer_keeps_cache_and_backpressure(config, policy, samples):
    from concurrent.futures import Future

    executor = Mock()
    failed = Future()
    failed.set_exception(ValueError("missing value"))
    executor.submit.return_value = failed
    session = DirectSession(config, executor=executor)
    assert session.start(policy, samples[2])
    retained = session.active.path
    session.stop("success", samples[3])
    with pytest.raises(RuntimeError, match="retained failed"):
        session.start(policy, samples[2])
    assert retained.exists()
    session.close()
    executor.submit.return_value = Future()
    recovered = DirectSession(config, executor=executor)
    recovered.reap()
    assert str(retained) in recovered.futures
    recovered.futures["other"] = Future()
    assert recovered.start(policy, samples[2]) is False
    assert recovered.phase == "wait conversion"
    for future in recovered.futures.values():
        future.set_result(None)
    recovered.close()


def test_saver_connection_timeout_and_cache_rejection(config, monkeypatch):
    config["capture"]["connect_timeout_s"] = 0
    monkeypatch.setattr(
        "consumer.async_infer_data_saver_lerobot.RingStream",
        Mock(side_effect=FileNotFoundError),
    )
    session = Mock()
    saver = DirectLeRobotSaver(config, session=session)
    with pytest.raises(TimeoutError, match="SHM missing"):
        saver.run()
    session.close.assert_called_once()
    session.active.record.return_value = False
    saver.readers = {"state": SimpleNamespace(read_new=lambda: [(1, np.zeros(14))])}
    with pytest.raises(RuntimeError, match="rejected"):
        saver.collect()


def test_missing_and_dropped_sources_never_publish(config, policy, samples):
    from concurrent.futures import Future

    executor = Mock()
    executor.submit.return_value = Future()
    session = DirectSession(config, executor=executor)
    session.start(policy, samples[2])
    path = session.active.path
    session.stop("success", samples[3])
    with pytest.raises(ValueError, match="Missing captured"):
        finalize_direct_episode(path)
    manifest = read_json(path / "manifest.json")
    manifest["dropped"]["state"] = 1
    atomic_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="dropped"):
        finalize_direct_episode(path)
    assert not (Path(config["output_dir"]) / "ready").exists()
    executor.submit.return_value.set_result(None)
    session.close()
