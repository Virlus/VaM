"""Small recordings with real control cadence and distinct human targets."""

from concurrent.futures import Future

import numpy as np
import pytest

from utils.lerobot_direct import STREAMS, DirectSession


class InlineExecutor:
    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True):
        pass


@pytest.fixture
def config(tmp_path):
    return {
        "inputs": {key: key for key in STREAMS},
        "output_dir": str(tmp_path / "recordings"),
        "run_name": "test_run",
        "task_name": "pick cube",
        "window_size": 10,
        "fps": 30,
        "value_stride": 10,
        "alignment_tolerance_ms": 250,
        "image_color_order": "rgb",
        "latency_ms": {},
        "advantage": {
            "mode": "cloud_p70",
            "percentile": 70,
            "gamma": 0.999,
            "gae_lambda": 0.9,
            "horizon": 1,
        },
        "capture": {
            "queue_size": 2000,
            "max_pending_episodes": 2,
            "min_free_bytes": 0,
            "value_drain_s": 0,
        },
        "writer": {
            "image_writer_processes": 0,
            "image_writer_threads": 1,
            "video": {
                "vcodec": "h264",
                "pix_fmt": "yuv420p",
                "g": 2,
                "crf": 28,
                "fast_decode": 0,
            },
        },
        "signal_files": {
            name: str(tmp_path / name)
            for name in (
                "start_recording",
                "stop_recording",
                "episode_result",
                "rollout_ready",
            )
        },
    }


@pytest.fixture
def policy():
    return {
        "policy_id": 2,
        "checkpoint_index": 39999,
        "checkpoint_id": "test_ckpt",
        "checkpoint_dir": "/model/39999",
    }


@pytest.fixture
def samples():
    base = 10_000_000_000
    ticks = base + np.arange(61, dtype=np.int64) * 10_000_000
    cameras = base + np.arange(19, dtype=np.int64) * 100_000_000 // 3
    data, timestamps = {}, {}
    for i, name in enumerate(("img_ego", "img_left", "img_right")):
        data[name] = np.full((len(cameras), 16, 16, 3), 30 + i * 60, dtype=np.uint8)
        timestamps[name] = cameras.copy()
    for name, value in (("state", 3), ("policy_action", 1), ("human_action", 2)):
        data[name] = np.full((len(ticks), 14), value, dtype=np.float32)
        timestamps[name] = ticks.copy()
    data["value_prediction"] = np.linspace(0.1, 0.7, len(ticks), dtype=np.float32)[:, None]
    data["control_flag"] = np.zeros((len(ticks), 1), dtype=np.float32)
    data["control_flag"][20:40] = 1
    timestamps["control_flag"] = ticks.copy()
    timestamps["value_prediction"] = ticks.copy()
    return data, timestamps, int(ticks[0]), int(ticks[-1])


@pytest.fixture
def recorded(config, policy, samples):
    from pathlib import Path

    data, times, start, end = samples
    session = DirectSession(config, executor=InlineExecutor())
    assert session.start(policy, start)
    for name in data:
        for ts, array in zip(times[name], data[name], strict=True):
            assert session.active.record(config["inputs"][name], ts, array)
    session.stop("success", end)
    session.reap()
    session.close()
    assert not session.errors, session.errors
    return next((Path(config["output_dir"]) / "ready").iterdir())
