"""Compatibility with the frozen pre-refactor offline converter, not a new formula."""

from pathlib import Path

import numpy as np
import pytest
from collect_samples import make_streams

from utils.lerobot_recording import _builder_module


def builder(config):
    return _builder_module()(
        features=config["features"],
        hardware_latency_s=config["conversion"]["latency_s"],
    )


@pytest.mark.parametrize("count", [12, 80, 240])
def test_pre_refactor_reference(collect_config, count):
    """Short windows, 2-window lookahead, terminal alignment and every saved field."""
    data, timestamps = make_streams(collect_config, count)
    frames = builder(collect_config).build_offline_frames(
        data, timestamps, collect_config["inputs"], collect_config["task_name"]
    )
    reference = np.load(Path(__file__).parents[1] / "fixtures" / "collect_offline_reference.npz")
    for key in collect_config["features"]:
        np.testing.assert_array_equal(np.stack([f[key] for f in frames]), reference[f"{count}/{key}"])
    assert all(f["task"] == collect_config["task_name"] for f in frames)
    assert frames[-1]["terminate"].item() == 1


def test_empty_missing_and_misaligned(collect_config, streams):
    """Missing streams and unalignable camera timestamps cannot create successful output."""
    data, ts = streams
    device = collect_config["inputs"]["img_left"]
    for modified in (
        {k: v for k, v in ts.items() if k != device},
        {**ts, device: np.array([])},
    ):
        with pytest.raises(ValueError, match="Empty or missing"):
            builder(collect_config).build_offline_frames(data, modified, collect_config["inputs"], "test")
    ts[device] = ts[device] + 1000
    with pytest.raises(ValueError, match="No valid"):
        builder(collect_config).build_offline_frames(data, ts, collect_config["inputs"], "test")


def test_timestamp_validation_and_duplicate(collect_config, streams):
    """Input integrity is checked, and equal timestamps retain legacy nearest-frame behavior."""
    data, ts = streams
    device = collect_config["inputs"]["img_ego"]
    bad = {**ts, device: ts[device][:-1]}
    with pytest.raises(ValueError, match="length mismatch"):
        builder(collect_config).build_offline_frames(data, bad, collect_config["inputs"], "test")
    bad = {**ts, device: ts[device][::-1]}
    with pytest.raises(ValueError, match="decreasing"):
        builder(collect_config).build_offline_frames(data, bad, collect_config["inputs"], "test")
    ts[device][1] = ts[device][0]
    frames = builder(collect_config).build_offline_frames(data, ts, collect_config["inputs"], "test")
    assert frames


def test_unequal_cameras_and_missing_future(collect_config, streams):
    """The shortest camera is the reference; unavailable future observations use legacy fallback."""
    data, ts = streams
    device = collect_config["inputs"]["img_right"]
    data[device], ts[device] = data[device][:5], ts[device][:5]
    frames = builder(collect_config).build_offline_frames(data, ts, collect_config["inputs"], "test")
    assert len(frames) == 5
    np.testing.assert_array_equal(frames[-1]["next_right_wrist_image"], data[device][-1])
