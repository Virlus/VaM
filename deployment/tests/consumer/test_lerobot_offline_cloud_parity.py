"""Compare the offline recorder with independent, frozen cloud converter outputs."""

from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from PIL import Image

from utils import lerobot_recording as recording
from utils.lerobot_images import legacy_jpeg_rgb

FIXTURES = Path(__file__).parents[1] / "fixtures" / "collect_offline_cloud"
CASES = ("12", "80", "240", "600", "gaps", "duplicate", "rgb", "rgb64")


def source(case):
    with np.load(FIXTURES / f"source_{case}.npz") as archive:
        data = {key: archive[key] for key in archive.files if not key.startswith("ts_")}
        return data, {key: archive["ts_" + key] for key in data}


def configure_images(config, data):
    for feature in config["features"].values():
        if feature["dtype"] == "video":
            feature["shape"] = list(data[config["inputs"]["img_ego"]].shape[1:])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("action_stride", [1, 30])
def test_finalizer_matches_cloud_reference(collect_config, tmp_path, monkeypatch, case, action_stride):
    """Raw-action stride, last observation, chunks and JPEG pixels match the old converter.

    Replace only the raw-cache reader and dataset sink; exercise production role selection.
    Duplicate timestamps intentionally test alignment independently of capture deduplication.
    Stride-30 action rows must match the shared fields of the frozen value output.
    """
    collect_config["outputs"]["action"]["frame_stride"] = action_stride
    data, timestamps = source(case)
    configure_images(collect_config, data)
    cache = recording.EpisodeCache(tmp_path / "episode_000000_temp", collect_config)
    cache.close(normal_end=True)
    monkeypatch.setattr(recording, "load_cached_streams", lambda *args: (data, timestamps))
    checked = []

    def compare(root, frames, indices, config):
        root.mkdir()
        reference_role = "value" if action_stride == 30 else root.name
        with np.load(FIXTURES / f"expected_{case}_{reference_role}.npz") as expected:
            assert len(indices) == len(expected["state"])
            for key, feature in config["features"].items():
                actual = np.stack(
                    [
                        (legacy_jpeg_rgb(frames[index][key]) if feature["dtype"] == "video" else frames[index][key])
                        for index in indices
                    ]
                )
                assert actual.dtype == expected[key].dtype
                np.testing.assert_array_equal(actual, expected[key], err_msg=f"{case}/{root.name}/{key}")
        checked.append(root.name)
        return {}

    monkeypatch.setattr(recording, "write_lerobot_output", compare)
    recording.finalize_episode(cache.path)
    assert checked == ["action", "value"]


def decoded_video(path):
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def test_saved_av1_and_parquet_match_cloud_reference(collect_config, tmp_path):
    """Exercise cache + real writer: numeric columns and decoded AV1 pixels match.

    Reference video uses cloud-produced JPEG-decoded arrays and the pinned encoder directly;
    it never passes through the recorder's JPEG helper, detecting missing/double compression.
    """
    import pyarrow.parquet as pq
    from lerobot.common.datasets.video_utils import encode_video_frames

    data, timestamps = source("rgb64")
    configure_images(collect_config, data)
    collect_config["writer"]["video"]["vcodec"] = "libsvtav1"
    cache = recording.EpisodeCache(tmp_path / "episode_000000_temp", collect_config)
    for device, rows in data.items():
        for timestamp, row in zip(timestamps[device], rows):
            assert cache.record(device, round(timestamp * 1e9), row)
    cache.close(normal_end=True)
    final = Path(recording.finalize_episode(cache.path))
    for role in ("action", "value"):
        root = final / role
        table = pq.read_table(next((root / "data").rglob("*.parquet")))
        with np.load(FIXTURES / f"expected_rgb64_{role}.npz") as expected:
            assert table.num_rows == len(expected["state"])
            for key, feature in collect_config["features"].items():
                if feature["dtype"] != "video":
                    actual = np.asarray(table[key].to_pylist(), dtype=np.float32)
                    # LeRobot v2.1 stores shape-[1] features as Arrow scalars.
                    actual = actual.reshape(table.num_rows, *feature["shape"])
                    np.testing.assert_array_equal(actual, expected[key], err_msg=f"{role}/{key}")
                    continue
                png_dir = tmp_path / f"reference_{role}_{key}"
                png_dir.mkdir()
                for index, pixels in enumerate(expected[key]):
                    Image.fromarray(pixels).save(png_dir / f"frame_{index:06d}.png")
                video_path = png_dir.with_suffix(".mp4")
                encode_video_frames(
                    png_dir,
                    video_path,
                    30,
                    overwrite=True,
                    **collect_config["writer"]["video"],
                )
                actual_video = root / "videos/chunk-000" / key / "episode_000000.mp4"
                np.testing.assert_array_equal(
                    decoded_video(actual_video),
                    decoded_video(video_path),
                    err_msg=f"{role}/{key}",
                )


def test_jpeg_bgr_rgb_and_no_source_mutation():
    """Online BGR and offline RGB paths must produce the same decoded pixels."""
    image = np.random.default_rng(17).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    before = image.copy()
    np.testing.assert_array_equal(legacy_jpeg_rgb(image), legacy_jpeg_rgb(image[..., ::-1], "bgr"))
    np.testing.assert_array_equal(image, before)
    with pytest.raises(ValueError, match="color order"):
        legacy_jpeg_rgb(image, "invalid")


@pytest.mark.parametrize("stage", ["encode", "decode"])
def test_jpeg_failure_is_explicit(monkeypatch, stage):
    """A codec failure must fail the episode instead of publishing invalid pixels."""
    if stage == "encode":
        monkeypatch.setattr(cv2, "imencode", lambda *args: (False, None))
    else:
        monkeypatch.setattr(cv2, "imdecode", lambda *args: None)
    with pytest.raises(ValueError, match=f"Cannot {stage}"):
        legacy_jpeg_rgb(np.zeros((16, 16, 3), dtype=np.uint8))
