"""Native encoder output stays in logs; failures never swallow terminal output."""

import ctypes
import os
import subprocess
import sys

import pytest

from utils.lerobot_conversion_logging import conversion_output, quiet_lerobot_writer


def test_python_native_and_child_output_are_captured(tmp_path, capfd):
    log = tmp_path / "conversion.log"
    with conversion_output(log):
        print("python stdout")
        print("python stderr", file=sys.stderr)
        os.write(1, b"native stdout\n")
        os.write(2, b"native stderr\n")
        ctypes.CDLL(None).printf(b"buffered native output\n")
        subprocess.run([sys.executable, "-c", "print('child stdout')"], check=True)
    assert capfd.readouterr() == ("", "")
    for message in (
        "python stdout",
        "python stderr",
        "native stdout",
        "native stderr",
        "buffered native output",
        "child stdout",
    ):
        assert message in log.read_text()
    print("terminal restored")
    os.write(2, b"native restored\n")
    captured = capfd.readouterr()
    assert captured.out == "terminal restored\n"
    assert captured.err == "native restored\n"


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_failure_restores_output_and_retains_traceback(tmp_path, capfd, error):
    log = tmp_path / "conversion.log"
    with pytest.raises(error, match="encoder failed"):
        with conversion_output(log):
            raise error("encoder failed")
    assert capfd.readouterr() == ("", "")
    assert "encoder failed" in log.read_text()
    os.write(1, b"after failure\n")
    assert capfd.readouterr().out == "after failure\n"


def test_writer_keeps_logs_outside_dataset_and_preserves_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @quiet_lerobot_writer
    def writer(root, *, fail=False):
        print("encoder details")
        if fail:
            raise ValueError("bad frame")
        return 42

    root = tmp_path / "building" / "episode-id" / "action"
    assert writer(root=root) == 42
    with pytest.raises(ValueError, match="bad frame") as raised:
        writer(root, fail=True)
    log = tmp_path / "logs/lerobot_conversion/episode-id_action.log"
    assert str(log) in raised.value.__notes__[0]
    assert log.read_text().count("encoder details") == 2
    assert not root.exists()


def test_real_av1_encoder_is_quiet_and_output_decodes(tmp_path, capfd):
    """Exercise the actual native encoder used by the robot, with tiny frames."""
    import av
    import numpy as np
    from PIL import Image
    from lerobot.common.datasets.video_utils import encode_video_frames

    images = tmp_path / "images"
    images.mkdir()
    for index in range(2):
        Image.fromarray(np.full((64, 64, 3), index * 100, dtype=np.uint8)).save(images / f"frame_{index:06d}.png")
    capfd.readouterr()
    video = tmp_path / "test.mp4"
    with conversion_output(tmp_path / "av1.log"):
        encode_video_frames(images, video, 30, vcodec="libsvtav1", overwrite=True)
    assert capfd.readouterr() == ("", "")
    with av.open(str(video)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 2
