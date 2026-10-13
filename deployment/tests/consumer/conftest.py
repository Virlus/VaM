"""Hardware-free fixtures for direct collect recording."""

import sys
from pathlib import Path

import pytest
from collect_samples import make_streams
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def collect_config(tmp_path):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="collect")
        config = OmegaConf.to_container(cfg.consumer, resolve=True)
    config["output_dir"] = str(tmp_path / "recordings")
    config["start_index"] = 0  # Independent of the operator's current batch.
    # Legacy writer unit tests keep exercising independent episode output.
    # Incremental publication and legacy anomaly checks have dedicated integration tests.
    config.pop("publication", None)
    config.pop("anomaly_detection", None)
    # Frozen legacy references use dense action rows regardless of the operator's preset.
    config["outputs"]["action"]["frame_stride"] = 1
    # Frozen fixtures were generated with 33 ms camera latency compensation.
    for camera in ("img_ego", "img_left", "img_right"):
        config["conversion"]["latency_s"][camera] = 0.033
    config["writer"]["image_writer_processes"] = 0
    config["writer"]["image_writer_threads"] = 2
    config["writer"]["video"]["vcodec"] = "h264"
    for key, feature in config["features"].items():
        if "image" in key:
            feature["shape"] = [16, 16, 3]
    for key in ("initial_home_wait_s", "episode_home_wait_s", "teleop_start_wait_s"):
        config["capture"][key] = 0
    return config


@pytest.fixture
def streams(collect_config):
    return make_streams(collect_config)
