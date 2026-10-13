"""Frozen outputs from the deployed cloud converter, not a second local recipe."""

import json
from pathlib import Path

import numpy as np
import pytest

from utils.lerobot_direct import OnlineFrames

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "direct_legacy"


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "failure",
        "takeover",
        "compacted_takeover",
        "gaps",
        "sparse_values",
        "epoch_rgb",
    ],
)
@pytest.mark.parametrize("action_stride", [1, 3])
def test_cloud_legacy_arrays_are_preserved(case, action_stride):
    """Sampling preserves every retained legacy field, including dense GAE."""
    source = FIXTURES / case
    metadata = json.loads((source / "source.json").read_text())
    metadata["config"]["action_stride"] = action_stride
    with np.load(source / "source.npz") as archive:
        data = {key: archive[key] for key in archive.files if not key.startswith("ts_")}
        timestamps = {key: archive["ts_" + key] for key in data}
    frames = OnlineFrames(
        data,
        timestamps,
        metadata["config"],
        metadata["policy"],
        metadata["result"],
        metadata["start_ns"],
        metadata["stop_ns"],
    )
    for role, ends in (
        ("action", [frames.end]),
        ("value", [prefix["end_index"] for prefix in frames.prefixes]),
    ):
        assert len(ends) == len(list(source.glob(f"{role}_*.npz")))
        for episode, end in enumerate(ends):
            rows = list(frames.frames(role, end, episode))
            with np.load(source / f"{role}_{episode}.npz") as reference:
                selected = np.ones(len(reference[reference.files[0]]), dtype=bool)
                if role == "action":
                    dense_indices = np.array(sorted(frames.observations))
                    assert len(dense_indices) == len(selected)
                    selected = dense_indices % action_stride == 0
                    selected[-1] = True
                for key in reference.files:
                    actual = np.stack([row[key] for row in rows])
                    assert actual.dtype == reference[key].dtype, key
                    np.testing.assert_array_equal(
                        actual,
                        reference[key][selected],
                        err_msg=f"{case}/{role}/{episode}/{key}",
                    )


def test_production_recipe_preserves_legacy_options():
    from hydra import compose, initialize_config_dir

    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="example")
    rollout = cfg["policy"]["infer_rollout"]
    recipe = rollout["lerobot"]
    assert recipe["action_stride"] == 3
    assert recipe["value_stride"] == 30
    assert rollout["excluded_control_flag_values"] == [0.5]
    assert recipe["alignment_tolerance_ms"] == 250
    assert recipe["capture"]["preroll_s"] == recipe["capture"]["value_drain_s"] == 0
    assert recipe["writer"]["video"] == {
        "vcodec": "libsvtav1",
        "pix_fmt": "yuv420p",
        "g": 2,
        "crf": 30,
        "fast_decode": 0,
    }
