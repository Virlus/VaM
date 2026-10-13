"""Real pinned LeRobot writer/reader, followed by cloud p70 without re-encoding."""

import shutil

import numpy as np

from utils.lerobot_direct import validate_direct_episode
from utils.lerobot_online.protocol import sha256_file
from preprocess_data import register_lerobot_uploads as uploads
from argparse import Namespace


def test_real_action_value_and_cloud_label(policy, request, tmp_path):
    policy.update(policy_id=0, checkpoint_index=0)
    recorded = request.getfixturevalue("recorded")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import DataLoader

    manifest = validate_direct_episode(recorded, decode=True)
    assert manifest["outputs"]["action"]["episode_lengths"] == [19]
    assert manifest["outputs"]["value"]["episode_count"] == 2
    assert manifest["advantage_status"] == "cloud_p70_pending"
    assert not list((recorded.parent.parent / "capture").iterdir())
    reader = LeRobotDataset("collection/action", root=recorded / "action", video_backend="pyav")
    batch = next(iter(DataLoader(reader, batch_size=2)))
    assert batch["actions"].shape == (2, 10, 14)
    assert batch["base_image"].shape == (2, 3, 16, 16)
    assert "advantage_raw" in batch and "advantage" not in batch
    value = LeRobotDataset("collection/value", root=recorded / "value", video_backend="pyav")
    assert value.num_episodes == 2
    assert value[0]["success"].item() == 1
    assert value[len(value) - 1]["success"].item() == 1
    args = Namespace(
        inbox=tmp_path / "inbox",
        dataset_home=tmp_path / "datasets",
        cloud_root=tmp_path / "cloud",
        repo_prefix="example/run",
        action_offline_repo="example/offline_action",
        value_offline_repo="example/offline_value",
    )
    for role in ("action", "value"):
        shutil.copytree(recorded / role, args.dataset_home / getattr(args, f"{role}_offline_repo"))
    bundle = args.inbox / recorded.name
    shutil.copytree(recorded, bundle)
    (bundle / "complete.txt").touch()
    before = {p.relative_to(bundle): sha256_file(p) for p in bundle.rglob("*.mp4")}
    state = {"uploads": {}, "published": True}
    state_path = args.cloud_root / "conversion_state/lerobot_uploads.json"
    uploads.run_once(args, state, state_path)
    record = state["uploads"][bundle.name]
    cloud = args.dataset_home / record["repos"]["action"]["repo_id"]
    labeled = LeRobotDataset("collection/cloud", root=cloud, video_backend="pyav")
    raw = np.array([row["advantage_raw"].item() for row in labeled])
    flags = np.array([row["control_flag"].item() for row in labeled])
    labels = np.array([row["advantage"].item() for row in labeled])
    threshold = np.percentile(raw, 70)
    np.testing.assert_array_equal(labels, (raw >= threshold) | (flags == 1))
    assert before == {p.relative_to(bundle): sha256_file(p) for p in bundle.rglob("*.mp4")}
    snapshots = {
        role: (args.cloud_root / "dataset_views" / f"{role}_latest.json").read_bytes() for role in ("action", "value")
    }
    uploads.run_once(args, state, state_path)
    for role in ("action", "value"):
        assert (args.cloud_root / "dataset_views" / f"{role}_latest.json").read_bytes() == snapshots[role]


def test_stride_three_real_writer(config, request):
    """Sampled Parquet, audit indices and all video lengths survive validation."""
    import json
    import pyarrow.parquet as pq

    config["action_stride"] = 3
    recorded = request.getfixturevalue("recorded")
    manifest = validate_direct_episode(recorded, decode=True)
    assert manifest["recipe"]["action_stride"] == 3
    assert manifest["outputs"]["action"]["frame_count"] < 19
    indices = []
    for path in sorted((recorded / "action/data").rglob("*.parquet")):
        indices.extend(pq.read_table(path)["source_action_index"].to_pylist())
    indices = np.asarray(indices).reshape(-1)
    assert np.all(indices[:-1] % 3 == 0)
    assert indices[-1] == 60
    audit = json.loads((recorded / "alignment.json").read_text())["action"]
    assert [row["frame_index"] for row in audit] == list(range(len(indices)))
    np.testing.assert_array_equal([row["source_action_index"] for row in audit], indices)
