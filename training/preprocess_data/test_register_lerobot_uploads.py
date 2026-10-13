"""Exercise direct registration with real LeRobot v2.1 datasets and existing views."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
from lerobot.common.datasets import lerobot_dataset
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from preprocess_data import register_lerobot_uploads as uploads
from preprocess_data.utils.dataset_registry import _atomic_write_json


def encode_small_video(img_dir, video_path, fps, **_kwargs):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    images = [cv2.imread(str(path)) for path in sorted(img_dir.glob("*.png"))]
    height, width = images[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for image in images:
        writer.write(image)
    writer.release()


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    root = tmp_path_factory.mktemp("recorded")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lerobot_dataset, "encode_video_frames", encode_small_video)
        for role, count in (("action", 1), ("value", 2)):
            features = {}
            for prefix in ("", "next_"):
                features[prefix + "state"] = {"dtype": "float32", "shape": (14,), "names": None}
                features[prefix + "actions"] = {"dtype": "float32", "shape": (100, 14), "names": None}
                for camera in ("base_image", "left_wrist_image", "right_wrist_image"):
                    features[prefix + camera] = {
                        "dtype": "video",
                        "shape": (16, 16, 3),
                        "names": ["height", "width", "channel"],
                    }
            for key in (
                "value_prediction",
                "next_value_prediction",
                "reward",
                "terminate",
                "policy_id",
                "checkpoint_index",
                "data_source",
                "control_flag",
                "advantage",
                "advantage_raw",
            ):
                features[key] = {"dtype": "float32", "shape": (1,), "names": None}
            features["source_action_index"] = {"dtype": "int64", "shape": (1,), "names": None}
            dataset = lerobot_dataset.LeRobotDataset.create(
                f"fixture/{role}",
                fps=30,
                root=root / role,
                features=features,
                robot_type="aloha_i2rt",
                video_backend="pyav",
            )
            for _episode in range(count):
                for _frame in range(3):
                    frame = {
                        key: np.zeros(spec["shape"], dtype=np.uint8 if spec["dtype"] == "video" else spec["dtype"])
                        for key, spec in features.items()
                    }
                    frame["data_source"][:] = 1
                    frame["advantage"][:] = 1
                    frame["source_action_index"][:] = _frame * 30
                    frame["task"] = "fold the t-shirt"
                    dataset.add_frame(frame)
                dataset.save_episode()
    for role, count in (("action", 1), ("value", 2)):
        (root / role / "meta/policy_metadata_map.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "target_episode_index": index,
                        "policy_id": 0,
                        "checkpoint_index": 0,
                        "collector_prefix_annotation": f"prefix_{index}",
                    }
                )
                + "\n"
                for index in range(count)
            )
        )
    (root / "episode_policy.json").write_text(json.dumps({"policy_id": 0, "checkpoint_index": 0}))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "sample_semantics": "lerobot_camera_v1",
                "recipe": {
                    "fps": 30,
                    "window_size": 1,
                    "advantage": {"action_fps": 100, "gamma": 0.99, "gae_lambda": 0.9, "horizon": 1},
                },
            }
        )
    )
    (root / "complete.txt").touch()
    return root


@pytest.fixture
def setup(tmp_path, recorded):
    args = argparse.Namespace(
        inbox=tmp_path / "inbox",
        dataset_home=tmp_path / "datasets",
        cloud_root=tmp_path / "cloud",
        repo_prefix="aloha/run",
        action_offline_repo="aloha/offline_action",
        value_offline_repo="aloha/offline_value",
    )
    for role in ("action", "value"):
        shutil.copytree(recorded / role, args.dataset_home / getattr(args, f"{role}_offline_repo"))
    bundle = args.inbox / "episode_000001"
    shutil.copytree(recorded, bundle)
    state = {"uploads": {}, "published": True}
    state_path = args.cloud_root / "conversion_state/lerobot_uploads.json"
    return args, state, state_path, bundle


def latest_view(args, role):
    pointer = uploads.read_json(args.cloud_root / "dataset_views" / f"{role}_latest.json")
    return uploads.read_json(Path(pointer["view_path"]))


def test_register_load_repeat_and_shared_policy_repo(setup):
    args, state, state_path, bundle = setup
    uploads.run_once(args, state, state_path)
    record = state["uploads"][bundle.name]
    assert record["repos"]["action"]["episode_ids"] == [0]
    assert record["repos"]["value"]["episode_ids"] == [0, 1]
    for role in ("action", "value"):
        repo_id = record["repos"][role]["repo_id"]
        root = args.dataset_home / repo_id
        for source in (bundle / role).rglob("*"):
            if source.is_file():
                target = root / source.relative_to(bundle / role)
                if source.suffix == ".parquet":
                    old, new = pq.read_table(source), pq.read_table(target)
                    for key in old.column_names:
                        if key != "advantage":
                            assert old[key].equals(new[key])
                elif source.suffix == ".mp4":
                    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest()
        dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root, video_backend="pyav")
        assert dataset[0]["base_image"].shape == (3, 16, 16)
        assert dataset[0]["actions"].shape == (100, 14)
        view = latest_view(args, role)
        assert view["schema_version"] == "dataset_view/v2"
        assert view["repos"][repo_id]["episode_ids"] == record["repos"][role]["episode_ids"]
    before = latest_view(args, "action")
    uploads.run_once(args, state, state_path)
    assert latest_view(args, "action") == before
    shutil.copytree(bundle, args.inbox / "episode_000002")
    uploads.run_once(args, state, state_path)
    assert len(state["uploads"]) == 2
    assert len(latest_view(args, "action")["repos"]) == 2
    # Load the emitted view through the unmodified training control plane.
    from vam.training.async_control.dataset_view import load_dataset_view

    pointer = uploads.read_json(args.cloud_root / "dataset_views/action_latest.json")
    view = load_dataset_view(Path(pointer["view_path"]))
    assert len(view.repo_specs) == 2
    assert view.target_policy_id == 0


def test_partial_and_invalid_uploads_not_published(setup):
    args, state, state_path, bundle = setup
    (bundle / "complete.txt").unlink()
    uploads.run_once(args, state, state_path)
    assert not state["uploads"]
    (bundle / "complete.txt").touch()
    next((bundle / "value/videos").rglob("*.mp4")).unlink()
    uploads.run_once(args, state, state_path)
    assert not state["uploads"]
    assert not (args.cloud_root / "dataset_views/action_latest.json").exists()


def test_retry_after_first_repo_copy(setup, monkeypatch):
    args, state, state_path, _bundle = setup
    original = uploads.append_uploaded_episodes

    def interrupt(source, destination, *a, **kw):
        if Path(source).name == "value":
            raise OSError("simulated upload copy failure")
        return original(source, destination, *a, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(uploads, "append_uploaded_episodes", interrupt)
        with pytest.raises(RuntimeError, match="Append interrupted"):
            uploads.run_once(args, state, state_path)
    assert not state["uploads"]
    uploads.run_once(args, state, state_path)
    assert len(state["uploads"]) == 1
    assert latest_view(args, "value")["target_policy_id"] == 0


def test_restart_repairs_second_view(setup, monkeypatch):
    args, state, state_path, _bundle = setup
    original = uploads.publish_dataset_view

    def interrupt(**kwargs):
        if kwargs["latest_name"] == "value_latest.json":
            raise OSError("simulated second view failure")
        return original(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(uploads, "publish_dataset_view", interrupt)
        with pytest.raises(OSError, match="simulated second view failure"):
            uploads.run_once(args, state, state_path)
    state = uploads.read_json(state_path)
    assert not state["published"]
    uploads.run_once(args, state, state_path)
    assert state["published"]
    assert len(state["uploads"]) == 1
    assert latest_view(args, "value")["target_policy_id"] == 0


def test_future_and_stale_policy_gate(setup):
    args, state, state_path, bundle = setup
    _atomic_write_json(bundle / "episode_policy.json", {"policy_id": 1, "checkpoint_index": 10})
    uploads.run_once(args, state, state_path)
    assert not state["uploads"]
    checkpoint = args.cloud_root / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "DONE").touch()
    _atomic_write_json(
        args.cloud_root / "checkpoints/latest.json", {"policy_id": 2, "checkpoint_path": str(checkpoint)}
    )
    uploads.run_once(args, state, state_path)
    assert not state["uploads"]


def set_inputs(bundle, raw, *, flags=(0, 0, 0), pid=0):
    for role in ("action", "value"):
        root = bundle / role
        for path in (root / "data").rglob("*.parquet"):
            table = pq.read_table(path)
            for name, values in (("policy_id", [pid] * 3), ("checkpoint_index", [pid * 10] * 3)):
                table = table.set_column(table.column_names.index(name), name, pa.array([[float(x)] for x in values]))
            if role == "action":
                for name, values in (("advantage_raw", raw), ("control_flag", flags)):
                    table = table.set_column(
                        table.column_names.index(name), name, pa.array([[float(x)] for x in values])
                    )
                # With zero reward/next value, endpoint GAE is exactly -V(current).
                table = table.set_column(
                    table.column_names.index("value_prediction"),
                    "value_prediction",
                    pa.array([[-float(x)] for x in raw]),
                )
                table = table.drop(["advantage"])
                metadata = dict(table.schema.metadata)
                hf = json.loads(metadata[b"huggingface"])
                del hf["info"]["features"]["advantage"]
                metadata[b"huggingface"] = json.dumps(hf).encode()
                table = table.replace_schema_metadata(metadata)
            pq.write_table(table, path)
        if role == "action":
            info = uploads.read_json(root / "meta/info.json")
            del info["features"]["advantage"]
            stats_path = root / "meta/episodes_stats.jsonl"
            rows = uploads.read_rows(stats_path)
            for row in rows:
                row["stats"].pop("advantage", None)
            stats_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            _atomic_write_json(root / "meta/info.json", info)
        rows = uploads.read_rows(root / "meta/policy_metadata_map.jsonl")
        for row in rows:
            row.update(policy_id=pid, checkpoint_index=pid * 10)
        (root / "meta/policy_metadata_map.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    _atomic_write_json(bundle / "episode_policy.json", {"policy_id": pid, "checkpoint_index": pid * 10})


def action_labels(args, record):
    root = args.dataset_home / record["repos"]["action"]["repo_id"]
    dataset = lerobot_dataset.LeRobotDataset(record["repos"]["action"]["repo_id"], root=root, video_backend="pyav")
    ids = record["repos"]["action"]["episode_ids"]
    return [
        dataset[i]["advantage"].item()
        for i in range(len(dataset))
        if int(dataset.hf_dataset[i]["episode_index"]) in ids
    ]


def test_policy_wide_raw_labels_and_isolation(setup, recorded):
    args, state, state_path, bundle = setup
    set_inputs(bundle, [0, 1, 2], flags=(1, 0, 0))
    original = {str(p.relative_to(bundle)): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    uploads.run_once(args, state, state_path)
    first = state["uploads"][bundle.name]
    assert action_labels(args, first) == [1, 0, 1]
    from torch.utils.data import DataLoader

    from scripts.train_async import AsyncTrainStream

    root = args.dataset_home / first["repos"]["action"]["repo_id"]
    old_dataset = lerobot_dataset.LeRobotDataset(first["repos"]["action"]["repo_id"], root=root, video_backend="pyav")
    old_loader = DataLoader(old_dataset, batch_size=3)
    stream = AsyncTrainStream(
        name="bc_action",
        default_train_type_values=("bc",),
        train_type_by_target_policy_id={},
        train_types=set(),
        data_config=None,
        watcher=None,
        data_loader=old_loader,
        data_iter=iter(old_loader),
        batch=None,
        steps_per_cycle=1,
        ptrain_step=None,
        episode_replay_state=None,
    )
    stream._next_batch = next(iter(old_loader))  # noqa: SLF001
    second = args.inbox / "second"
    shutil.copytree(recorded, second)
    set_inputs(second, [3, 4, 5])
    uploads.run_once(args, state, state_path)
    assert state["annotations"]["0"]["threshold"] == 3.5
    assert state["annotations"]["0"]["frame_count"] == 6
    assert action_labels(args, first) == [1, 0, 0]
    assert old_dataset[2]["advantage"].item() == 1  # existing in-memory data is stale
    new_dataset = lerobot_dataset.LeRobotDataset(first["repos"]["action"]["repo_id"], root=root, video_backend="pyav")
    new_loader = DataLoader(new_dataset, batch_size=3)
    new_iter = iter(new_loader)
    stream.replace_data_pipeline(
        data_config=None,
        data_loader=new_loader,
        data_iter=new_iter,
        batch=next(new_iter),
        skip_next_batch_prefetch_once=True,
        enable_device_prefetch_after_step=False,
        context="annotation test",
    )
    assert stream._next_batch is None  # noqa: SLF001
    assert stream.batch["advantage"].reshape(-1).tolist() == [1, 0, 0]
    assert action_labels(args, state["uploads"]["second"]) == [0, 1, 1]
    for relative, data in original.items():
        assert (bundle / relative).read_bytes() == data
    root = args.dataset_home / first["repos"]["action"]["repo_id"]
    stats = uploads.read_rows(root / "meta/episodes_stats.jsonl")[0]["stats"]["advantage"]
    assert stats["mean"][0] == pytest.approx(1 / 3)
    assert stats["count"] == [3]
    _atomic_write_json(args.cloud_root / "checkpoints/latest.json", {"policy_id": 1})
    third = args.inbox / "third"
    shutil.copytree(recorded, third)
    set_inputs(third, [-30, -20, -10], pid=1)
    uploads.run_once(args, state, state_path)
    assert state["annotations"]["0"]["threshold"] == 3.5
    assert state["annotations"]["1"]["threshold"] == -16
    assert action_labels(args, first) == [1, 0, 0]


def test_annotation_interruption_and_label_only_refresh(setup, monkeypatch):
    args, state, state_path, bundle = setup
    set_inputs(bundle, [0, 1, 2])
    original = uploads.atomic_write_parquet

    def interrupt(table, path):
        original(table, path)
        raise OSError("annotation interrupted after parquet replace")

    with monkeypatch.context() as patch:
        patch.setattr(uploads, "atomic_write_parquet", interrupt)
        with pytest.raises(OSError, match="annotation interrupted"):
            uploads.run_once(args, state, state_path)
    assert not (args.cloud_root / "dataset_views/action_latest.json").exists()
    state = uploads.read_json(state_path)
    assert state["uploads"][bundle.name]["status"] == "copied"
    assert state["pending_policies"] == [0]
    # Pending admission survives checkpoint advancement.
    _atomic_write_json(args.cloud_root / "checkpoints/latest.json", {"policy_id": 1})
    uploads.run_once(args, state, state_path)
    assert action_labels(args, state["uploads"][bundle.name]) == [0, 0, 1]
    before = latest_view(args, "action")
    value_before = latest_view(args, "value")
    state["pending_policies"] = [0]
    _atomic_write_json(state_path, state)
    uploads.run_once(args, state, state_path)
    after = latest_view(args, "action")
    assert after["repos"].keys() == before["repos"].keys()
    for repo_id in before["repos"]:
        assert after["repos"][repo_id]["episode_ids"] == before["repos"][repo_id]["episode_ids"]
    online_repo = state["uploads"][bundle.name]["repos"]["action"]["repo_id"]
    assert after["repos"][online_repo]["advantage_snapshot"] != before["repos"][online_repo]["advantage_snapshot"]
    assert after["annotation_revision"] > before["annotation_revision"]
    assert "annotation_revision" not in latest_view(args, "value")
    assert latest_view(args, "value") == value_before
    # A stream without a dual-source loader still cannot take the live-refresh fast path.
    from types import SimpleNamespace

    from scripts.train_async import _try_grow_dual_source_in_place
    from vam.training.async_control.dataset_view import load_dataset_view

    pointer = uploads.read_json(args.cloud_root / "dataset_views/action_latest.json")
    view = load_dataset_view(pointer["view_path"])
    assert not _try_grow_dual_source_in_place(SimpleNamespace(), view, step=1, wandb_logger=None)


@pytest.mark.parametrize("raw", [[0, float("nan"), 2], [0, float("inf"), 2]])
def test_nonfinite_raw_rejected(setup, raw):
    args, state, state_path, bundle = setup
    set_inputs(bundle, raw)
    uploads.run_once(args, state, state_path)
    assert not state["uploads"]
    assert not (args.cloud_root / "dataset_views/action_latest.json").exists()


def test_second_upload_annotation_failure_preserves_view(setup, recorded, monkeypatch):
    args, state, state_path, bundle = setup
    set_inputs(bundle, [0, 1, 2])
    uploads.run_once(args, state, state_path)
    before = latest_view(args, "action")
    second = args.inbox / "second"
    shutil.copytree(recorded, second)
    set_inputs(second, [3, 4, 5])
    original = uploads.atomic_write_parquet
    writes = []

    def interrupt(table, path):
        original(table, path)
        writes.append(path)
        if len(writes) == 1:
            raise OSError("interrupted old episode relabel")

    with monkeypatch.context() as patch:
        patch.setattr(uploads, "atomic_write_parquet", interrupt)
        with pytest.raises(OSError, match="old episode"):
            uploads.run_once(args, state, state_path)
    assert latest_view(args, "action") == before
    state = uploads.read_json(state_path)
    uploads.run_once(args, state, state_path)
    assert action_labels(args, state["uploads"][bundle.name]) == [0, 0, 0]
    assert action_labels(args, state["uploads"]["second"]) == [0, 1, 1]
    assert latest_view(args, "action")["annotation_revision"] == 2


def test_online_launcher_registers_lerobot(setup, tmp_path):
    """One YAML selects registration, preserves role views and survives a restart."""
    import subprocess
    import sys

    import yaml

    from run_online import ROOT

    args, _state, _state_path, _bundle = setup
    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "include": str(ROOT / "configs/online.yaml"),
                "variables": {
                    "run_name": "fixture",
                    "split_root": str(tmp_path / "split"),
                    "action_offline_repo": args.action_offline_repo,
                    "value_offline_repo": args.value_offline_repo,
                },
                "runtime": {"python": {"ingest": sys.executable}},
                "job": {
                    "cloud_root": str(args.cloud_root),
                    "inbox": str(args.inbox),
                    "dataset_home": str(args.dataset_home),
                    "offline_dataset_home": str(args.dataset_home),
                    "repo_prefix": args.repo_prefix,
                    "run_once": True,
                },
            }
        )
    )
    command = [sys.executable, str(ROOT / "run_online.py"), str(config)]
    for _ in range(2):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
        assert result.returncode == 0, result.stderr
        assert latest_view(args, "action")["target_policy_id"] == 0
        assert latest_view(args, "value")["target_policy_id"] == 0
    assert not (tmp_path / "split/work/fixture/converter/configs/live_action.yaml").exists()


def test_camera_migration_restores_upload_raw_and_preserves_old_snapshot(setup):
    args, state, state_path, bundle = setup
    set_inputs(bundle, [0, 1, 2])
    source = next((bundle / "action/data").rglob("*.parquet"))
    original_bytes = source.read_bytes()
    uploads.run_once(args, state, state_path)
    record = state["uploads"][bundle.name]
    root = args.dataset_home / record["repos"]["action"]["repo_id"]
    registered = next((root / "data").rglob("*.parquet"))
    np.testing.assert_array_equal(np.asarray(pq.read_table(registered)["advantage_raw"]).ravel(), [0, 1, 2])
    assert action_labels(args, record) == [0, 0, 1]
    old_snapshot = root / record["repos"]["action"]["advantage_snapshot"]
    old_snapshot_bytes = old_snapshot.read_bytes()
    value_before = latest_view(args, "value")
    revision = state["annotation_revision"]
    # Simulate a registry and raw values written by the action-endpoint algorithm.
    state["annotations"]["0"]["algorithm"] = "source_action_endpoint_gae/v1"
    state["uploads"][bundle.name].pop("advantage_recipe")
    table = pq.read_table(registered)
    table = table.set_column(table.column_names.index("advantage_raw"), "advantage_raw", pa.array([[99.0]] * 3))
    pq.write_table(table, registered)
    _atomic_write_json(state_path, state)
    uploads.run_once(args, state, state_path)
    assert state["annotation_revision"] == revision + 1
    assert state["annotations"]["0"]["algorithm"] == uploads.ADVANTAGE_ALGORITHM
    assert state_path.with_suffix(".before_camera_advantage.json").exists()
    np.testing.assert_array_equal(np.asarray(pq.read_table(registered)["advantage_raw"]).ravel(), [0, 1, 2])
    assert uploads.read_rows(root / "meta/episodes_stats.jsonl")[0]["stats"]["advantage_raw"]["mean"] == [1.0]
    assert old_snapshot.read_bytes() == old_snapshot_bytes
    assert source.read_bytes() == original_bytes
    assert latest_view(args, "value") == value_before
    uploads.run_once(args, state, state_path)
    assert state["annotation_revision"] == revision + 1


@pytest.mark.parametrize("missing", ["manifest", "advantage_raw"])
def test_camera_inputs_must_not_fall_back_to_action_endpoint(setup, missing):
    args, state, state_path, bundle = setup
    if missing == "manifest":
        (bundle / "manifest.json").unlink()
    else:
        path = next((bundle / "action/data").rglob("*.parquet"))
        pq.write_table(pq.read_table(path).drop([missing]), path)
    uploads.run_once(args, state, state_path)
    assert not state["uploads"]
    assert not (args.cloud_root / "dataset_views/action_latest.json").exists()


def test_advantage_recipe_cannot_change_within_policy(setup, recorded):
    args, state, state_path, bundle = setup
    uploads.run_once(args, state, state_path)
    second = args.inbox / "second"
    shutil.copytree(recorded, second)
    manifest = uploads.read_json(second / "manifest.json")
    manifest["recipe"]["advantage"]["gamma"] = 0.5
    _atomic_write_json(second / "manifest.json", manifest)
    uploads.run_once(args, state, state_path)
    assert list(state["uploads"]) == [bundle.name]


def test_camera_raw_does_not_use_action_endpoint_value(setup):
    args, state, state_path, bundle = setup
    set_inputs(bundle, [0, 1, 2])
    source = next((bundle / "action/data").rglob("*.parquet"))
    table = pq.read_table(source)
    table = table.set_column(
        table.column_names.index("next_value_prediction"), "next_value_prediction", pa.array([[999.0]] * 3)
    )
    pq.write_table(table, source)
    uploads.run_once(args, state, state_path)
    assert action_labels(args, state["uploads"][bundle.name]) == [0, 0, 1]
