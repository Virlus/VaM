"""Publish immutable episode deltas with cumulative LeRobot v2.1 metadata.

Videos are linked without re-encoding. Only dataset indices and their statistics
are rebased; sample timestamps, actions, rewards and image payloads stay intact.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path, PurePosixPath

import numpy as np

from utils.lerobot_online.protocol import atomic_json, exclusive_lock, read_json

SCHEMA = "lerobot_collect_delta/v1"
ROLES = ("action", "value")


def relative_path(value):
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Expected relative dataset path: {value!r}")
    return str(path)


def validate_publication(config):
    repos = config["repo_ids"]
    if set(repos) != set(ROLES):
        raise ValueError("Publication requires action and value repo_ids")
    paths = [PurePosixPath(relative_path(repos[role])) for role in ROLES]
    if paths[0] == paths[1] or any(a in b.parents for a, b in (paths, paths[::-1])):
        raise ValueError("Action/value repo paths must be distinct and not nested")
    if any(p.parts[0].startswith(".") for p in paths):
        raise ValueError("Hidden repository roots are reserved for upload state")


def check_offline_streams(data, timestamps, config):
    """The four checks implemented by legacy SimpleAnomalyDetector, in seconds."""
    settings = config.get("anomaly_detection")
    if not settings:
        return
    for role, device in config["inputs"].items():
        if device not in data or not len(data[device]):
            raise ValueError(f"Missing required stream: {role}")
        if len(data[device]) < settings.get("min_frames", 1):
            raise ValueError(f"Too few frames: {role}")
        ts = timestamps[device]
        gap = settings.get("max_timestamp_gap", 0)
        if gap > 0 and len(ts) > 1 and np.max(np.diff(ts)) > gap:
            raise ValueError(f"Timestamp gap exceeds {gap}s: {role}")
        if role.endswith("joint_pos") and len(data[device]) > 1:
            diff = np.max(np.abs(np.diff(data[device], axis=0)), axis=1)
            frozen = np.mean(diff < settings.get("frozen_frame_diff_threshold", 1e-4))
            if frozen > settings.get("frozen_data_threshold", 0.3):
                raise ValueError(f"Frozen joint data: {role} ({frozen:.2%})")


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _append_role(source, target, previous, sequence):
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = read_json(source / "meta/info.json")
    if info["codebase_version"] != "v2.1" or info["total_episodes"] != 1:
        raise ValueError("Expected one complete LeRobot v2.1 source episode")
    old = read_json(previous / "meta/info.json") if previous else None
    if old:
        for key in (
            "features",
            "fps",
            "robot_type",
            "chunks_size",
            "data_path",
            "video_path",
        ):
            if old[key] != info[key]:
                raise ValueError(f"Cannot append changed LeRobot {key}")
        if old["total_episodes"] != sequence:
            raise ValueError("Non-contiguous episode metadata")
        if (previous / "meta/tasks.jsonl").read_bytes() != (source / "meta/tasks.jsonl").read_bytes():
            raise ValueError("Task changed within collection batch; use a new batch")
    offset = old["total_frames"] if old else 0
    args = {"episode_chunk": sequence // info["chunks_size"], "episode_index": sequence}
    source_args = {"episode_chunk": 0, "episode_index": 0}
    table = pq.read_table(source / relative_path(info["data_path"].format(**source_args)))
    if table.num_rows != info["total_frames"] or not table.num_rows:
        raise ValueError("Source Parquet frame count mismatch")
    for key, delta in (("episode_index", sequence), ("index", offset)):
        column = table[key]
        values = np.asarray(column.to_pylist()) + delta
        table = table.set_column(table.schema.get_field_index(key), key, pa.array(values, type=column.type))
    parquet = target / relative_path(info["data_path"].format(**args))
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet)
    for key, feature in info["features"].items():
        if feature["dtype"] == "video":
            src = relative_path(info["video_path"].format(**source_args, video_key=key))
            dst = relative_path(info["video_path"].format(**args, video_key=key))
            _link(source / src, target / dst)
    meta = target / "meta"
    meta.mkdir()
    shutil.copy2(source / "meta/tasks.jsonl", meta / "tasks.jsonl")
    for filename in ("episodes.jsonl", "episodes_stats.jsonl"):
        rows = (source / "meta" / filename).read_text().splitlines()
        if len(rows) != 1:
            raise ValueError(f"Expected single-episode {filename}")
        row = json.loads(rows[0])
        row["episode_index"] = sequence
        if "stats" in row:
            for key, delta in (("episode_index", sequence), ("index", offset)):
                for stat in ("min", "max", "mean"):
                    row["stats"][key][stat] = (np.asarray(row["stats"][key][stat]) + delta).tolist()
        prefix = (previous / "meta" / filename).read_text() if previous else ""
        (meta / filename).write_text(prefix + json.dumps(row, allow_nan=False) + "\n")
    info.update(
        total_episodes=sequence + 1,
        total_frames=offset + info["total_frames"],
        total_videos=(old["total_videos"] if old else 0) + info["total_videos"],
        total_chunks=args["episode_chunk"] + 1,
        splits={"train": f"0:{sequence + 1}"},
    )
    atomic_json(meta / "info.json", info)


def materialize_local(root, bundle, manifest):
    """Apply a committed delta locally. Re-applying after interruption is safe."""
    for role, repo in manifest["repo_ids"].items():
        target = root / "datasets" / repo
        for file in sorted((bundle / role).rglob("*")):
            if not file.is_file():
                continue
            dest = target / file.relative_to(bundle / role)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + ".tmp")
            temp.unlink(missing_ok=True)
            if "meta" in file.relative_to(bundle / role).parts:
                shutil.copy2(file, temp)
            else:
                _link(file, temp)
            temp.replace(dest)


def _recover_local(root, bundles):
    marker = root / ".materialized.json"
    last = read_json(marker)["sequence"] if marker.exists() else -1
    for bundle in bundles[last + 1 :]:
        manifest = read_json(bundle / "manifest.json")
        materialize_local(root, bundle, manifest)
        atomic_json(marker, {"sequence": manifest["sequence"]})


def publish_episode(source, config, source_manifest):
    """Both roles become visible in one rename; ready metadata is never mutated."""
    validate_publication(config["publication"])
    root = Path(config["output_dir"])
    ready = root / "ready"
    ready.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / ".publish.lock"):
        bundles = sorted(ready.glob("episode_*"))
        _recover_local(root, bundles)
        previous_manifest = None
        for index, bundle in enumerate(bundles):
            item = read_json(bundle / "manifest.json")
            if item["sequence"] != index:
                raise ValueError("Missing ready episode; retain committed ready packages")
            if item["source_episode_id"] == source_manifest["source_episode_id"]:
                return str(bundle)
            previous_manifest = item
        repo_ids = config["publication"]["repo_ids"]
        if previous_manifest and previous_manifest["repo_ids"] != repo_ids:
            raise ValueError("Repository IDs changed; use a new collection batch")
        start_index = config.get("start_index", 0)
        if previous_manifest and previous_manifest.get("start_index", 0) != start_index:
            raise ValueError("start_index changed; keep it fixed when resuming a batch")
        sequence = len(bundles)
        target = ready / f"episode_{start_index + sequence:06d}"
        building = root / ".building_collect"
        if building.exists():
            shutil.rmtree(building)
        building.mkdir()
        for role in ROLES:
            _append_role(
                Path(source) / role,
                building / role,
                bundles[-1] / role if bundles else None,
                sequence,
            )
        manifest = {
            "schema_version": SCHEMA,
            "sequence": sequence,
            "start_index": start_index,
            "source_episode_id": source_manifest["source_episode_id"],
            "source_episode_name": Path(source).name.removesuffix("_temp"),
            # Preserve capture diagnostics before removing the temporary source.
            "recording": copy.deepcopy(source_manifest),
            "batch_uid": (
                previous_manifest["batch_uid"] if previous_manifest else source_manifest["source_episode_id"]
            ),
            "repo_ids": copy.deepcopy(repo_ids),
            "files": {
                str(p.relative_to(building)): p.stat().st_size for p in sorted(building.rglob("*")) if p.is_file()
            },
        }
        atomic_json(building / "manifest.json", manifest)
        building.rename(target)
        _recover_local(root, [*bundles, target])
        return str(target)
