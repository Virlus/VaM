"""Register completed LeRobot action/value uploads using the existing dataset-view publisher.

Uploads append to one repo per policy and role. Action copies preserve GAE computed before camera
downsampling and receive policy-wide p70 labels before publication. Original uploads stay immutable.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from preprocess_data.utils.dataset_processor import _stats_entry_float_singleton
from preprocess_data.utils.dataset_registry import _atomic_write_json
from preprocess_data.utils.dataset_registry import _read_ready_checkpoint_policy_id
from preprocess_data.utils.dataset_registry import atomic_write_parquet
from preprocess_data.utils.dataset_registry import publish_dataset_view
from preprocess_data.utils.upload_advantage import ADVANTAGE_ALGORITHM
from preprocess_data.utils.upload_advantage import raw_advantage_column
from preprocess_data.utils.upload_advantage import read_advantage_recipe
from preprocess_data.utils.upload_advantage import uploaded_camera_advantage
from preprocess_data.utils.upload_append import append_uploaded_episodes

logger = logging.getLogger(__name__)


def read_json(path: Path):
    return json.loads(path.read_text())


def read_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def child(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset: {relative}")
    return path


def offline_repo(root: Path, repo_id: str) -> Path:
    """Allow launcher-managed offline symlinks; this path is only read, never rewritten."""
    relative = Path(repo_id)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid offline repo id: {repo_id}")
    return root / relative


def validate_repo(
    root: Path, role: str, policy: dict, *, final: bool = False, episode_ids: list[int] | None = None
) -> list[int]:
    """Check the training-facing essentials before exposing either role to a view."""
    info = read_json(root / "meta/info.json")
    if info["codebase_version"] != "v2.1":
        raise ValueError(f"{root}: expected LeRobot v2.1")
    episodes = read_rows(root / "meta/episodes.jsonl")
    ids = [row["episode_index"] for row in episodes]
    if not ids or ids != list(range(info["total_episodes"])):
        raise ValueError(f"{root}: episode indices must be nonempty and contiguous")
    mapping = root / "meta/policy_metadata_map.jsonl"
    if mapping.exists():
        rows = read_rows(mapping)
        if [row["target_episode_index"] for row in rows] != ids or any(
            row.get("policy_id") != policy["policy_id"] or row.get("checkpoint_index") != policy["checkpoint_index"]
            for row in rows
            if episode_ids is None or row["target_episode_index"] in episode_ids
        ):
            raise ValueError(f"{root}: inconsistent policy metadata mapping")
    stats = read_rows(root / "meta/episodes_stats.jsonl")
    if [row["episode_index"] for row in stats] != ids:
        raise ValueError(f"{root}: missing episode statistics")
    tasks = {row["task_index"] for row in read_rows(root / "meta/tasks.jsonl")}
    required = {"state", "actions", "value_prediction", "policy_id", "checkpoint_index", "data_source", "control_flag"}
    if role == "action":
        required.add("advantage_raw")
        if final:
            required.add("advantage")
    else:
        required.update({"next_state", "next_actions", "next_value_prediction", "reward", "terminate"})
    if not required <= info["features"].keys():
        raise ValueError(f"{root}: missing features {sorted(required - info['features'].keys())}")
    videos = {key for key, spec in info["features"].items() if spec["dtype"] == "video"}
    expected_videos = {"base_image", "left_wrist_image", "right_wrist_image"}
    if role == "value":
        expected_videos |= {f"next_{name}" for name in expected_videos}
    if not expected_videos <= videos:
        raise ValueError(f"{root}: missing training camera features")
    frames = 0
    for episode in episodes:
        index = episode["episode_index"]
        if episode_ids is not None and index not in episode_ids:
            frames += episode["length"]
            continue
        values = {"episode_index": index, "episode_chunk": index // info["chunks_size"]}
        path = child(root, info["data_path"].format(**values))
        columns = sorted(required | {"episode_index", "frame_index", "index", "task_index", "timestamp"})
        table = pq.read_table(path, columns=columns)
        count = episode["length"]
        if count <= 0 or table.num_rows != count:
            raise ValueError(f"{path}: incorrect frame count")
        for key, expected in (
            ("episode_index", index),
            ("policy_id", policy["policy_id"]),
            ("checkpoint_index", policy["checkpoint_index"]),
            ("data_source", 1),
        ):
            if not np.all(np.asarray(table[key].to_pylist()).reshape(-1) == expected):
                raise ValueError(f"{path}: inconsistent {key}")
        for key, expected in (("frame_index", np.arange(count)), ("index", np.arange(frames, frames + count))):
            if not np.array_equal(np.asarray(table[key].to_pylist()).reshape(-1), expected):
                raise ValueError(f"{path}: inconsistent {key}")
        if not set(np.asarray(table["task_index"].to_pylist()).reshape(-1)) <= tasks:
            raise ValueError(f"{path}: unknown task index")
        for key in required | {"timestamp"}:
            if not np.isfinite(np.asarray(table[key].to_pylist(), dtype=float)).all():
                raise ValueError(f"{path}: non-finite {key}")
        flags = np.asarray(table["control_flag"].to_pylist()).reshape(-1)
        if not np.isin(flags, [0, 1]).all():
            raise ValueError(f"{path}: control_flag must be 0 or 1")
        if role == "action" and np.asarray(table["advantage_raw"].to_pylist()).shape not in {(count,), (count, 1)}:
            raise ValueError(f"{path}: advantage_raw must contain one scalar per frame")
        if role == "action" and final:
            labels = np.asarray(table["advantage"].to_pylist()).reshape(-1)
            if not np.isin(labels, [0, 1]).all() or not np.all(labels[flags == 1] == 1):
                raise ValueError(f"{path}: invalid advantage labels")
        for key in videos:
            video = child(root, info["video_path"].format(**values, video_key=key))
            if not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f"Missing video: {video}")
        frames += count
    if frames != info["total_frames"]:
        raise ValueError(f"{root}: total_frames mismatch")
    return ids


def register(args, state: dict, bundle: Path) -> bool:
    source_id = bundle.name
    # Stable names preserve idempotence after a crash between copying and registration.
    if source_id in state["uploads"]:
        return False
    policy = read_json(bundle / "episode_policy.json")
    for key in ("policy_id", "checkpoint_index"):
        value = policy[key]
        if type(value) is not int or not 0 <= value <= 2**24:
            raise ValueError(f"{key} must be a nonnegative exact float32 integer")
    pid = policy["policy_id"]
    ready = _read_ready_checkpoint_policy_id(args.cloud_root / "checkpoints/latest.json")
    admitted = state.get("pending_upload") == source_id
    if not admitted and ready is not None and pid < ready:
        logger.warning("Skipping stale upload %s (policy %d < %d)", bundle, pid, ready)
        return False
    if not admitted and pid > (ready if ready is not None else 0):
        logger.info("Waiting for policy %d checkpoint: %s", pid, bundle)
        return False
    if bundle.is_symlink() or any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError(f"Uploaded symlinks are not supported: {bundle}")
    recipe = read_advantage_recipe(bundle)
    for record in state["uploads"].values():
        if record["policy_id"] == pid and record["advantage_recipe"] != recipe:
            raise ValueError(f"Upload {source_id} changes advantage parameters within policy {pid}")
    for role in ("action", "value"):
        validate_repo(bundle / role, role, policy)
    state["pending_upload"] = source_id
    _atomic_write_json(args.cloud_root / "conversion_state/lerobot_uploads.json", state)
    repos = {}
    for role in ("action", "value"):
        repo = f"{args.repo_prefix}_policy_{pid:06d}_deployment_{role}"
        target = child(args.dataset_home, repo)
        try:
            episode_ids = append_uploaded_episodes(bundle / role, target, source_id=source_id, policy=policy)
        except Exception as exc:
            raise RuntimeError(f"Append interrupted for {source_id}; restart to retry before other uploads") from exc
        repos[role] = {"repo_id": repo, "episode_ids": episode_ids}
    state["uploads"][source_id] = {**policy, "repos": repos, "status": "copied", "advantage_recipe": recipe}
    state.pop("pending_upload", None)
    pending = state.setdefault("pending_policies", [])
    if pid not in pending:
        pending.append(pid)
    state["published"] = False
    return True


def postprocess(args, state: dict, state_path: Path) -> None:
    """Recover each dirty policy as a whole, including previously published episodes."""
    for pid in list(state.get("pending_policies", [])):
        records = {key: record for key, record in state["uploads"].items() if record["policy_id"] == pid}
        recipe = next(iter(records.values()))["advantage_recipe"]
        if any(record["advantage_recipe"] != recipe for record in records.values()):
            raise ValueError(f"Policy {pid} has inconsistent advantage parameters")
        inputs = []
        for source_id, record in records.items():
            source_root = child(args.inbox, source_id) / "action"
            source_info = read_json(source_root / "meta/info.json")
            source_ids = [row["episode_index"] for row in read_rows(source_root / "meta/episodes.jsonl")]
            root = child(args.dataset_home, record["repos"]["action"]["repo_id"])
            info = read_json(root / "meta/info.json")
            for source_ep, ep in zip(source_ids, record["repos"]["action"]["episode_ids"], strict=True):
                source_path = child(
                    source_root,
                    source_info["data_path"].format(
                        episode_index=source_ep, episode_chunk=source_ep // source_info["chunks_size"]
                    ),
                )
                path = child(root, info["data_path"].format(episode_index=ep, episode_chunk=ep // info["chunks_size"]))
                columns = ["advantage_raw", "control_flag", "data_source", "value_prediction"]
                table = pq.read_table(path, columns=columns)
                raw = uploaded_camera_advantage(pq.read_table(source_path, columns=columns), table)
                # Use the persisted dtype for both p70 and labels, including old float32 uploads.
                raw = np.asarray(raw_advantage_column(table, raw).to_pylist(), dtype=float).reshape(-1)
                flags = np.asarray(table["control_flag"].to_pylist()).reshape(-1)
                if len(raw) != table.num_rows or not np.isfinite(raw).all():
                    raise ValueError(f"{path}: invalid advantage_raw")
                if not np.isin(flags, [0, 1]).all() or not np.all(np.asarray(table["data_source"].to_pylist()) == 1):
                    raise ValueError(f"{path}: invalid online/control flags")
                inputs.append((root, ep, path, raw, flags))
        values = np.concatenate([entry[3] for entry in inputs])
        threshold = float(np.percentile(values, 70, method="linear"))
        revision = state.get("annotation_revision", 0) + 1
        audit = {
            "policy_id": pid,
            "percentile": 70,
            "threshold": threshold,
            "upload_ids": sorted(records),
            "frame_count": len(values),
            "revision": revision,
            "algorithm": ADVANTAGE_ALGORITHM,
            "recipe": recipe,
        }
        state.setdefault("annotations", {})[str(pid)] = {**audit, "status": "pending"}
        _atomic_write_json(state_path, state)  # intent is durable before touching published copies
        stats_by_root = {}
        labels_by_root = {}
        raw_by_root = {}
        raw_stats_by_root = {}
        snapshots = {}
        for root, ep, path, raw, flags in inputs:
            labels = np.where(flags == 1, 1, raw >= threshold).astype(np.float32)
            table = pq.read_table(path)
            table = table.set_column(
                table.column_names.index("advantage_raw"), "advantage_raw", raw_advantage_column(table, raw)
            )
            column = pa.array(labels[:, None].tolist(), type=pa.list_(pa.float32()))
            if "advantage" in table.column_names:
                table = table.set_column(table.column_names.index("advantage"), "advantage", column)
            else:
                table = table.append_column("advantage", column)
            metadata = dict(table.schema.metadata or {})
            hf = json.loads(metadata.get(b"huggingface", b'{"info":{"features":{}}}'))
            hf["info"]["features"]["advantage"] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": 1,
                "_type": "Sequence",
            }
            metadata[b"huggingface"] = json.dumps(hf).encode()
            atomic_write_parquet(table.replace_schema_metadata(metadata), path)
            stats_by_root.setdefault(root, {})[ep] = _stats_entry_float_singleton(labels)
            labels_by_root.setdefault(root, []).append(labels)
            raw_stats_by_root.setdefault(root, {})[ep] = _stats_entry_float_singleton(raw)
            raw_by_root.setdefault(root, []).append(raw)
            snapshots.setdefault(root, {})[str(ep)] = labels
        for root, stats in stats_by_root.items():
            info = read_json(root / "meta/info.json")
            info["features"]["advantage"] = {"dtype": "float32", "shape": [1], "names": None}
            _atomic_write_json(root / "meta/info.json", info)
            stats_path = root / "meta/episodes_stats.jsonl"
            rows = read_rows(stats_path)
            for row in rows:
                row["stats"]["advantage"] = stats[row["episode_index"]]
                row["stats"]["advantage_raw"] = raw_stats_by_root[root][row["episode_index"]]
            temporary = stats_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(row) + "\n" for row in rows))
            temporary.replace(stats_path)
            aggregate_path = root / "meta/stats.json"
            if aggregate_path.exists():
                aggregate = read_json(aggregate_path)
                aggregate["advantage"] = _stats_entry_float_singleton(np.concatenate(labels_by_root[root]))
                aggregate["advantage_raw"] = _stats_entry_float_singleton(np.concatenate(raw_by_root[root]))
                _atomic_write_json(aggregate_path, aggregate)
            snapshot = f"meta/advantage_revision_{revision:08d}.npz"
            snapshot_path = child(root, snapshot)
            temporary = snapshot_path.with_suffix(".tmp")
            with temporary.open("wb") as fh:
                np.savez_compressed(fh, **snapshots[root])
            temporary.replace(snapshot_path)
            _atomic_write_json(root / "meta/advantage_annotation.json", {**audit, "snapshot": snapshot})
            for record in records.values():
                if record["repos"]["action"]["repo_id"] == str(root.relative_to(args.dataset_home)):
                    record["repos"]["action"]["advantage_snapshot"] = snapshot
        for record in records.values():
            for role in ("action", "value"):
                validate_repo(
                    child(args.dataset_home, record["repos"][role]["repo_id"]),
                    role,
                    record,
                    final=True,
                    episode_ids=record["repos"][role]["episode_ids"],
                )
            record["status"] = "postprocessed"
        state["annotations"][str(pid)]["status"] = "complete"
        state["annotation_revision"] = revision
        state["pending_policies"].remove(pid)
        state["published"] = False
        _atomic_write_json(state_path, state)


def publish(args, state: dict) -> None:
    """Reuse the recorded data path's v2 views, defaults and two existing latest pointers."""
    if not state["uploads"] or state["published"]:
        return
    if state.get("pending_policies") or any(r.get("status") == "copied" for r in state["uploads"].values()):
        raise ValueError("Cannot publish before policy annotation completes")
    target = max(record["policy_id"] for record in state["uploads"].values())
    timestamp = time.time_ns()
    upload_ids = sorted(state["uploads"])
    for role in ("action", "value"):
        if role == "value" and state.get("value_upload_ids") == upload_ids:
            continue
        defaults = {"policy_id": -1.0, "checkpoint_index": -1.0, "control_flag": -1.0}
        if role == "action":
            defaults["advantage"] = 1.0
        offline = getattr(args, f"{role}_offline_repo")
        specs = {
            offline: {
                "filters": {"data_source": 0},
                "missing_feature_defaults": defaults,
                "repo_path": offline_repo(args.dataset_home, offline),
            }
        }
        for record in state["uploads"].values():
            repo = record["repos"][role]
            spec = specs.setdefault(repo["repo_id"], {"filters": {}, "episode_ids": []})
            spec["episode_ids"].extend(repo["episode_ids"])
            spec["episode_ids"].sort()
            if role == "action" and repo.get("advantage_snapshot"):
                spec["advantage_snapshot"] = repo["advantage_snapshot"]
        publish_dataset_view(
            output_dir=args.cloud_root / "dataset_views",
            latest_name=f"{role}_latest.json",
            view_id=f"live_{role}_policy_{target:06d}_{timestamp}",
            repo_specs=specs,
            target_policy_id=target,
            include_episode_ids=True,
            annotation_revision=state.get("annotation_revision") if role == "action" else None,
        )
    state["value_upload_ids"] = upload_ids
    state["published"] = True
    for record in state["uploads"].values():
        record["status"] = "published"


def migrate_policy_repos(args, state: dict, state_path: Path) -> None:
    """Restart migration: retain legacy upload repos and build the canonical policy repos."""
    legacy = [
        (key, record)
        for key, record in state["uploads"].items()
        if any(
            record["repos"][role]["repo_id"] != f"{args.repo_prefix}_policy_{record['policy_id']:06d}_deployment_{role}"
            for role in ("action", "value")
        )
    ]
    if not legacy:
        return
    backup = state_path.with_suffix(".before_policy_repos.json")
    if not backup.exists():
        _atomic_write_json(backup, state)
    for source_id, record in legacy:
        repos = {}
        for role in ("action", "value"):
            repo = f"{args.repo_prefix}_policy_{record['policy_id']:06d}_deployment_{role}"
            source = child(args.dataset_home, record["repos"][role]["repo_id"])
            policy = {key: record[key] for key in ("policy_id", "checkpoint_index")}
            ids = append_uploaded_episodes(source, child(args.dataset_home, repo), source_id=source_id, policy=policy)
            repos[role] = {"repo_id": repo, "episode_ids": ids}
        record["legacy_repos"] = record["repos"]
        record["repos"] = repos
        record["status"] = "copied"
        pending = state.setdefault("pending_policies", [])
        if record["policy_id"] not in pending:
            pending.append(record["policy_id"])
        state["published"] = False
        state.pop("value_upload_ids", None)
        _atomic_write_json(state_path, state)


def migrate_advantage_algorithm(args, state: dict, state_path: Path) -> None:
    """Reannotate accepted policies once, using persisted or explicitly recorded recipes."""
    recipes = {}
    by_policy = {}
    pending = set(state.get("pending_policies", []))
    for source_id, record in state["uploads"].items():
        recipe = read_advantage_recipe(child(args.inbox, source_id))
        pid = record["policy_id"]
        if pid in by_policy and by_policy[pid] != recipe:
            raise ValueError(f"Policy {pid} has inconsistent advantage parameters")
        by_policy[pid] = recipe
        recipes[source_id] = recipe
        audit = state.get("annotations", {}).get(str(pid), {})
        if audit.get("algorithm") != ADVANTAGE_ALGORITHM or audit.get("recipe") != recipe:
            pending.add(pid)
    if not recipes:
        return
    changed = any(record.get("advantage_recipe") != recipes[key] for key, record in state["uploads"].items())
    changed |= pending != set(state.get("pending_policies", []))
    if not changed:
        return
    backup = state_path.with_suffix(".before_camera_advantage.json")
    if not backup.exists():
        _atomic_write_json(backup, state)
    for source_id, recipe in recipes.items():
        state["uploads"][source_id]["advantage_recipe"] = recipe
    state["pending_policies"] = sorted(pending)
    state["published"] = False
    _atomic_write_json(state_path, state)


def run_once(args, state: dict, state_path: Path) -> None:
    # Upgrade the earlier copy-only registry; never silently retain recorder labels.
    for source_id, record in state["uploads"].items():
        if "status" not in record:
            record["checkpoint_index"] = read_json(args.inbox / source_id / "episode_policy.json")["checkpoint_index"]
            record["status"] = "copied"
            pending = state.setdefault("pending_policies", [])
            if record["policy_id"] not in pending:
                pending.append(record["policy_id"])
            state["published"] = False
            _atomic_write_json(state_path, state)
    migrate_policy_repos(args, state, state_path)
    migrate_advantage_algorithm(args, state, state_path)
    # Recover annotation before publication, regardless of checkpoint advancement.
    postprocess(args, state, state_path)
    publish(args, state)
    _atomic_write_json(state_path, state)
    bundles = sorted(args.inbox.iterdir())
    if state.get("pending_upload"):
        pending = child(args.inbox, state["pending_upload"])
        if not (pending / "complete.txt").is_file():
            raise ValueError(f"Admitted upload is missing: {pending}")
        bundles = [pending, *(p for p in bundles if p != pending)]
    for bundle in bundles:
        if not bundle.is_dir() or not (bundle / "complete.txt").is_file():
            continue
        try:
            changed = register(args, state, bundle)
        except (ValueError, KeyError, OSError) as exc:
            if state.get("pending_upload") == bundle.name:
                raise RuntimeError(f"Cannot recover admitted upload {bundle.name}") from exc
            logger.warning("Upload not registered %s: %s", bundle, exc)
            continue
        if changed:
            _atomic_write_json(state_path, state)
            postprocess(args, state, state_path)
            publish(args, state)
            _atomic_write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inbox", "dataset-home", "cloud-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("repo-prefix", "action-offline-repo", "value-offline-repo"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--poll-interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    for name in ("inbox", "dataset_home", "cloud_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    child(args.dataset_home, args.repo_prefix)
    for role in ("action", "value"):
        offline = offline_repo(args.dataset_home, getattr(args, f"{role}_offline_repo"))
        if not (offline / "meta/info.json").is_file():
            parser.error(f"Offline {role} repo is not visible: {offline}")
    args.inbox.mkdir(parents=True, exist_ok=True)
    state_path = args.cloud_root / "conversion_state/lerobot_uploads.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        key: str(getattr(args, key))
        for key in (
            "inbox",
            "dataset_home",
            "repo_prefix",
            "action_offline_repo",
            "value_offline_repo",
        )
    }
    logging.basicConfig(level=logging.INFO)
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = (
            read_json(state_path) if state_path.exists() else {"identity": identity, "uploads": {}, "published": True}
        )
        if state["identity"] != identity:
            raise ValueError("Registration paths/repos changed; use the original settings or a new run")
        if not state_path.exists() and any(
            (args.cloud_root / "dataset_views" / f"{r}_latest.json").exists() for r in ("action", "value")
        ):
            raise ValueError("Existing views have no upload registry; use a new run instead of overwriting them")
        while True:
            run_once(args, state, state_path)
            if args.once:
                break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
