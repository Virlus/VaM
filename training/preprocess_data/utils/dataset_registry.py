"""Atomic view publication and checkpoint readiness shared by both input formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def atomic_write_parquet(table: pa.Table, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        pq.write_table(table, tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _read_ready_checkpoint_policy_id(latest_path: Path) -> int | None:
    if not latest_path.exists():
        return None
    try:
        latest = json.loads(latest_path.read_text())
    except json.JSONDecodeError:
        return None
    checkpoint_path = latest.get("checkpoint_path")
    if checkpoint_path and not (Path(checkpoint_path).expanduser() / "DONE").exists():
        return None
    policy_id = int(latest.get("policy_id", -1))
    return policy_id if policy_id >= 0 else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _episode_ids_for_repo_path(repo_id_or_path: str | Path) -> list[int]:
    path = Path(repo_id_or_path)
    if path.exists():
        meta_dir = path / "meta"
        episodes_path = meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            with episodes_path.open() as f:
                episode_ids = [int(json.loads(line)["episode_index"]) for line in f if line.strip()]
            if episode_ids:
                return episode_ids
        parquet_ids = [
            int(parquet_path.stem.removeprefix("episode_"))
            for parquet_path in sorted((path / "data").glob("chunk-*/episode_*.parquet"))
        ]
        if parquet_ids:
            return parquet_ids
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        meta = LeRobotDatasetMetadata(str(repo_id_or_path))
        episodes = getattr(meta, "episodes", None)
        if episodes is not None:
            return [int(ep["episode_index"]) for ep in episodes]
        total = int(getattr(meta, "total_episodes", 0) or meta.info.get("total_episodes", 0))
        return list(range(total))
    except Exception:
        meta_dir = path / "meta"
        episodes_path = meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            with episodes_path.open() as f:
                return [int(json.loads(line)["episode_index"]) for line in f if line.strip()]
        return [
            int(parquet_path.stem.removeprefix("episode_"))
            for parquet_path in sorted((path / "data").glob("chunk-*/episode_*.parquet"))
        ]


def publish_dataset_view(
    *,
    output_dir: Path,
    latest_name: str,
    view_id: str,
    repo_specs: dict[str, dict[str, Any]],
    target_policy_id: int | None,
    include_episode_ids: bool = False,
    annotation_revision: int | None = None,
) -> Path:
    """Publish a dataset view JSON with repo_id -> filters mapping.

    Args:
        output_dir: Directory to write the dataset view JSON files
        latest_name: Name for the 'latest' pointer file (e.g., "latest.json")
        view_id: Unique identifier for this dataset view
        repo_specs: Dict mapping repo_id to a spec with:
            - "filters": Dict of column filters (e.g., {"control_flag": 1, "data_source": 0})
            - "repo_path": Optional local path for episode_ids resolution
            - "episode_ids": Optional explicit episode list
            - "missing_feature_defaults": Optional default values for columns absent from this repo
        target_policy_id: The policy_id this view is associated with
        include_episode_ids: Whether to include episode_ids in the output
        annotation_revision: Optional label revision; marked views require rereading existing rows.

    Returns:
        Path to the written dataset view JSON file
    """
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    repos: dict[str, Any] = {}

    for repo_id, spec in repo_specs.items():
        if not isinstance(spec, dict):
            raise ValueError(f"repo_specs[{repo_id!r}] must be a dict with 'filters' key")

        repo_entry: dict[str, Any] = {
            "filters": spec.get("filters", {}),
        }
        if spec.get("advantage_snapshot") is not None:
            repo_entry["advantage_snapshot"] = spec["advantage_snapshot"]
        missing_feature_defaults = spec.get("missing_feature_defaults")
        if missing_feature_defaults:
            if not isinstance(missing_feature_defaults, dict):
                raise ValueError(f"repo_specs[{repo_id!r}].missing_feature_defaults must be a dict")
            repo_entry["missing_feature_defaults"] = dict(missing_feature_defaults)

        if include_episode_ids:
            if "episode_ids" in spec and spec["episode_ids"] is not None:
                episode_ids = spec["episode_ids"]
            elif "repo_path" in spec and spec["repo_path"] is not None:
                episode_ids = _episode_ids_for_repo_path(spec["repo_path"])
            else:
                episode_ids = _episode_ids_for_repo_path(repo_id)
            if episode_ids:
                repo_entry["episode_ids"] = episode_ids

        repos[repo_id] = repo_entry

    view = {
        "schema_version": "dataset_view/v2",
        "dataset_view_id": view_id,
        "target_policy_id": target_policy_id,
        "created_at": created_at,
        "repos": repos,
    }
    if annotation_revision is not None:
        view["annotation_revision"] = annotation_revision
    view_path = output_dir / f"{view_id}.json"
    _atomic_write_json(view_path, view)
    _atomic_write_json(
        output_dir / latest_name,
        {
            "schema_version": "dataset_view_latest/v1",
            "dataset_view_id": view_id,
            "view_path": str(view_path),
            "created_at": created_at,
        },
    )
    logger.info(f"Published dataset view {view_id} to {view_path}")
    return view_path
