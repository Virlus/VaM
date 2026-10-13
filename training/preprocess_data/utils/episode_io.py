"""LeRobot v2.1 episode and metadata I/O shared by maintenance and registration."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    entries = []
    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    tmp_path.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    with path.open("r") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object JSON in {path}, got {type(loaded).__name__}")
    return loaded


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    tmp_path.replace(path)


def _replace_table_column(table: pa.Table, name: str, values: list[int] | np.ndarray) -> pa.Table:
    if name not in table.column_names:
        return table
    idx = table.column_names.index(name)
    return table.set_column(idx, name, pa.array(values, type=pa.int64()))


def _remap_episode_stats(
    stats_entry: dict[str, Any], episode_index: int, frame_offset: int, task_index_map: dict[int, int]
) -> None:
    stats_entry["episode_index"] = episode_index
    stats = stats_entry.get("stats", {})
    if not isinstance(stats, dict):
        return

    if isinstance(stats.get("episode_index"), dict):
        count = stats["episode_index"].get("count", [1])
        stats["episode_index"] = {
            "min": [episode_index],
            "max": [episode_index],
            "mean": [float(episode_index)],
            "std": [0.0],
            "count": count,
        }
    if isinstance(stats.get("index"), dict):
        for key in ("min", "max", "mean"):
            if isinstance(stats["index"].get(key), list):
                stats["index"][key] = [value + frame_offset for value in stats["index"][key]]
    if isinstance(stats.get("task_index"), dict):
        old_min = int(stats["task_index"].get("min", [0])[0])
        old_max = int(stats["task_index"].get("max", [old_min])[0])
        if old_min == old_max:
            new_task_index = task_index_map[old_min]
            count = stats["task_index"].get("count", [1])
            stats["task_index"] = {
                "min": [new_task_index],
                "max": [new_task_index],
                "mean": [float(new_task_index)],
                "std": [0.0],
                "count": count,
            }


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = _read_jsonl(path)
    for expected_idx, task in enumerate(tasks):
        task_idx = int(task.get("task_index", expected_idx))
        if task_idx != expected_idx:
            raise ValueError(
                f"{path} must contain contiguous task_index values from 0; got {task_idx} at {expected_idx}"
            )
    return tasks


def _copy_remapped_episode(
    *,
    source_root: Path,
    target_root: Path,
    source_episode_index: int,
    target_episode_index: int,
    frame_offset: int,
    source_chunks_size: int,
    target_chunks_size: int,
    task_index_map: dict[int, int],
    video_keys: list[str],
) -> int:
    source_parquet = (
        source_root
        / "data"
        / f"chunk-{source_episode_index // source_chunks_size:03d}"
        / f"episode_{source_episode_index:06d}.parquet"
    )
    if not source_parquet.exists():
        raise FileNotFoundError(f"Missing source parquet: {source_parquet}")

    table = pq.read_table(source_parquet)
    frame_count = table.num_rows
    table = _replace_table_column(table, "episode_index", np.full(frame_count, target_episode_index, dtype=np.int64))
    table = _replace_table_column(
        table, "index", np.asarray(table.column("index").to_pylist(), dtype=np.int64) + frame_offset
    )
    if "task_index" in table.column_names:
        old_task_indices = [int(value) for value in table.column("task_index").to_pylist()]
        table = _replace_table_column(table, "task_index", [task_index_map[value] for value in old_task_indices])

    target_parquet = (
        target_root
        / "data"
        / f"chunk-{target_episode_index // target_chunks_size:03d}"
        / f"episode_{target_episode_index:06d}.parquet"
    )
    target_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target_parquet)

    source_chunk = f"chunk-{source_episode_index // source_chunks_size:03d}"
    target_chunk = f"chunk-{target_episode_index // target_chunks_size:03d}"
    for video_key in video_keys:
        source_video = source_root / "videos" / source_chunk / video_key / f"episode_{source_episode_index:06d}.mp4"
        if not source_video.exists():
            raise FileNotFoundError(f"Missing source video: {source_video}")
        target_video = target_root / "videos" / target_chunk / video_key / f"episode_{target_episode_index:06d}.mp4"
        target_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_video, target_video)

    return frame_count
