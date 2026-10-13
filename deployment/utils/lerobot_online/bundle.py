"""Paired LeRobot output, immutable manifests and local atomic publication.

Frame building remains a separate prerequisite. This module never fills missing
predictions/rewards/labels, nor deletes raw recordings after publication.
"""

from __future__ import annotations
import json
from pathlib import Path
from .protocol import ProtocolError, json_bytes, read_json, relative_path, sha256_file


def inventory(root):
    root = Path(root)
    if root.is_symlink():
        raise ProtocolError("Bundle root cannot be a symlink")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ProtocolError(f"Bundle contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ProtocolError(f"Bundle contains non-regular file: {path}")
        name = relative_path(path.relative_to(root).as_posix())
        if name in {"manifest.json", "complete.txt"}:
            continue
        files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return files


def validate_repo(root, lengths, role, contract, *, raw_advantage=False):
    """Read every parquet and decode every video, including the terminal frame."""
    import av
    import numpy as np
    import pyarrow.parquet as pq

    root = Path(root)
    info = read_json(root / "meta" / "info.json")
    if (
        info["codebase_version"] != "v2.1"
        or info["total_episodes"] != len(lengths)
        or info["total_frames"] != sum(lengths)
    ):
        raise ProtocolError("LeRobot metadata count/version mismatch")
    metadata = {}
    for name in ("episodes", "tasks", "episodes_stats"):
        rows = [json.loads(line) for line in (root / "meta" / f"{name}.jsonl").read_text().splitlines() if line.strip()]
        json_bytes(rows)
        metadata[name] = rows
    for name in ("episodes", "episodes_stats"):
        if [row["episode_index"] for row in metadata[name]] != list(range(len(lengths))):
            raise ProtocolError(f"LeRobot {name} indices mismatch")
    if [row["length"] for row in metadata["episodes"]] != lengths:
        raise ProtocolError("Episode metadata lengths mismatch")
    tasks = metadata["tasks"]
    if (
        len(tasks) != info["total_tasks"]
        or not tasks
        or [row["task_index"] for row in tasks] != list(range(len(tasks)))
    ):
        raise ProtocolError("Task metadata mismatch")
    video_keys = [key for key, spec in info["features"].items() if spec["dtype"] == "video"]
    if len(video_keys) != 6:
        raise ProtocolError("Expected six current/next video features")
    if len(list((root / "data").rglob("*.parquet"))) != len(lengths):
        raise ProtocolError("Unexpected parquet count")
    if len(list((root / "videos").rglob("*.mp4"))) != len(lengths) * len(video_keys):
        raise ProtocolError("Unexpected video count")
    for index, count in enumerate(lengths):
        format_args = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
        table = pq.read_table(root / relative_path(info["data_path"].format(**format_args)))
        if len(table) != count:
            raise ProtocolError("Parquet row count mismatch")
        for key, expected in (
            ("episode_index", np.full(count, index)),
            ("frame_index", np.arange(count)),
            ("index", np.arange(sum(lengths[:index]), sum(lengths[: index + 1]))),
        ):
            if not np.array_equal(np.asarray(table[key].to_pylist()).reshape(-1), expected):
                raise ProtocolError(f"Parquet {key} mismatch")
        if not np.isin(np.asarray(table["task_index"].to_pylist()), range(len(tasks))).all():
            raise ProtocolError("Parquet task index mismatch")
        if not np.allclose(
            np.asarray(table["timestamp"].to_pylist()).reshape(-1), np.arange(count) / info["fps"], atol=1e-06
        ):
            raise ProtocolError("Parquet logical timestamps mismatch")
        required = {
            "value_prediction",
            "next_value_prediction",
            "source_action_index",
            "control_flag",
            "policy_id",
            "checkpoint_index",
        }
        if role == "action":
            required |= {"advantage_raw"} if raw_advantage else {"advantage_raw", "advantage"}
        if not required <= set(table.column_names):
            raise ProtocolError("Required online fields are missing")
        for key, spec in info["features"].items():
            if spec["dtype"] not in {"float32", "float64", "int64"}:
                continue
            values = np.asarray(table[key].to_pylist())
            if not np.isfinite(values).all():
                raise ProtocolError(f"Non-finite feature: {key}")
            if spec["shape"] == [1] and values.shape == (count,):
                values = values.reshape(count, 1)
            if key not in {"timestamp", "frame_index", "episode_index", "index", "task_index"} and values.shape != (
                count,
                *spec["shape"],
            ):
                raise ProtocolError(f"Feature shape mismatch: {key}")
        flags = np.asarray(table["control_flag"].to_pylist()).reshape(-1)
        if not np.isin(flags, [0, 1]).all():
            raise ProtocolError("Excluded control flags in output")
        for key in ("policy_id", "checkpoint_index"):
            if not np.equal(np.asarray(table[key].to_pylist()), contract[key]).all():
                raise ProtocolError("Parquet policy identity mismatch")
        if role == "action" and (not raw_advantage):
            raw = np.asarray(table["advantage_raw"].to_pylist(), dtype=np.float32).reshape(-1)
            labels = np.asarray(table["advantage"].to_pylist()).reshape(-1)
            expected = (flags == 1) | (raw.astype(np.float64) >= contract["advantage"]["threshold"])
            if not np.equal(labels, expected).all():
                raise ProtocolError("Invalid final advantage labels")
        for key in video_keys:
            video_path = root / relative_path(info["video_path"].format(video_key=key, **format_args))
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                height, width, _ = info["features"][key]["shape"]
                if (stream.height, stream.width, stream.average_rate) != (height, width, info["fps"]):
                    raise ProtocolError("Video shape/fps mismatch")
                if sum((1 for _ in container.decode(video=0))) != count:
                    raise ProtocolError("Video decoded frame count mismatch")
