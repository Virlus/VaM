from __future__ import annotations

from contextlib import suppress
import json
import logging
from pathlib import Path
import shutil
from typing import Any

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger("vam")

INTERNAL_KEYS = {"episode_index", "frame_index", "index", "timestamp", "task_index"}
DEFAULT_MISSING_FEATURE_SPECS = {
    "advantage": {"dtype": "float32", "shape": [1], "names": ["advantage"]},
    "value_prediction": {"dtype": "float32", "shape": [1], "names": ["value_pred"]},
}


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.handlers:
        root.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping YAML in {path}, got {type(loaded).__name__}")
    loaded.setdefault("_config_path", str(path))
    return loaded


def resolve_dataset_root(
    *,
    dataset_root: str | Path | None = None,
    repo_id: str | None = None,
    output_path: str | Path | None = None,
) -> Path:
    if dataset_root is not None:
        return Path(dataset_root)
    if repo_id is None:
        raise ValueError("Either dataset_root or repo_id must be provided.")
    if output_path is not None:
        return Path(output_path) / repo_id
    return HF_LEROBOT_HOME / repo_id


def resolve_target_dataset_path(target_repo_id: str, output_path: str | Path | None = None) -> Path:
    return resolve_dataset_root(repo_id=target_repo_id, output_path=output_path)


def ensure_dataset_exists(dataset_root: Path) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not (dataset_root / "data").exists():
        raise FileNotFoundError(f"Dataset data directory does not exist: {dataset_root / 'data'}")


def find_episode_parquet_files(dataset_root: Path) -> list[Path]:
    parquet_files = sorted(dataset_root.glob("data/chunk-*/episode_*.parquet"))
    if parquet_files:
        return parquet_files
    parquet_files = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return parquet_files


def backup_episode_parquet(parquet_path: Path, dataset_root: Path) -> Path:
    """Move an episode parquet backup outside data/ so HF parquet loading cannot ingest it."""
    relative_path = parquet_path.relative_to(dataset_root)
    backup_path = dataset_root / "meta" / "parquet_backups" / relative_path
    backup_path = backup_path.with_suffix(backup_path.suffix + ".bak")
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    if backup_path.exists():
        backup_path.unlink()
    shutil.move(str(parquet_path), str(backup_path))
    return backup_path


def read_parquet_allow_duplicate_columns(parquet_path: Path) -> pa.Table:
    return pq.ParquetFile(parquet_path).read()


def parquet_column_names(parquet_path: Path) -> list[str]:
    return list(pq.ParquetFile(parquet_path).schema_arrow.names)


def find_parquet_files_with_column(dataset_root: Path, column_name: str) -> list[Path]:
    return [path for path in find_episode_parquet_files(dataset_root) if column_name in parquet_column_names(path)]


def ensure_annotation_column_writable(dataset_root: Path, column_name: str, *, overwrite: bool) -> None:
    if overwrite:
        return
    ensure_dataset_exists(dataset_root)
    info_path = dataset_root / "meta" / "info.json"
    info_has_column = False
    if info_path.exists():
        with info_path.open("r") as f:
            info = json.load(f)
        info_has_column = column_name in info.get("features", {})
    parquet_hits = find_parquet_files_with_column(dataset_root, column_name)
    if info_has_column or parquet_hits:
        locations = []
        if info_has_column:
            locations.append(str(info_path))
        locations.extend(str(path) for path in parquet_hits[:5])
        suffix = "" if len(parquet_hits) <= 5 else f" ... and {len(parquet_hits) - 5} more parquet files"
        raise ValueError(
            f"Dataset {dataset_root} already has column {column_name!r}. "
            f"Pass --overwrite to recompute it. Existing locations: {locations}{suffix}"
        )


def get_missing_feature_defaults(config: dict[str, Any]) -> dict[str, Any]:
    conversion = config.get("conversion") or {}
    raw_defaults = conversion.get("missing_feature_defaults", config.get("missing_feature_defaults", {})) or {}
    if not isinstance(raw_defaults, dict):
        raise ValueError("missing_feature_defaults must be a mapping from feature key to default value.")
    return {str(key): value for key, value in raw_defaults.items()}


def inherit_configured_missing_feature_specs(
    features: dict[str, dict[str, Any]],
    source_features: dict[str, dict[str, Any]],
    missing_feature_defaults: dict[str, Any],
) -> None:
    for key in missing_feature_defaults:
        if key in source_features:
            features.setdefault(key, source_features[key])


def inherit_configured_missing_feature_specs_from_parquet(
    features: dict[str, dict[str, Any]],
    parquet_files: list[Path],
    missing_feature_defaults: dict[str, Any],
) -> None:
    for key in missing_feature_defaults:
        if key in features or key not in DEFAULT_MISSING_FEATURE_SPECS:
            continue
        if any(key in parquet_column_names(parquet_file) for parquet_file in parquet_files):
            features[key] = dict(DEFAULT_MISSING_FEATURE_SPECS[key])


def default_feature_array(feature: dict[str, Any], default_value: Any) -> np.ndarray:
    dtype_name = str(feature.get("dtype", "float32"))
    if dtype_name == "video":
        raise ValueError("missing_feature_defaults only supports low-dimensional non-video features.")
    dtype = np.dtype(dtype_name)
    shape = tuple(feature.get("shape", ()))
    value = np.asarray(default_value, dtype=dtype)
    if value.shape == ():
        value = np.full(shape, value.item(), dtype=dtype)
    else:
        value = np.broadcast_to(value, shape).astype(dtype, copy=True)
    return np.ascontiguousarray(value)


def default_feature_column(row_count: int, feature: dict[str, Any], default_value: Any) -> pa.Array:
    default = default_feature_array(feature, default_value)
    if default.ndim != 1:
        raise ValueError(
            f"missing_feature_defaults only supports 1D low-dimensional features, got shape {default.shape}."
        )
    if default.dtype != np.float32:
        raise ValueError("missing_feature_defaults parquet fill currently supports float32 features only.")
    return pa.array([default.tolist() for _ in range(row_count)], type=pa.list_(pa.float32()))


def fill_missing_frame_features(
    frame: dict[str, Any],
    features: dict[str, dict[str, Any]],
    missing_feature_defaults: dict[str, Any],
) -> None:
    for key, default_value in missing_feature_defaults.items():
        if key in features and key not in frame:
            frame[key] = default_feature_array(features[key], default_value)


def write_info_json_features(
    dataset_root: Path,
    *,
    add_or_update: dict[str, dict[str, Any]] | None = None,
    remove: list[str] | None = None,
) -> bool:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        logger.warning("meta/info.json not found at %s", info_path)
        return False

    with info_path.open("r") as f:
        info = json.load(f)

    features = info.setdefault("features", {})
    changed = False

    if add_or_update:
        for key, spec in add_or_update.items():
            if features.get(key) != spec:
                features[key] = spec
                changed = True

    if remove:
        for key in remove:
            if key in features:
                del features[key]
                changed = True

    if changed:
        with info_path.open("w") as f:
            json.dump(info, f, indent=2)

    return changed


def infer_robot_type(repo_id: str, fallback: str = "aloha_i2rt") -> str:
    try:
        return LeRobotDatasetMetadata(repo_id).robot_type
    except Exception:
        logger.warning("Could not infer robot_type for %s, using %s", repo_id, fallback)
        return fallback


def to_numpy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def to_writable_array(value: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(value))
    if not arr.flags["C_CONTIGUOUS"] or not arr.flags["WRITEABLE"]:
        arr = np.array(arr, copy=True)
    return arr


def convert_to_feature_shape(arr: np.ndarray, feature_spec: dict[str, Any]) -> np.ndarray:
    expected_shape = tuple(feature_spec.get("shape", ()))
    if not expected_shape or arr.shape == expected_shape:
        return arr
    if arr.ndim == 3 and len(expected_shape) == 3:
        h, w, c = expected_shape
        if arr.shape == (c, h, w):
            return np.transpose(arr, (1, 2, 0))
    if arr.ndim == 0 and expected_shape == (1,):
        return np.asarray([arr.item()], dtype=arr.dtype)
    if arr.shape and arr.shape[0] == 1 and arr.shape == (1, *expected_shape):
        return arr[0]
    return arr


def coerce_frame_value_for_feature(value: Any, feature_spec: dict[str, Any]) -> Any:
    dtype = str(feature_spec.get("dtype", "")).lower()

    if dtype == "string":
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return ""
            value = value.reshape(-1)[0]
        if isinstance(value, list | tuple):
            if not value:
                return ""
            value = value[0]
        return str(value)

    arr = convert_to_feature_shape(to_writable_array(value), feature_spec)
    if dtype and dtype not in {"image", "video"}:
        with suppress(TypeError):
            arr = arr.astype(np.dtype(dtype), copy=False)
    return arr


def dedupe_parquet_column(
    parquet_path: Path,
    *,
    column_name: str,
    keep: str = "last",
    dry_run: bool = False,
    backup: bool = False,
    compression: str = "snappy",
) -> bool:
    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    dup_idxs = [i for i, name in enumerate(names) if name == column_name]
    if len(dup_idxs) <= 1:
        return False

    if keep == "first":
        keep_idx = dup_idxs[0]
    elif keep == "last":
        keep_idx = dup_idxs[-1]
    else:
        raise ValueError(f"Unsupported keep policy: {keep!r}")

    keep_col_idxs = [i for i, name in enumerate(names) if name != column_name or i == keep_idx]
    if dry_run:
        logger.info(
            "[DRY-RUN] %s: %s x%d -> keep index %d, drop %d",
            parquet_path,
            column_name,
            len(dup_idxs),
            keep_idx,
            len(dup_idxs) - 1,
        )
        return True

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for row_group_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(row_group_idx).select(keep_col_idxs)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression=compression)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    rewritten = pq.ParquetFile(tmp_path)
    count = sum(name == column_name for name in rewritten.schema_arrow.names)
    if count != 1:
        raise RuntimeError(
            f"Failed to dedupe {parquet_path}: expected 1 {column_name!r} column after rewrite, got {count}"
        )

    if backup:
        backup_episode_parquet(parquet_path, dataset_root=parquet_path.parents[2])
    tmp_path.replace(parquet_path)
    logger.info("%s: removed %d duplicate %r columns", parquet_path, len(dup_idxs) - 1, column_name)
    return True


def dedupe_dataset_column(
    dataset_root: Path,
    *,
    column_name: str,
    keep: str = "last",
    dry_run: bool = False,
    backup: bool = False,
    compression: str = "snappy",
) -> int:
    ensure_dataset_exists(dataset_root)
    parquet_files = find_episode_parquet_files(dataset_root)
    changed = 0
    for parquet_path in parquet_files:
        if dedupe_parquet_column(
            parquet_path,
            column_name=column_name,
            keep=keep,
            dry_run=dry_run,
            backup=backup,
            compression=compression,
        ):
            changed += 1
    logger.info("Done. Files with duplicate %r fixed: %d/%d", column_name, changed, len(parquet_files))
    return changed


def prepare_frame_for_write(frame: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    frame_dict: dict[str, Any] = {}
    for key, feature_spec in features.items():
        if key not in frame or key in INTERNAL_KEYS:
            continue
        frame_dict[key] = coerce_frame_value_for_feature(frame[key], feature_spec)
    if "task" in frame:
        frame_dict["task"] = frame["task"]
    return frame_dict
