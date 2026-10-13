"""Preserve recorder advantages computed on the dense camera sequence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa

ADVANTAGE_ALGORITHM = "dense_camera_gae/v1"


def read_advantage_recipe(bundle: Path) -> dict:
    """Require explicit collector parameters; never guess gamma from training defaults."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    # Accept the old marker on already-recorded LeRobot packages; no old-format reader is used.
    if manifest.get("sample_semantics") not in {"lerobot_camera_v1", "legacy_nedf_online_v1"}:
        raise ValueError(f"{bundle}: unsupported camera advantage semantics")
    recipe = manifest["recipe"]
    advantage = recipe["advantage"]
    result = {
        "window_size": recipe["window_size"],
        "camera_fps": float(recipe["fps"]),
        "action_fps": float(advantage["action_fps"]),
        "gamma": float(advantage["gamma"]),
        "gae_lambda": float(advantage["gae_lambda"]),
        "horizon": advantage["horizon"],
    }
    if advantage.get("method", "gae") != "gae" or advantage.get("percentile", 70) != 70:
        raise ValueError(f"{bundle}: registration requires GAE with policy-wide p70")
    for key in ("window_size", "horizon"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{bundle}: {key} must be a positive integer")
    for key in ("gamma", "gae_lambda"):
        if not np.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"{bundle}: {key} must be finite and between 0 and 1")
    for key in ("camera_fps", "action_fps"):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"{bundle}: {key} must be finite and positive")
    return result


def scalar_column(table: pa.Table, key: str) -> np.ndarray:
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if values.shape not in {(table.num_rows,), (table.num_rows, 1)} or not np.isfinite(values).all():
        raise ValueError(f"{key} must have one finite scalar per frame")
    return values.reshape(-1)


def uploaded_camera_advantage(source: pa.Table, registered: pa.Table) -> np.ndarray:
    """Read pre-downsampling GAE from the immutable upload, not action-endpoint values.

    The retained table does not contain every dense camera value. Shifting its rows
    by ten cannot reconstruct the original camera +30 computation at irregular gaps.
    """
    if source.num_rows != registered.num_rows:
        raise ValueError("Uploaded camera advantage row count differs from registered episode")
    for key in ("value_prediction", "control_flag", "data_source"):
        if not np.array_equal(scalar_column(source, key), scalar_column(registered, key)):
            raise ValueError(f"Uploaded camera advantage alignment differs for {key}")
    return scalar_column(source, "advantage_raw")


def raw_advantage_column(table: pa.Table, values: np.ndarray) -> pa.Array:
    """Preserve the uploaded scalar/list layout and dtype in registered copies."""
    dtype = table.schema.field("advantage_raw").type
    nested = pa.types.is_list(dtype) or pa.types.is_fixed_size_list(dtype)
    return pa.array(values[:, None].tolist() if nested else values.tolist(), type=dtype)
