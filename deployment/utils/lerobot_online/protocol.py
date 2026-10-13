"""Durable local files and shared, fail-closed collection protocol checks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath


class ProtocolError(ValueError):
    """Invalid identity, content or state; caller must retain source data."""


def json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def read_json(path):
    def reject(value):
        raise ProtocolError(f"Non-finite JSON number: {value}")

    return json.loads(Path(path).read_bytes(), parse_constant=reject)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_uuid(value):
    try:
        valid = isinstance(value, str) and str(uuid.UUID(value)) == value
    except ValueError:
        valid = False
    if not valid:
        raise ProtocolError("Expected a canonical UUID")
    return value


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ProtocolError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(ord(char) < 32 for char in value)
    ):
        raise ProtocolError(f"Unsafe relative path: {value!r}")
    return value


def require_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ProtocolError("Expected a lowercase SHA-256 digest")
    return value


def validate_contract(contract):
    """Validate identities and the supported fixed-per-policy horizon-one recipe."""
    if contract.get("schema_version") != "lerobot_collection_contract/v1":
        raise ProtocolError("Unsupported collection contract")
    for key in (
        "contract_id",
        "run_name",
        "checkpoint_id",
        "value_checkpoint_id",
        "recipe_id",
    ):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ProtocolError(f"Missing contract {key}")
    for key in ("policy_id", "checkpoint_index"):
        if type(contract.get(key)) is not int or not 0 <= contract[key] <= 2**24:
            raise ProtocolError(f"{key} cannot be represented exactly as float32")
    for key in (
        "checkpoint_ref_sha256",
        "recipe_sha256",
        "feature_schema_sha256",
        "action_semantics_sha256",
    ):
        require_hash(contract.get(key))
    relative_path(contract["checkpoint_ref"])
    if (
        contract.get("codebase_version") != "v2.1"
        or contract.get("lerobot_revision") != "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
    ):
        raise ProtocolError("Unsupported LeRobot revision")
    advantage = contract["advantage"]
    expected = {
        "method": "gae",
        "horizon": 1,
        "threshold_mode": "fixed_per_policy",
        "human_override": 1,
        "comparison": "ge",
        "raw_dtype": "float32",
    }
    if any(advantage.get(key) != value for key, value in expected.items()):
        raise ProtocolError("Unsupported advantage recipe")
    for key in ("gamma", "gae_lambda", "threshold"):
        value = advantage.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ProtocolError(f"Invalid advantage {key}")
    if not 0 < advantage["gamma"] <= 1 or not 0 <= advantage["gae_lambda"] <= 1:
        raise ProtocolError("Invalid discount parameters")
    if not advantage.get("threshold_version") or not advantage.get("threshold_source"):
        raise ProtocolError("Threshold provenance is required")
    json_bytes(contract)
    return contract
