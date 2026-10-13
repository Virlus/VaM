#!/usr/bin/env python3
"""Helpers for online-training checkpoint and episode policy metadata."""

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from utils.logger_config import logger

CURRENT_POLICY_FILENAME = "current_policy.json"
EPISODE_POLICY_FILENAME = "episode_policy.json"
EPISODE_POLICY_MAP_FILENAME = "episode-policy_id.json"
POLICY_ROLLOUT_STATE_FILENAME = "policy_rollout_state.json"

_EPISODE_PATTERN = re.compile(r"episode_(?P<idx>\d+)(?:_temp)?$")
_POLICY_PATTERN = re.compile(r"(?:^|[^0-9])policy[_-]?(?P<value>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_CKPT_PATTERN = re.compile(r"(?:^|[^0-9])ckpt[_-]?(?P<value>\d+)(?:[^0-9]|$)", re.IGNORECASE)
_STEP_PATTERN = re.compile(r"(?:^|[^0-9])step[_-]?(?P<value>\d+)(?:[^0-9]|$)", re.IGNORECASE)


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser()
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def parse_episode_index(name_or_path: str | Path) -> Optional[int]:
    name = Path(name_or_path).name
    match = _EPISODE_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group("idx"))


def should_count_policy_episode_result(result: Any) -> bool:
    """Return whether an episode result should consume current-policy rollout budget."""
    return str(result).strip().lower() in {"success", "failure"}


def infer_policy_id_from_text(text: str | Path | None) -> Optional[int]:
    if text is None:
        return None
    match = _POLICY_PATTERN.search(str(text))
    return int(match.group("value")) if match is not None else None


def infer_checkpoint_index_from_text(text: str | Path | None) -> Optional[int]:
    if text is None:
        return None
    text_value = str(text)
    for pattern in (_CKPT_PATTERN, _STEP_PATTERN):
        match = pattern.search(text_value)
        if match is not None:
            return int(match.group("value"))
    return None


def default_checkpoint_id(ckpt_dir: str | Path | None) -> Optional[str]:
    if ckpt_dir is None:
        return None
    path = Path(ckpt_dir)
    if path.name == "params":
        path = path.parent
    return path.name or None


def policy_directory_from_config(config: Dict[str, Any] | None, fallback_dir: str | Path) -> Path:
    config = config or {}
    policy_dir = config.get("policy_directory") or config.get("policy_dir")
    if policy_dir:
        return Path(policy_dir).expanduser()
    return Path(fallback_dir).expanduser()


def current_policy_path_from_config(config: Dict[str, Any] | None, policy_directory: str | Path) -> Path:
    config = config or {}
    path = config.get("current_policy_path") or config.get("current_policy_file")
    if path:
        return Path(path).expanduser()
    return Path(policy_directory).expanduser() / CURRENT_POLICY_FILENAME


def episode_policy_map_path_from_config(config: Dict[str, Any] | None, policy_directory: str | Path) -> Path:
    config = config or {}
    path = config.get("episode_policy_map_path") or config.get("episode_policy_id_path")
    if path:
        return Path(path).expanduser()
    return Path(policy_directory).expanduser() / EPISODE_POLICY_MAP_FILENAME


def policy_rollout_state_path_from_config(
    config: Dict[str, Any] | None,
    rollout_data_dir: str | Path,
) -> Path:
    config = config or {}
    path = config.get("policy_rollout_state_path") or config.get("rollout_policy_state_path")
    if path:
        return Path(path).expanduser()
    return Path(rollout_data_dir).expanduser() / POLICY_ROLLOUT_STATE_FILENAME


def normalize_policy_info(
    raw: Dict[str, Any] | None,
    *,
    ckpt_dir: str | Path | None = None,
    default_policy_id: Optional[int] = None,
) -> Dict[str, Any]:
    raw = dict(raw or {})
    checkpoint_dir = str(
        ckpt_dir or raw.get("checkpoint_dir") or raw.get("ckpt_dir") or raw.get("checkpoint_path") or ""
    )
    if checkpoint_dir.endswith("/params"):
        checkpoint_dir = str(Path(checkpoint_dir).parent)

    policy_id = raw.get("policy_id", raw.get("current_policy_id", default_policy_id))
    if policy_id is None:
        policy_id = infer_policy_id_from_text(checkpoint_dir)
    if policy_id is not None:
        try:
            policy_id = int(policy_id)
        except (TypeError, ValueError):
            policy_id = default_policy_id

    checkpoint_id = raw.get("checkpoint_id", raw.get("current_checkpoint_id"))
    if checkpoint_id is None:
        checkpoint_id = default_checkpoint_id(checkpoint_dir)

    checkpoint_index = raw.get("checkpoint_index", raw.get("train_step"))
    if checkpoint_index is None:
        checkpoint_index = infer_checkpoint_index_from_text(checkpoint_id or checkpoint_dir)
    if checkpoint_index is not None:
        try:
            checkpoint_index = int(checkpoint_index)
        except (TypeError, ValueError):
            checkpoint_index = None

    policy_info: Dict[str, Any] = {
        "schema_version": raw.get("schema_version", "current_policy/v1"),
        "policy_id": policy_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_index": checkpoint_index,
        "checkpoint_dir": checkpoint_dir or None,
        "params_dir": raw.get("params_dir") or (str(Path(checkpoint_dir) / "params") if checkpoint_dir else None),
        "policy_loaded_at": raw.get("policy_loaded_at") or raw.get("loaded_at") or utc_now_iso(),
        "policy_source": raw.get("policy_source", raw.get("source", "local")),
    }

    for key in (
        "base_checkpoint_id",
        "base_checkpoint_remote_path",
        "base_checkpoint_remote_params_path",
        "dataset_view_id",
        "dataset_view_ids",
        "train_step",
        "sha256",
        "checkpoint_name",
        "remote_checkpoint_root",
        "remote_checkpoint_path",
        "remote_params_path",
        "assets_dir",
        "local_assets_path",
        "remote_mtime",
        "current_policy_episode_cnt",
        "max_current_policy_rollouts",
    ):
        if key in raw:
            policy_info[key] = raw[key]
    return policy_info


def load_current_policy_info(
    current_policy_path: str | Path,
    *,
    fallback_ckpt_dir: str | Path | None = None,
    default_policy_id: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(current_policy_path).expanduser()
    if path.exists():
        try:
            return normalize_policy_info(read_json_file(path), default_policy_id=default_policy_id)
        except Exception as exc:
            logger.warning(f"Failed to read current policy info from {path}: {exc}")
    return normalize_policy_info({}, ckpt_dir=fallback_ckpt_dir, default_policy_id=default_policy_id)


def write_current_policy_info(current_policy_path: str | Path, policy_info: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_policy_info(policy_info)
    normalized["schema_version"] = "current_policy/v1"
    normalized["updated_at"] = utc_now_iso()
    atomic_write_json(current_policy_path, normalized)
    return normalized


def load_policy_rollout_state(path: str | Path) -> Dict[str, Any]:
    state_path = Path(path).expanduser()
    if not state_path.exists():
        return {}
    try:
        state = normalize_policy_info(read_json_file(state_path))
        try:
            state["current_policy_episode_cnt"] = max(0, int(state.get("current_policy_episode_cnt", 0)))
        except (TypeError, ValueError):
            state["current_policy_episode_cnt"] = 0
        return state
    except Exception as exc:
        logger.warning(f"Failed to read policy rollout state from {state_path}: {exc}")
        return {}


def write_policy_rollout_state(path: str | Path, policy_info: Dict[str, Any]) -> Dict[str, Any]:
    state = normalize_policy_info(policy_info)
    try:
        state["current_policy_episode_cnt"] = max(0, int(policy_info.get("current_policy_episode_cnt", 0)))
    except (TypeError, ValueError):
        state["current_policy_episode_cnt"] = 0
    if state.get("train_step") is None and state.get("checkpoint_index") is not None:
        state["train_step"] = state["checkpoint_index"]
    state["schema_version"] = "policy_rollout_state/v1"
    state["updated_at"] = utc_now_iso()
    atomic_write_json(path, state)
    return state


def load_episode_policy_map(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    try:
        return read_json_file(path)
    except Exception as exc:
        logger.warning(f"Failed to read episode policy map from {path}: {exc}")
        return {}


def _episode_map_keys(episode_index: int, episode_name: str | None = None) -> list[str]:
    keys = [str(episode_index), f"{episode_index:04d}", f"episode_{episode_index:04d}"]
    if episode_name:
        keys.extend([episode_name, episode_name.removesuffix("_temp")])
    return keys


def policy_info_from_episode_map(
    episode_map: Dict[str, Any],
    episode_index: int,
    *,
    episode_name: str | None = None,
) -> Dict[str, Any]:
    for key in _episode_map_keys(episode_index, episode_name=episode_name):
        if key not in episode_map:
            continue
        value = episode_map[key]
        if isinstance(value, dict):
            return normalize_policy_info(value)
        return normalize_policy_info({"policy_id": value})
    return {}


def update_episode_policy_map(
    path: str | Path,
    episode_index: int,
    policy_info: Dict[str, Any],
    *,
    episode_name: str | None = None,
) -> None:
    data = load_episode_policy_map(path)
    record = normalize_policy_info(policy_info)
    record["episode_index"] = int(episode_index)
    if episode_name:
        record["episode_name"] = episode_name
    data[str(int(episode_index))] = record
    if episode_name:
        data[episode_name] = record
        data[episode_name.removesuffix("_temp")] = record
    atomic_write_json(path, data)


def write_episode_policy_metadata(
    episode_dir: str | Path,
    policy_info: Dict[str, Any],
    *,
    episode_index: Optional[int] = None,
    episode_name: Optional[str] = None,
    result: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    episode_path = Path(episode_dir)
    metadata_path = episode_path / EPISODE_POLICY_FILENAME
    if metadata_path.exists() and not overwrite:
        return metadata_path

    if episode_index is None:
        episode_index = parse_episode_index(episode_path)
    payload = normalize_policy_info(policy_info)
    payload["schema_version"] = "episode_policy/v1"
    payload["data_source"] = 1.0
    payload["recorded_at"] = utc_now_iso()
    if episode_index is not None:
        payload["episode_index"] = int(episode_index)
    payload["episode_name"] = episode_name or episode_path.name
    if result is not None:
        payload["episode_result"] = result

    atomic_write_json(metadata_path, payload)
    return metadata_path


def list_complete_checkpoints(local_checkpoint_root: str | Path) -> list[Dict[str, Any]]:
    root = Path(local_checkpoint_root).expanduser()
    candidates = []
    for parent in (root / "archive", root):
        if not parent.exists():
            continue
        for ckpt_dir in parent.iterdir():
            if not ckpt_dir.is_dir() or ckpt_dir.name.endswith(".partial"):
                continue
            if ckpt_dir.name in {"archive", "current", "downloading"}:
                continue
            if not (ckpt_dir / "params").exists():
                continue
            if not ((ckpt_dir / "complete.txt").exists() or (ckpt_dir / "DONE").exists()):
                continue
            manifest = {}
            if (ckpt_dir / "manifest.json").exists():
                try:
                    manifest = read_json_file(ckpt_dir / "manifest.json")
                except Exception as exc:
                    logger.warning(f"Failed to read checkpoint manifest {ckpt_dir / 'manifest.json'}: {exc}")
            info = normalize_policy_info(
                manifest, ckpt_dir=ckpt_dir, default_policy_id=infer_policy_id_from_text(ckpt_dir.name)
            )
            info["checkpoint_dir"] = str(ckpt_dir)
            info["params_dir"] = str(ckpt_dir / "params")
            info["checkpoint_name"] = ckpt_dir.name
            policy_id = info.get("policy_id")
            checkpoint_index = info.get("checkpoint_index") or info.get("train_step")
            try:
                policy_key = int(policy_id)
            except (TypeError, ValueError):
                inferred_policy_id = infer_policy_id_from_text(ckpt_dir.name)
                policy_key = inferred_policy_id if inferred_policy_id is not None else -1
            try:
                checkpoint_key = int(checkpoint_index)
            except (TypeError, ValueError):
                checkpoint_key = -1
            candidates.append(((policy_key, checkpoint_key, ckpt_dir.stat().st_mtime_ns, ckpt_dir.name), info))
    candidates.sort(key=lambda item: item[0])
    return [info for _, info in candidates]


def find_latest_complete_checkpoint(local_checkpoint_root: str | Path) -> Optional[Dict[str, Any]]:
    checkpoints = list_complete_checkpoints(local_checkpoint_root)
    if not checkpoints:
        return None
    return checkpoints[-1]
