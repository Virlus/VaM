"""Human-readable rollout directories with UUID identities inside manifests."""

import re
from pathlib import Path

from utils.lerobot_online.protocol import read_json, require_uuid
from utils.lerobot_progress import run_rollout_history


def episode_name(index):
    if type(index) is not int or index < 0:
        raise ValueError("Episode index must be a nonnegative integer")
    return f"episode_{index:03d}"


def bundle_directory_name(manifest):
    uid = require_uuid(manifest["episode_uid"])
    name = manifest.get("episode_name")
    if name is None:
        return uid  # Previously sealed/uploaded bundles keep their original path.
    if name != episode_name(manifest.get("rollout_index")):
        raise ValueError("Episode directory name does not match rollout_index")
    return name


def next_episode_index(root, run_name):
    """Continue history and never reuse a numbered directory or retained cache."""
    root = Path(root)
    next_index = len(run_rollout_history(root, run_name))
    for stage in ("capture", "building", "ready"):
        for directory in (root / stage).glob("*"):
            if not directory.is_dir() or directory.is_symlink():
                continue
            match = re.fullmatch(r"episode_(\d{3,})", directory.name)
            if match:
                next_index = max(next_index, int(match.group(1)) + 1)
            path = directory / "manifest.json"
            try:
                manifest = read_json(path)
            except FileNotFoundError:
                continue  # A worker may have just promoted the cache to ready.
            if manifest.get("episode_name") is not None:
                bundle_directory_name(manifest)
                next_index = max(next_index, manifest["rollout_index"] + 1)
    return next_index
