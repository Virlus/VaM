"""Sequential upload of collect deltas into ordinary cumulative LeRobot repos."""

from __future__ import annotations

import hashlib
import json
import posixpath
from pathlib import Path

from utils.lerobot_collect_publish import SCHEMA, relative_path, validate_publication
from utils.lerobot_online.protocol import exclusive_lock, read_json, sha256_file


def _read(sftp, path):
    try:
        with sftp.open(path, "rb") as handle:
            return json.loads(handle.read())
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == 2:
            return None
        raise


def _write(sftp, path, value):
    with sftp.open(path + ".tmp", "wb") as handle:
        handle.write(json.dumps(value, sort_keys=True).encode())
        handle.flush()
    sftp.posix_rename(path + ".tmp", path)


def _exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except OSError as exc:
        if isinstance(exc, FileNotFoundError) or exc.errno == 2:
            return False
        raise


def _validate_bundle(folder):
    manifest = read_json(folder / "manifest.json")
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Invalid collect upload manifest")
    validate_publication(manifest)
    if type(manifest["sequence"]) is not int or manifest["sequence"] < 0:
        raise ValueError("Invalid upload sequence")
    from utils.lerobot_online.protocol import require_uuid

    require_uuid(manifest["batch_uid"])
    require_uuid(manifest["source_episode_id"])
    actual = {}
    for path in folder.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlinks are not supported inside ready packages")
        if path.is_file() and path.name != "manifest.json":
            name = relative_path(path.relative_to(folder).as_posix())
            parts = name.split("/")
            if len(parts) < 3 or parts[0] not in ("action", "value") or parts[1] not in ("data", "videos", "meta"):
                raise ValueError(f"Unexpected collect payload: {name}")
            actual[name] = path.stat().st_size
    if actual != manifest["files"]:
        raise ValueError("Collect payload file sizes/inventory changed")
    for role in ("action", "value"):
        info = read_json(folder / role / "meta/info.json")
        if info["total_episodes"] != manifest["sequence"] + 1:
            raise ValueError("Collect metadata sequence mismatch")
    return manifest


def _upload_one(manager, sftp, folder, manifest, state_path, state):
    digest = sha256_file(folder / "manifest.json")
    sequence = manifest["sequence"]
    identity = {"batch_uid": manifest["batch_uid"], "repo_ids": manifest["repo_ids"]}
    if any(state[key] != identity[key] for key in identity):
        raise ValueError("Remote dataset belongs to another collection batch")
    receipts = state["receipts"]
    if sequence < len(receipts):
        if receipts[sequence] != digest:
            raise ValueError("Previously uploaded episode changed")
        return
    if sequence != len(receipts):
        raise ValueError("Missing predecessor; upload order cannot be skipped")
    if state.get("pending") not in (None, digest):
        raise ValueError("Conflicting interrupted upload")
    state["pending"] = digest
    _write(sftp, state_path, state)

    # Claim each repo independently, also rejecting partially overlapping repo pairs.
    for repo in manifest["repo_ids"].values():
        target = posixpath.join(manager.remote_path, repo)
        owner_path = posixpath.join(target, ".collect_owner.json")
        owner = _read(sftp, owner_path)
        if owner is None:
            if _exists(sftp, target) and sftp.listdir(target):
                raise ValueError(f"Refusing to overwrite existing unowned dataset: {target}")
            manager._ensure_remote_dirs_recursive(sftp, target)
            _write(sftp, owner_path, identity)
        elif owner != identity:
            raise ValueError(f"Dataset already owned by another batch: {target}")

    stage = posixpath.join(manager.remote_path, ".collect_uploads", manifest["batch_uid"], folder.name)
    manager._ensure_remote_dirs_recursive(sftp, stage)
    transferred = False
    if manager._rollout_upload_method() == "rsync":
        try:
            command = manager._build_rollout_rsync_command(str(folder), stage)
            manager._run_external_transfer(command, "collect rsync upload")
            transferred = True
        except Exception:
            if not manager._rollout_rsync_fallback_enabled():
                raise
    if not transferred:
        for name in manifest["files"]:
            remote = posixpath.join(stage, name)
            manager._ensure_remote_dirs_recursive(sftp, posixpath.dirname(remote))
            method = manager._upload_file_via_sftp_from_scratch
            if "/meta/" not in name and manager._resume_rollout_upload_enabled():
                method = manager._upload_file_via_sftp_resumable
            if not method(sftp, str(folder / name), remote):
                raise OSError(f"Upload interrupted: {name}")
    if manager._stop_event.is_set():
        raise OSError("Upload stopped before publication")
    for name, size in manifest["files"].items():
        if sftp.stat(posixpath.join(stage, name)).st_size != size:
            raise ValueError(f"Incomplete remote payload: {name}")

    # Publish payload first, then metadata; info.json is last. Each file replace is atomic.
    def order(name):
        return ("/meta/" in name, name.endswith("/meta/info.json"), name)

    for name in sorted(manifest["files"], key=order):
        role, rel = name.split("/", 1)
        target = posixpath.join(manager.remote_path, manifest["repo_ids"][role], rel)
        manager._ensure_remote_dirs_recursive(sftp, posixpath.dirname(target))
        sftp.posix_rename(posixpath.join(stage, name), target)
    state["receipts"].append(digest)
    state["pending"] = None
    _write(sftp, state_path, state)
    manager._upload_log(f"Collect episode {sequence} uploaded; action/value metadata updated")


def poll_collect(manager):
    ready = Path(manager.data_dir)
    if ready.name != "ready":
        raise ValueError("Collect data_dir must point to ready/")
    if not ready.exists():
        return
    with exclusive_lock(ready.parent / ".collect_uploader.lock"):
        folders = sorted(p for p in ready.glob("episode_*") if p.is_dir() and not p.is_symlink())
        if not folders:
            return
        first = _validate_bundle(folders[0])
        key = hashlib.sha256(json.dumps(first["repo_ids"], sort_keys=True).encode()).hexdigest()
        ssh = manager._get_ssh_client(context="UPLOAD")
        sftp = None
        locked = False
        lock = posixpath.join(manager.remote_path, ".collect-upload.lock")
        try:
            sftp = ssh.open_sftp()
            manager._ensure_remote_dirs_recursive(sftp, manager.remote_path)
            sftp.mkdir(lock)  # Server-wide serialization; never steal an existing lock.
            locked = True
            state_dir = posixpath.join(manager.remote_path, ".collect_uploads")
            manager._ensure_remote_dirs_recursive(sftp, state_dir)
            state_path = posixpath.join(state_dir, key + ".json")
            state = _read(sftp, state_path) or {
                "batch_uid": first["batch_uid"],
                "repo_ids": first["repo_ids"],
                "receipts": [],
                "pending": None,
            }
            for folder in folders:
                if manager._stop_event.is_set():
                    break
                manifest = _validate_bundle(folder)
                _upload_one(manager, sftp, folder, manifest, state_path, state)
        finally:
            try:
                if locked:
                    sftp.rmdir(lock)
            finally:
                if sftp is not None:
                    sftp.close()
                ssh.close()
