"""Upload a finished pair through existing SSH helpers; no sample conversion."""

from __future__ import annotations

import posixpath
import tempfile
from pathlib import Path

from utils.lerobot_direct import validate_direct_episode
from utils.lerobot_episode_names import bundle_directory_name
from utils.lerobot_online.protocol import (
    atomic_json,
    exclusive_lock,
    json_bytes,
    read_json,
    require_uuid,
    sha256_file,
)
from utils.lerobot_online.transfer import remote_json


def upload_direct(manager, folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    uid = require_uuid(manifest["episode_uid"])
    name = bundle_directory_name(manifest)
    if folder.name != name:
        raise ValueError("Episode directory/manifest identity mismatch")
    state_root = folder.parent.parent / "state" / "uploads"
    with exclusive_lock(state_root / f"{uid}.lock"):
        manifest = read_json(folder / "manifest.json")
        if manifest["episode_uid"] != uid or bundle_directory_name(manifest) != name:
            raise ValueError("Episode directory/manifest identity mismatch")
        marker = {
            "schema_version": "lerobot_direct_upload/v1",
            "episode_uid": uid,
            "run_name": manifest["run_name"],
            "manifest_sha256": sha256_file(folder / "manifest.json"),
        }
        remote = posixpath.join(manager.remote_path, name)
        ssh = manager._get_ssh_client(context="UPLOAD")
        try:
            sftp = ssh.open_sftp()
            try:
                existing = remote_json(sftp, posixpath.join(remote, "complete.txt"))
                if existing is not None:
                    if existing != marker:
                        raise ValueError("Conflicting remote episode: refusing to overwrite")
                else:
                    manifest = validate_direct_episode(folder)
                    manager._ensure_remote_dirs_recursive(sftp, remote)
                    paths = [item["path"] for item in manifest["files"]] + ["manifest.json"]
                    transferred = False
                    if manager._rollout_upload_method() == "rsync":
                        try:
                            with tempfile.NamedTemporaryFile("wb") as listing:
                                listing.write(b"\0".join(p.encode() for p in paths) + b"\0")
                                listing.flush()
                                command = manager._build_rollout_rsync_command(str(folder), remote)
                                command[1:1] = [
                                    "--checksum",
                                    "--from0",
                                    f"--files-from={listing.name}",
                                ]
                                manager._run_external_transfer(command, "direct LeRobot upload")
                            transferred = True
                        except Exception:
                            if not manager._rollout_rsync_fallback_enabled():
                                raise
                    if not transferred:
                        for name in paths:
                            target = posixpath.join(remote, name)
                            manager._ensure_remote_dirs_recursive(sftp, posixpath.dirname(target))
                            if not manager._upload_file_via_sftp_from_scratch(sftp, str(folder / name), target):
                                raise OSError(f"LeRobot upload failed: {name}")
                    if sha256_file(folder / "manifest.json") != marker["manifest_sha256"]:
                        raise ValueError("Manifest changed during transfer")
                    commit = posixpath.join(remote, "complete.txt")
                    with sftp.open(commit + ".tmp", "wb") as handle:
                        handle.write(json_bytes(marker))
                        handle.flush()
                    sftp.rename(commit + ".tmp", commit)
                atomic_json(state_root / f"{uid}.json", {"status": "UPLOADED", **marker})
            finally:
                sftp.close()
        finally:
            ssh.close()
    return True
