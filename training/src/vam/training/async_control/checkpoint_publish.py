from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Any

import numpy as np
import orbax.checkpoint as ocp

import vam.models.model as _model
from vam.shared import array_typing as at
from vam.training.async_control._common import _atomic_write_json
from vam.training.async_control._common import _read_json
from vam.training.async_control._common import _resolve_path
from vam.training.async_control.train_state_switch import validate_checkpoint_ref
import vam.training.config as _config


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode())
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _copytree_data_only(src: Path, dst: Path) -> None:
    """Copy checkpoint bytes without preserving file metadata unsupported by some object-store mounts."""
    dst.mkdir(parents=True, exist_ok=False)
    for src_path in src.rglob("*"):
        relative_path = src_path.relative_to(src)
        dst_path = dst / relative_path
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
        elif src_path.is_file():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)


def _checkpoint_dir_from_ref(checkpoint_ref: dict[str, Any]) -> Path:
    checkpoint_path = checkpoint_ref.get("checkpoint_path") or checkpoint_ref.get("params_path")
    if checkpoint_path is None:
        raise ValueError(f"Checkpoint ref is missing checkpoint_path/params_path: {checkpoint_ref}")
    checkpoint_path = _resolve_path(checkpoint_path)
    return checkpoint_path.parent if checkpoint_path.name == "params" else checkpoint_path


def _copy_merged_assets(action_ref: dict[str, Any], value_ref: dict[str, Any], dst: Path) -> str | None:
    assets = []
    for role, checkpoint_ref in (("action", action_ref), ("value", value_ref)):
        assets_dir = _checkpoint_dir_from_ref(checkpoint_ref) / "assets"
        if assets_dir.exists():
            assets.append((role, assets_dir, _tree_digest(assets_dir)))

    if not assets:
        logging.warning("Merged checkpoint inputs did not contain assets directories; publishing params only.")
        return None

    digests = {digest for _, _, digest in assets}
    if len(digests) > 1:
        details = {role: digest for role, _, digest in assets}
        raise ValueError(f"Cannot merge checkpoint assets because role assets differ: {details}")

    source_role, source_assets_dir, digest = assets[0]
    _copytree_data_only(source_assets_dir, dst)
    logging.info("Copied merged checkpoint assets from %s checkpoint: %s", source_role, source_assets_dir)
    return digest


def _flatten_dict(tree: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    flat = {}
    for key, value in tree.items():
        path = (*prefix, str(key))
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, path))
        else:
            flat[path] = value
    return flat


def _unflatten_dict(flat: dict[tuple[str, ...], Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for path, value in flat.items():
        cursor = tree
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value
    return tree


def _path_to_str(path: tuple[str, ...]) -> str:
    return "/".join(path)


def _shape_of(value: Any) -> tuple[int, ...]:
    return tuple(np.shape(value))


def _matches_prefix(path: tuple[str, ...], prefixes: tuple[str, ...]) -> bool:
    path_str = _path_to_str(path)
    for prefix in prefixes:
        if prefix.endswith("/"):
            if path_str.startswith(prefix):
                return True
        elif path_str == prefix:
            return True
    return False


_VALUE_PARAM_PREFIXES = (
    "ValuePaliGemma/",
    "value_text_proj/",
    "value_params",
    "value_out_proj/",
    "TargetValuePaliGemma/",
    "target_value_text_proj/",
    "target_value_params",
    "target_value_out_proj/",
    "target_update_step",
)


def _merge_value_params_into_action_params(
    action_params: at.Params,
    value_params: at.Params,
) -> at.Params:
    flat_action = _flatten_dict(action_params)
    flat_value = _flatten_dict(value_params)
    value_paths = [path for path in flat_value if _matches_prefix(path, _VALUE_PARAM_PREFIXES)]
    if not value_paths:
        raise ValueError(f"Value checkpoint did not contain any params matching prefixes {_VALUE_PARAM_PREFIXES}.")

    merged = dict(flat_action)
    shape_mismatches = []
    for path in value_paths:
        if path not in merged:
            merged[path] = flat_value[path]
            continue
        if _shape_of(merged[path]) != _shape_of(flat_value[path]):
            path_str = _path_to_str(path)
            shape_mismatches.append(
                f"{path_str}: action_shape={_shape_of(merged[path])} value_shape={_shape_of(flat_value[path])}"
            )
            continue
        merged[path] = flat_value[path]

    if shape_mismatches:
        raise ValueError(f"Cannot merge value params into action checkpoint; shape mismatches: {shape_mismatches[:20]}")
    return _unflatten_dict(merged)


def _merged_checkpoint_ref_is_complete(checkpoint_ref: dict[str, Any], *, log_warnings: bool = True) -> bool:
    def _warn(message: str, *args: Any) -> None:
        if log_warnings:
            logging.warning(message, *args)

    try:
        validate_checkpoint_ref(checkpoint_ref)
    except (FileNotFoundError, ValueError) as exc:
        _warn("Existing merged checkpoint ref is not usable and will be republished: %s", exc)
        return False

    checkpoint_dir = _checkpoint_dir_from_ref(checkpoint_ref)
    if int(checkpoint_ref.get("publication_protocol", 0)) < 2:
        _warn(
            "Existing merged checkpoint %s predates verified publication; attempting republish.",
            checkpoint_dir,
        )
        return False

    ready_path = checkpoint_dir / "READY.json"
    if not ready_path.exists():
        _warn("Existing merged checkpoint %s has no READY.json; attempting republish.", checkpoint_dir)
        return False
    try:
        ready = _read_json(ready_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _warn("Existing merged checkpoint %s has invalid READY.json: %s", checkpoint_dir, exc)
        return False
    expected_params_sha256 = checkpoint_ref.get("sha256") or checkpoint_ref.get("params_sha256")
    if (
        ready.get("checkpoint_id") != checkpoint_ref.get("checkpoint_id")
        or ready.get("params_sha256") != expected_params_sha256
    ):
        _warn("Existing merged checkpoint %s has mismatched READY.json; attempting republish.", checkpoint_dir)
        return False

    assets_dir = checkpoint_dir / "assets"
    if "assets_sha256" not in checkpoint_ref:
        if assets_dir.exists():
            return True
        _warn(
            "Existing merged checkpoint %s predates assets metadata and has no assets; attempting republish.",
            checkpoint_dir,
        )
        return False

    expected_assets_sha256 = checkpoint_ref.get("assets_sha256")
    if expected_assets_sha256 is None:
        return True
    if not assets_dir.exists():
        _warn("Existing merged checkpoint %s is missing assets; attempting republish.", checkpoint_dir)
        return False
    actual_assets_sha256 = _tree_digest(assets_dir)
    if actual_assets_sha256 != expected_assets_sha256:
        _warn(
            "Existing merged checkpoint %s has assets digest mismatch; expected=%s actual=%s. Attempting republish.",
            checkpoint_dir,
            expected_assets_sha256,
            actual_assets_sha256,
        )
        return False
    return True


def _role_checkpoint_ref_is_complete(checkpoint_ref: dict[str, Any], *, log_warnings: bool = True) -> bool:
    try:
        validate_checkpoint_ref(checkpoint_ref)
    except (FileNotFoundError, ValueError) as exc:
        if log_warnings:
            logging.warning("Existing role checkpoint ref is not usable and will be republished: %s", exc)
        return False
    return True


def _save_params(params_dir: Path, params: at.Params) -> None:
    params_dir.parent.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_dir, {"params": params})


def _validate_saved_params(params_dir: Path) -> None:
    """Read every saved array before making a merged checkpoint discoverable."""
    restored = _model.restore_params(params_dir, restore_type=np.ndarray)
    flat_restored = _flatten_dict(restored)
    if not flat_restored:
        raise ValueError(f"Saved checkpoint contains no parameters: {params_dir}")
    # Numpy restore materializes every TensorStore array and therefore catches
    # missing or unreadable OCDBT chunks before publication.
    del flat_restored
    del restored
    gc.collect()


class CheckpointPublisher:
    def __init__(self, config: _config.CheckpointPublisherConfig):
        self.config = config
        initial_policy_id = int(config.policy_id or 0)
        self._next_policy_id = initial_policy_id + int(config.increment_policy_id)
        self._lock = threading.Lock()

    def reserve_policy_id(self) -> int:
        if not self.config.increment_policy_id:
            return int(self.config.policy_id or 0)
        with self._lock:
            policy_id = self._next_policy_id
            self._next_policy_id += 1
        return policy_id

    def resolve_policy_id(self, target_policy_ids: set[int | None]) -> int:
        """Resolve the published policy generation from the active dataset views."""
        if self.config.merge is None:
            return self.reserve_policy_id()
        if None in target_policy_ids or len(target_policy_ids) != 1:
            raise ValueError(
                "Split checkpoint publication requires one shared non-null target_policy_id, got "
                f"{sorted(str(policy_id) for policy_id in target_policy_ids)}."
            )
        return int(next(iter(target_policy_ids))) + 1

    def _checkpoints_root(self, output_dir: Path) -> Path:
        if self.config.merge is None:
            return output_dir / "checkpoints"
        return output_dir / "checkpoints" / self.config.merge.role

    def _write_latest_if_newer(
        self, output_dir: Path, latest: dict[str, Any], *, checkpoints_root: Path | None = None
    ) -> bool:
        latest_path = (checkpoints_root or output_dir / "checkpoints") / "latest.json"
        with self._lock:
            if latest_path.exists():
                current_latest = _read_json(latest_path)
                if int(current_latest.get("train_step", -1)) > int(latest.get("train_step", -1)):
                    logging.info(
                        "Published checkpoint %s without moving latest.json because current latest is newer: %s",
                        latest.get("checkpoint_id"),
                        current_latest.get("checkpoint_id"),
                    )
                    return False
            _atomic_write_json(latest_path, latest)
            return True

    def _write_by_policy_ref(self, checkpoints_root: Path, latest: dict[str, Any]) -> None:
        policy_id = int(latest["policy_id"])
        ref_path = checkpoints_root / "by_policy" / f"{policy_id}.json"
        with self._lock:
            if ref_path.exists():
                current = _read_json(ref_path)
                if int(current.get("train_step", -1)) > int(latest.get("train_step", -1)):
                    logging.info(
                        "Published checkpoint %s without moving by_policy/%s.json because current revision is "
                        "newer: %s",
                        latest.get("checkpoint_id"),
                        policy_id,
                        current.get("checkpoint_id"),
                    )
                    return
            _atomic_write_json(ref_path, latest)

    def _write_by_checkpoint_ref(self, checkpoints_root: Path, latest: dict[str, Any]) -> None:
        checkpoint_id = str(latest["checkpoint_id"])
        _atomic_write_json(checkpoints_root / "by_checkpoint" / f"{checkpoint_id}.json", latest)

    def _role_checkpoint_ref_path(self, checkpoints_root: Path, *, policy_id: int, step: int) -> Path:
        checkpoint_id = f"ckpt_{policy_id:06d}_step_{step:08d}"
        return checkpoints_root / "by_checkpoint" / f"{checkpoint_id}.json"

    def _merged_checkpoint_ref_path(self, output_dir: Path, *, policy_id: int, step: int) -> Path:
        checkpoint_id = f"merged_{policy_id:06d}_step_{step:08d}"
        return output_dir / "checkpoints" / "by_checkpoint" / f"{checkpoint_id}.json"

    def publish(
        self,
        *,
        source_checkpoint_dir: Path,
        step: int,
        dataset_view_id: str | None,
        dataset_view_ids: dict[str, str | None] | None = None,
        policy_id: int | None = None,
        base_checkpoint_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.config.output_dir:
            return None

        output_dir = _resolve_path(self.config.output_dir)
        policy_id = self.reserve_policy_id() if policy_id is None else int(policy_id)
        checkpoint_id = f"ckpt_{policy_id:06d}_step_{step:08d}"
        checkpoints_root = self._checkpoints_root(output_dir)
        existing_ref_path = checkpoints_root / "by_checkpoint" / f"{checkpoint_id}.json"
        if existing_ref_path.exists():
            existing = _read_json(existing_ref_path)
            if _role_checkpoint_ref_is_complete(existing):
                logging.info(
                    "Checkpoint role %s already published %s for policy_id=%d step=%d; reusing it.",
                    self.config.merge.role if self.config.merge is not None else "default",
                    existing.get("checkpoint_id"),
                    policy_id,
                    step,
                )
                if self.config.merge is not None:
                    self._maybe_publish_merged_policy(output_dir=output_dir, latest=existing)
                return existing
            logging.warning(
                "Checkpoint role %s policy_id=%d step=%d registry exists but is incomplete; overwriting %s.",
                self.config.merge.role if self.config.merge is not None else "default",
                policy_id,
                step,
                existing_ref_path,
            )
        registry_checkpoint_dir = checkpoints_root / checkpoint_id
        if self.config.copy_checkpoint:
            tmp_dir = registry_checkpoint_dir.with_suffix(".partial")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            logging.info(
                "Publishing checkpoint %s by copying %s to %s",
                checkpoint_id,
                source_checkpoint_dir,
                registry_checkpoint_dir,
            )
            try:
                _copytree_data_only(source_checkpoint_dir, tmp_dir)
                if registry_checkpoint_dir.exists():
                    shutil.rmtree(registry_checkpoint_dir)
                os.replace(tmp_dir, registry_checkpoint_dir)
            except Exception:
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
        else:
            registry_checkpoint_dir = source_checkpoint_dir

        manifest = {
            "schema_version": "checkpoint_manifest/v1",
            "policy_id": policy_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(registry_checkpoint_dir),
            "train_step": int(step),
            "dataset_view_id": dataset_view_id,
            "dataset_view_ids": dataset_view_ids or {},
            "base_checkpoint_id": base_checkpoint_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_write_json(registry_checkpoint_dir / "manifest.json", manifest)
        (registry_checkpoint_dir / "DONE").write_text("done\n")
        latest = {
            **manifest,
            "schema_version": "checkpoint_registry/v1",
            "sha256": _tree_digest(registry_checkpoint_dir / "params"),
        }
        if self.config.merge is not None:
            latest["role"] = self.config.merge.role
            manifest["role"] = self.config.merge.role
            _atomic_write_json(registry_checkpoint_dir / "manifest.json", manifest)
        self._write_latest_if_newer(output_dir, latest, checkpoints_root=checkpoints_root)
        self._write_by_policy_ref(checkpoints_root, latest)
        self._write_by_checkpoint_ref(checkpoints_root, latest)
        registries_dir = (
            output_dir / "registries" / (self.config.merge.role if self.config.merge is not None else "checkpoints")
        )
        registries_dir.mkdir(parents=True, exist_ok=True)
        with (registries_dir / "checkpoints.jsonl").open("a") as f:
            f.write(json.dumps(latest, sort_keys=True) + "\n")
        logging.info("Published checkpoint %s to %s", checkpoint_id, registry_checkpoint_dir)
        if self.config.merge is not None:
            self._maybe_publish_merged_policy(output_dir=output_dir, latest=latest)
        return latest

    def _maybe_publish_merged_policy(self, *, output_dir: Path, latest: dict[str, Any]) -> dict[str, Any] | None:
        merge_config = self.config.merge
        if merge_config is None:
            return None
        role = merge_config.role
        peer_role = "value" if role == "action" else "action"

        policy_id = int(latest["policy_id"])
        train_step = int(latest["train_step"])
        public_ref_path = self._merged_checkpoint_ref_path(output_dir, policy_id=policy_id, step=train_step)
        if public_ref_path.exists():
            public_ref = _read_json(public_ref_path)
            if _merged_checkpoint_ref_is_complete(public_ref, log_warnings=False):
                return public_ref
            logging.warning(
                "Existing merged checkpoint ref %s is incomplete; attempting to republish.",
                public_ref_path,
            )

        peer_ref_path = self._role_checkpoint_ref_path(
            output_dir / "checkpoints" / peer_role,
            policy_id=policy_id,
            step=train_step,
        )
        if not peer_ref_path.exists():
            logging.info(
                "Checkpoint %s for role %s is waiting for peer role %s policy_id=%s step=%s.",
                latest.get("checkpoint_id"),
                role,
                peer_role,
                policy_id,
                train_step,
            )
            return None

        lock_dir = output_dir / "locks" / f"merge_policy_{policy_id:06d}_step_{train_step:08d}.lock"
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logging.info(
                "Merge for policy_id=%s step=%s is already claimed by another publisher.",
                policy_id,
                train_step,
            )
            return None

        try:
            if public_ref_path.exists():
                public_ref = _read_json(public_ref_path)
                if _merged_checkpoint_ref_is_complete(public_ref, log_warnings=False):
                    return public_ref
                logging.warning(
                    "Existing merged checkpoint ref %s is incomplete; republishing it.",
                    public_ref_path,
                )
            peer_latest = _read_json(peer_ref_path)
            if not _role_checkpoint_ref_is_complete(peer_latest):
                logging.info(
                    "Checkpoint %s for role %s is waiting for peer role %s policy_id=%s step=%s to finish.",
                    latest.get("checkpoint_id"),
                    role,
                    peer_role,
                    policy_id,
                    train_step,
                )
                return None
            refs = {role: latest, peer_role: peer_latest}
            action_ref = refs.get("action")
            value_ref = refs.get("value")
            if action_ref is None or value_ref is None:
                raise ValueError(f"Merge requires action and value refs, got roles {sorted(refs)}.")
            return self._publish_merged_policy(
                output_dir=output_dir,
                policy_id=policy_id,
                action_ref=action_ref,
                value_ref=value_ref,
            )
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)

    def wait_for_merged_policy(
        self,
        *,
        policy_id: int,
        role_checkpoint: dict[str, Any],
        poll_interval_s: float = 5.0,
    ) -> dict[str, Any]:
        if self.config.merge is None or not self.config.output_dir:
            raise ValueError("wait_for_merged_policy requires split checkpoint merge configuration.")
        output_dir = _resolve_path(self.config.output_dir)
        train_step = int(role_checkpoint["train_step"])
        public_ref_path = self._merged_checkpoint_ref_path(output_dir, policy_id=int(policy_id), step=train_step)
        last_log_time = 0.0
        while True:
            self._maybe_publish_merged_policy(output_dir=output_dir, latest=role_checkpoint)
            if public_ref_path.exists():
                merged = _read_json(public_ref_path)
                if int(merged.get("policy_id", -1)) != int(policy_id):
                    raise ValueError(
                        f"Merged checkpoint policy mismatch: expected {policy_id}, got {merged.get('policy_id')}."
                    )
                if _merged_checkpoint_ref_is_complete(merged, log_warnings=False):
                    return merged
            now = time.monotonic()
            if now - last_log_time >= 30:
                logging.info(
                    "Role %s checkpoint %s is waiting for merged policy_id=%d step=%d.",
                    self.config.merge.role,
                    role_checkpoint.get("checkpoint_id"),
                    policy_id,
                    train_step,
                )
                last_log_time = now
            time.sleep(max(0.1, float(poll_interval_s)))

    def wait_for_latest_merged_policy(self, *, policy_id: int, poll_interval_s: float = 5.0) -> dict[str, Any]:
        if self.config.merge is None or not self.config.output_dir:
            raise ValueError("wait_for_latest_merged_policy requires split checkpoint merge configuration.")
        output_dir = _resolve_path(self.config.output_dir)
        public_ref_path = output_dir / "checkpoints" / "by_policy" / f"{int(policy_id)}.json"
        last_log_time = 0.0
        while True:
            if public_ref_path.exists():
                merged = _read_json(public_ref_path)
                if int(merged.get("policy_id", -1)) != int(policy_id):
                    raise ValueError(
                        f"Merged checkpoint policy mismatch: expected {policy_id}, got {merged.get('policy_id')}."
                    )
                if _merged_checkpoint_ref_is_complete(merged, log_warnings=False):
                    return merged
            now = time.monotonic()
            if now - last_log_time >= 30:
                logging.info("Waiting for latest merged checkpoint for policy_id=%d.", policy_id)
                last_log_time = now
            time.sleep(max(0.1, float(poll_interval_s)))

    def _publish_merged_policy(
        self,
        *,
        output_dir: Path,
        policy_id: int,
        action_ref: dict[str, Any],
        value_ref: dict[str, Any],
    ) -> dict[str, Any]:
        merge_config = self.config.merge
        if merge_config is None:
            raise ValueError("Cannot publish a merged policy without checkpoint_publisher.merge.")
        merged_root = _resolve_path(merge_config.output_dir)
        action_step = int(action_ref.get("train_step", -1))
        value_step = int(value_ref.get("train_step", -1))
        if action_step != value_step:
            raise ValueError(
                "Split checkpoint merge requires matching action/value train_step revisions, got "
                f"action={action_step} value={value_step} for policy_id={policy_id}."
            )
        train_step = max(int(action_ref.get("train_step", -1)), int(value_ref.get("train_step", -1)))
        checkpoint_id = f"merged_{policy_id:06d}_step_{train_step:08d}"
        checkpoint_dir = merged_root / str(policy_id) / checkpoint_id
        tmp_dir = checkpoint_dir.with_suffix(".partial")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)

        action_params = _model.restore_params(validate_checkpoint_ref(action_ref), restore_type=np.ndarray)
        value_params = _model.restore_params(validate_checkpoint_ref(value_ref), restore_type=np.ndarray)
        merged_params = _merge_value_params_into_action_params(action_params, value_params)
        _save_params(tmp_dir / "params", merged_params)
        assets_sha256 = _copy_merged_assets(action_ref, value_ref, tmp_dir / "assets")

        # Validate by reading the complete checkpoint back before publishing
        # any registry entry. Release merge inputs first to bound peak memory.
        del action_params
        del value_params
        del merged_params
        gc.collect()
        _validate_saved_params(tmp_dir / "params")
        params_sha256 = _tree_digest(tmp_dir / "params")

        manifest = {
            "schema_version": "checkpoint_manifest/v1",
            "publication_protocol": 2,
            "policy_id": policy_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(checkpoint_dir),
            "train_step": train_step,
            "dataset_view_id": action_ref.get("dataset_view_id") or value_ref.get("dataset_view_id"),
            "dataset_view_ids": {
                "action": action_ref.get("dataset_view_ids", {}),
                "value": value_ref.get("dataset_view_ids", {}),
            },
            "base_checkpoint_id": action_ref.get("base_checkpoint_id") or value_ref.get("base_checkpoint_id"),
            "action_checkpoint_id": action_ref.get("checkpoint_id"),
            "value_checkpoint_id": value_ref.get("checkpoint_id"),
            "action_checkpoint_path": action_ref.get("checkpoint_path"),
            "value_checkpoint_path": value_ref.get("checkpoint_path"),
            "merged_from_roles": ["action", "value"],
            "assets_sha256": assets_sha256,
            "params_sha256": params_sha256,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_write_json(tmp_dir / "manifest.json", manifest)
        _atomic_write_json(
            tmp_dir / "READY.json",
            {
                "schema_version": "checkpoint_ready/v1",
                "checkpoint_id": checkpoint_id,
                "params_sha256": params_sha256,
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        # DONE is written last inside staging. Public refs and latest.json are
        # still absent until the verified directory is atomically promoted.
        (tmp_dir / "DONE").write_text("done\n")
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        os.replace(tmp_dir, checkpoint_dir)

        latest = {
            **manifest,
            "schema_version": "checkpoint_registry/v1",
            "sha256": params_sha256,
        }
        # Publish immutable refs first. latest.json is the commit point observed
        # by rollout machines and must be the final registry write.
        self._write_by_checkpoint_ref(output_dir / "checkpoints", latest)
        self._write_by_policy_ref(output_dir / "checkpoints", latest)
        registries_dir = output_dir / "registries" / "merged"
        registries_dir.mkdir(parents=True, exist_ok=True)
        with (registries_dir / "checkpoints.jsonl").open("a") as f:
            f.write(json.dumps(latest, sort_keys=True) + "\n")
        self._write_latest_if_newer(output_dir, latest)
        logging.info(
            "Published merged policy %s from action=%s value=%s to %s",
            checkpoint_id,
            action_ref.get("checkpoint_id"),
            value_ref.get("checkpoint_id"),
            checkpoint_dir,
        )
        return latest


class CheckpointSwitchWatcher:
    def __init__(self, config: _config.CheckpointSwitchConfig):
        self.config = config
        self._last_checkpoint_id: str | None = None
        self._last_mtime_ns: int | None = None

    def poll(self, step: int) -> dict[str, Any] | None:
        if not self.config.latest_path:
            return None
        if step % max(1, self.config.poll_interval_steps) != 0:
            return None
        latest_path = _resolve_path(self.config.latest_path)
        if not latest_path.exists():
            logging.info("Checkpoint switch registry is not ready yet: %s", latest_path)
            return None
        stat = latest_path.stat()
        latest = _read_json(latest_path)
        checkpoint_id = str(latest.get("checkpoint_id") or "")
        if not checkpoint_id:
            raise ValueError(f"Checkpoint switch registry {latest_path} is missing checkpoint_id.")
        if checkpoint_id == self._last_checkpoint_id and stat.st_mtime_ns == self._last_mtime_ns:
            return None
        self._last_checkpoint_id = checkpoint_id
        self._last_mtime_ns = stat.st_mtime_ns
        return latest
