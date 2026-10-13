#!/usr/bin/env python3
"""
Data Upload Manager - Upload data folders to remote server via SFTP.

This module provides functionality to monitor a directory for new folders
and upload them to a remote server via SFTP.

Author: Dong Jiulong
"""

import os
import sys
import time
import threading
import json
import logging
import math
import posixpath
import shlex
import shutil
import stat
import subprocess
from typing import Set, Optional, List, Dict, Any
from pathlib import Path

# Ensure project root is on PYTHONPATH when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import paramiko
from paramiko import SSHClient, SFTPClient

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from utils.logger_config import logger
from utils.online_training import (
    atomic_write_json,
    current_policy_path_from_config,
    episode_policy_map_path_from_config,
    infer_checkpoint_index_from_text,
    infer_policy_id_from_text,
    normalize_policy_info,
    policy_directory_from_config,
    read_json_file,
    utc_now_iso,
    write_current_policy_info,
)
from utils.transfer_rate_limiter import TransferRateLimiter


def load_config_from_hydra(config_name: str) -> Dict[str, Any]:
    """
    Load configuration from a Hydra config file.

    Args:
        config_name: Name of the config file (e.g., collect)

    Returns:
        Dictionary containing the augmental_data_upload configuration
    """
    from hydra import initialize, compose
    from omegaconf import OmegaConf

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=[])

        augmental_cfg = cfg.get("augmental_data_upload", {})

    logger.info(f"Loaded config from: {config_name}")
    return OmegaConf.to_container(augmental_cfg, resolve=True) if augmental_cfg else {}


class DataUploadManager:
    """
    Manages uploading of data folders to a remote server via SFTP.

    This class runs as a background thread that polls a data directory
    at regular intervals and uploads new non-hidden folders (not ending with 'temp')
    to a remote SFTP server.
    """

    def __init__(
        self,
        data_dir: str,
        remote_path: str,
        host: str,
        port: int,
        username: str,
        poll_interval: float,
        ssh_key_path: Optional[str],
        online_training_config: Optional[Dict[str, Any]] = None,
        upload_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the Data Upload Manager.

        Args:
            data_dir: Local directory to monitor for new folders
            host: SFTP server hostname
            port: SFTP server port
            username: SFTP username
            remote_path: Remote directory path on SFTP server
            poll_interval: Polling interval in seconds (default: 5.0)
            ssh_key_path: Path to SSH private key file
        """
        self.data_dir = data_dir
        self.host = host
        self.port = port
        self.username = username
        self.remote_path = remote_path
        self.poll_interval = poll_interval
        self.ssh_key_path = os.path.expanduser(ssh_key_path) if ssh_key_path else None
        self.upload_config = dict(upload_config or {})
        self.online_training_config = dict(online_training_config or {})
        self.online_training_enabled = bool(self.online_training_config.get("enabled", False))
        self.policy_directory = policy_directory_from_config(
            self.online_training_config,
            fallback_dir=Path(self.data_dir) / "policy",
        )
        self.current_policy_path = current_policy_path_from_config(
            self.online_training_config,
            self.policy_directory,
        )
        self.episode_policy_map_path = episode_policy_map_path_from_config(
            self.online_training_config,
            self.policy_directory,
        )
        self.local_checkpoint_root = Path(
            self.online_training_config.get("local_checkpoint_root")
            or self.online_training_config.get("local_checkpoint_dir")
            or (self.policy_directory / "checkpoints")
        ).expanduser()

        self._running = False
        self._stop_event = threading.Event()
        # One payload stream per direction; budgets persist across SFTP files.
        self.upload_bandwidth_mbps = self._bandwidth_limit("upload", 50.0)
        self.download_bandwidth_mbps = self._bandwidth_limit("download", 70.0)
        self._upload_limiter = TransferRateLimiter(self.upload_bandwidth_mbps * 1_000_000, self._stop_event)
        self._download_limiter = TransferRateLimiter(self.download_bandwidth_mbps * 1_000_000, self._stop_event)
        self._thread: Optional[threading.Thread] = None
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._uploaded_folders: Set[str] = set()
        self._downloaded_checkpoints: Set[str] = set()
        self._latest_remote_policy_generation_id: Optional[int] = None
        self._latest_remote_policy_info: Dict[str, Any] = {}
        self._latest_remote_policy_observed_at: Optional[float] = None
        self._last_log_times: Dict[str, float] = {}

        logger.info(f"DataUploadManager initialized: data_dir={data_dir}, host={host}:{port}")
        if self.online_training_enabled:
            logger.info(
                "DataUploadManager online training enabled: "
                f"policy_directory={self.policy_directory}, local_checkpoint_root={self.local_checkpoint_root}"
            )

    def _bandwidth_limit(self, direction: str, default: float) -> float:
        # MBps means decimal megabytes per second, not megabits (Mbps).
        key = f"{direction}_bandwidth_mbps"
        legacy_key = f"{direction}_bandwidth_mib_s"
        value = default
        for config in (self.upload_config, self.online_training_config):
            if key in config:
                value = float(config[key])
                break
            if legacy_key in config:
                value = float(config[legacy_key]) * 1024 * 1024 / 1_000_000
                break
        if not math.isfinite(value) or value < 1024 / 1_000_000:
            raise ValueError(f"{key} must be finite and at least 0.001024 MBps")
        return value

    def _should_log(self, key: str, interval_seconds: Optional[float]) -> bool:
        if interval_seconds is None or interval_seconds <= 0:
            return True
        now = time.monotonic()
        last = self._last_log_times.get(key)
        if last is not None and now - last < interval_seconds:
            return False
        self._last_log_times[key] = now
        return True

    def _prefixed_log(
        self,
        tag: str,
        message: str,
        *,
        level: str = "info",
        color: str = "cyan",
        throttle_key: Optional[str] = None,
        throttle_seconds: Optional[float] = None,
    ) -> None:
        if throttle_key and not self._should_log(throttle_key, throttle_seconds):
            return
        log_message = f"<{color}>[{tag}]</{color}> {message}"
        try:
            logger.opt(colors=True).log(level.upper(), log_message)
        except Exception:
            plain_message = f"[{tag}] {message}"
            getattr(logger, level.lower())(plain_message)

    def _upload_log(
        self,
        message: str,
        *,
        level: str = "info",
        throttle_key: Optional[str] = None,
        throttle_seconds: Optional[float] = None,
    ) -> None:
        self._prefixed_log(
            "UPLOAD",
            message,
            level=level,
            color="green",
            throttle_key=throttle_key,
            throttle_seconds=throttle_seconds,
        )

    def _ckpt_log(
        self,
        message: str,
        *,
        level: str = "info",
        throttle_key: Optional[str] = None,
        throttle_seconds: Optional[float] = None,
    ) -> None:
        self._prefixed_log(
            "CKPT",
            message,
            level=level,
            color="magenta",
            throttle_key=throttle_key,
            throttle_seconds=throttle_seconds,
        )

    def _connect_log_interval(self, context: str) -> float:
        context_key = str(context).lower()
        specific_key = f"{context_key}_connect_log_interval"
        alias_key = "checkpoint_connect_log_interval" if context_key == "ckpt" else specific_key
        return float(
            self.online_training_config.get(specific_key)
            or self.online_training_config.get(alias_key)
            or self.online_training_config.get("sftp_connect_log_interval")
            or 60.0
        )

    def _status_log_interval(self, context: str) -> float:
        context_key = str(context).lower()
        specific_key = f"{context_key}_status_log_interval"
        alias_key = "checkpoint_status_log_interval" if context_key == "ckpt" else specific_key
        if context_key == "upload":
            configured = self._upload_config_value(specific_key, "status_log_interval")
            if configured:
                return float(configured)
        return float(
            self.online_training_config.get(specific_key)
            or self.online_training_config.get(alias_key)
            or self.online_training_config.get("status_log_interval")
            or 60.0
        )

    def _progress_log_interval(self, context: str) -> float:
        context_key = str(context).lower()
        specific_key = f"{context_key}_progress_log_interval"
        alias_key = "checkpoint_progress_log_interval" if context_key == "ckpt" else specific_key
        if context_key == "upload":
            configured = self._upload_config_value(
                specific_key,
                "rollout_progress_log_interval",
                "progress_log_interval",
            )
            if configured:
                return float(configured)
        return float(
            self.online_training_config.get(specific_key)
            or self.online_training_config.get(alias_key)
            or self.online_training_config.get("progress_log_interval")
            or 15.0
        )

    def _upload_config_value(self, *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in self.upload_config:
                return self.upload_config[key]
            if key in self.online_training_config:
                return self.online_training_config[key]
        return default

    @staticmethod
    def _suppress_paramiko_log_channel(log_channel: str) -> None:
        paramiko_logger = logging.getLogger(log_channel)
        paramiko_logger.propagate = False
        paramiko_logger.setLevel(logging.CRITICAL + 1)
        if not any(isinstance(handler, logging.NullHandler) for handler in paramiko_logger.handlers):
            paramiko_logger.addHandler(logging.NullHandler())

    def _get_ssh_client(self, context: str = "SFTP") -> SSHClient:
        """
        Create and configure an SSH client with key-based authentication.

        Returns:
            Configured SSHClient instance
        """
        tag = "CKPT" if str(context).upper() == "CKPT" else "UPLOAD" if str(context).upper() == "UPLOAD" else "SFTP"
        color = "magenta" if tag == "CKPT" else "green" if tag == "UPLOAD" else "cyan"
        context_key = tag.lower()
        timeout = float(
            self._upload_config_value(
                f"{context_key}_sftp_timeout",
                f"{context_key}_ssh_timeout",
                "sftp_timeout",
                "ssh_timeout",
                default=30,
            )
        )
        banner_timeout = float(
            self._upload_config_value(
                f"{context_key}_sftp_banner_timeout",
                f"{context_key}_ssh_banner_timeout",
                "sftp_banner_timeout",
                "ssh_banner_timeout",
                default=max(30.0, timeout),
            )
        )
        attempts = max(
            1,
            int(
                self._upload_config_value(
                    f"{context_key}_sftp_connect_attempts",
                    f"{context_key}_ssh_connect_attempts",
                    "sftp_connect_attempts",
                    "ssh_connect_attempts",
                    default=2,
                )
            ),
        )
        retry_delay = max(
            0.0,
            float(
                self._upload_config_value(
                    f"{context_key}_sftp_connect_retry_delay",
                    f"{context_key}_ssh_connect_retry_delay",
                    "sftp_connect_retry_delay",
                    "ssh_connect_retry_delay",
                    default=1.0,
                )
            ),
        )

        paramiko_log_channel = f"paramiko.{tag.lower()}.suppressed.{id(self)}"
        self._suppress_paramiko_log_channel(paramiko_log_channel)

        last_exc: Optional[Exception] = None
        connected_attempt = 0
        for attempt in range(1, attempts + 1):
            ssh = SSHClient()
            ssh.set_log_channel(paramiko_log_channel)
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    key_filename=self.ssh_key_path,
                    look_for_keys=True,
                    allow_agent=True,
                    timeout=timeout,
                    banner_timeout=banner_timeout,
                    auth_timeout=timeout,
                )
                connected_attempt = attempt
                break
            except Exception as exc:
                last_exc = exc
                try:
                    ssh.close()
                except Exception:
                    pass
                if attempt >= attempts:
                    self._prefixed_log(
                        tag,
                        f"SFTP connect failed after {attempts} attempt{'s' if attempts != 1 else ''}: {exc}",
                        level="error",
                        color=color,
                        throttle_key=f"{tag}:sftp_connect_failed:{type(exc).__name__}:{exc}",
                        throttle_seconds=min(10.0, self._connect_log_interval(tag)),
                    )
                    raise
                if retry_delay:
                    self._stop_event.wait(retry_delay)
        else:
            assert last_exc is not None
            raise last_exc

        if connected_attempt > 1:
            self._prefixed_log(
                tag,
                f"SFTP connect succeeded after retry ({connected_attempt}/{attempts} attempts)",
                color=color,
                throttle_key=f"{tag}:sftp_connect_retry_succeeded:{connected_attempt}:{attempts}",
                throttle_seconds=min(10.0, self._connect_log_interval(tag)),
            )

        self._prefixed_log(
            tag,
            f"SFTP connected to {self.username}@{self.host}:{self.port}",
            color=color,
            throttle_key=f"{tag}:sftp_connected",
            throttle_seconds=self._connect_log_interval(tag),
        )
        return ssh

    def _ensure_remote_dir(self, sftp: SFTPClient, remote_dir: str) -> None:
        """
        Ensure a remote directory exists, creating it if necessary.

        Args:
            sftp: SFTP client instance
            remote_dir: Remote directory path to ensure exists
        """
        try:
            sftp.stat(remote_dir)
        except IOError:
            self._upload_log(f"Creating remote directory: {remote_dir}")
            sftp.mkdir(remote_dir)

    def _ensure_remote_dirs_recursive(self, sftp: SFTPClient, remote_dir: str) -> None:
        """
        Recursively ensure remote directories exist, creating them if necessary.

        Args:
            sftp: SFTP client instance
            remote_dir: Remote directory path to ensure exists
        """
        parts = Path(remote_dir).parts
        current_path = ""
        for part in parts:
            current_path = os.path.join(current_path, part)
            self._ensure_remote_dir(sftp, current_path)

    def _remote_file_size(self, sftp: SFTPClient, remote_file: str) -> Optional[int]:
        try:
            attr = sftp.stat(remote_file)
            if self._is_remote_dir_attr(attr):
                return None
            return int(attr.st_size or 0)
        except IOError:
            return None

    def _upload_file_via_sftp_from_scratch(
        self,
        sftp: SFTPClient,
        local_file: str,
        remote_file: str,
    ) -> bool:
        local_size = os.path.getsize(local_file)
        self._upload_log(
            f"Uploading file from scratch: {local_file} -> {remote_file} ({self._format_bytes(local_size)})"
        )
        chunk_size = int(
            self._upload_config_value(
                "rollout_upload_chunk_size",
                "sftp_upload_chunk_size",
                default=8 * 1024 * 1024,
            )
        )
        chunk_size = self._upload_limiter.chunk_size(chunk_size)
        transferred = 0
        last_log_time = time.monotonic()
        last_log_bytes = 0
        progress_interval = self._progress_log_interval("UPLOAD")
        progress_bytes = int(
            self._upload_config_value(
                "rollout_upload_progress_bytes",
                "sftp_upload_progress_bytes",
                default=128 * 1024 * 1024,
            )
        )

        with open(local_file, "rb") as src, sftp.open(remote_file, "wb") as dst:
            while True:
                if self._thread is threading.current_thread() and not self._running:
                    raise RuntimeError("Upload interrupted because DataUploadManager is stopping")
                data = src.read(chunk_size)
                if not data:
                    break
                self._upload_limiter.acquire(len(data))
                dst.write(data)
                transferred += len(data)
                now = time.monotonic()
                should_log = (
                    transferred == local_size
                    or transferred - last_log_bytes >= progress_bytes
                    or now - last_log_time >= progress_interval
                )
                if should_log:
                    percent = transferred / local_size * 100.0 if local_size else 100.0
                    self._upload_log(
                        f"File upload progress: {os.path.basename(local_file)} "
                        f"{self._format_bytes(transferred)}/{self._format_bytes(local_size)} ({percent:.1f}%)"
                    )
                    last_log_time = now
                    last_log_bytes = transferred

        final_size = self._remote_file_size(sftp, remote_file)
        if final_size != local_size:
            self._upload_log(
                f"Remote upload size mismatch for {remote_file}: "
                f"{self._format_bytes(final_size)} != {self._format_bytes(local_size)}",
                level="warning",
            )
            return False
        return True

    def _upload_file_via_sftp_resumable(
        self,
        sftp: SFTPClient,
        local_file: str,
        remote_file: str,
    ) -> bool:
        local_size = os.path.getsize(local_file)
        remote_size = self._remote_file_size(sftp, remote_file)

        if remote_size == local_size:
            self._upload_log(
                f"Remote file already complete, skipping: {remote_file} ({self._format_bytes(local_size)})",
                level="debug",
            )
            return True

        if remote_size is not None and remote_size > local_size:
            self._upload_log(
                "Remote file is larger than local; restarting upload: "
                f"{remote_file} ({self._format_bytes(remote_size)} > {self._format_bytes(local_size)})",
                level="warning",
            )
            remote_size = 0

        resume_offset = remote_size or 0
        mode = "ab" if resume_offset else "wb"
        action = "Resuming" if resume_offset else "Uploading"
        self._upload_log(
            f"{action} file via SFTP: {local_file} -> {remote_file} "
            f"({self._format_bytes(resume_offset)}/{self._format_bytes(local_size)})"
        )

        chunk_size = int(
            self._upload_config_value(
                "rollout_upload_chunk_size",
                "sftp_upload_chunk_size",
                default=8 * 1024 * 1024,
            )
        )
        chunk_size = self._upload_limiter.chunk_size(chunk_size)
        transferred = resume_offset
        last_log_time = time.monotonic()
        last_log_bytes = transferred
        progress_interval = self._progress_log_interval("UPLOAD")
        progress_bytes = int(
            self._upload_config_value(
                "rollout_upload_progress_bytes",
                "sftp_upload_progress_bytes",
                default=128 * 1024 * 1024,
            )
        )

        with open(local_file, "rb") as src, sftp.open(remote_file, mode) as dst:
            if resume_offset:
                src.seek(resume_offset)
            while True:
                if self._thread is threading.current_thread() and not self._running:
                    raise RuntimeError("Upload interrupted because DataUploadManager is stopping")
                data = src.read(chunk_size)
                if not data:
                    break
                self._upload_limiter.acquire(len(data))
                dst.write(data)
                transferred += len(data)
                now = time.monotonic()
                should_log = (
                    transferred == local_size
                    or transferred - last_log_bytes >= progress_bytes
                    or now - last_log_time >= progress_interval
                )
                if should_log:
                    percent = transferred / local_size * 100.0 if local_size else 100.0
                    self._upload_log(
                        f"File upload progress: {os.path.basename(local_file)} "
                        f"{self._format_bytes(transferred)}/{self._format_bytes(local_size)} ({percent:.1f}%)"
                    )
                    last_log_time = now
                    last_log_bytes = transferred

        final_size = self._remote_file_size(sftp, remote_file)
        if final_size != local_size:
            self._upload_log(
                f"Remote upload size mismatch for {remote_file}: "
                f"{self._format_bytes(final_size)} != {self._format_bytes(local_size)}",
                level="warning",
            )
            return False
        return True

    def _rollout_upload_method(self) -> str:
        method = self._upload_config_value(
            "rollout_upload_method",
            "upload_method",
            "upload_transfer_method",
            default="rsync",
        )
        method_key = str(method).strip().lower().replace("-", "_")
        aliases = {
            "paramiko": "sftp",
            "python": "sftp",
            "python_sftp": "sftp",
            "rsync_then_sftp": "rsync",
            "rsync_sftp": "rsync",
            "auto": "rsync",
        }
        method_key = aliases.get(method_key, method_key)
        if method_key not in {"rsync", "sftp"}:
            self._upload_log(
                f"Unknown rollout upload method={method!r}; falling back to rsync",
                level="warning",
            )
            method_key = "rsync"
        return method_key

    def _rollout_rsync_fallback_enabled(self) -> bool:
        return bool(
            self._upload_config_value(
                "rollout_rsync_fallback_to_sftp",
                "upload_rsync_fallback_to_sftp",
                default=True,
            )
        )

    def _resume_rollout_upload_enabled(self) -> bool:
        return bool(
            self._upload_config_value(
                "resume_rollout_upload",
                "resume_data_upload",
                "resume_upload",
                default=True,
            )
        )

    def _rollout_rsync_executable(self) -> Optional[str]:
        configured = self._upload_config_value(
            "rollout_rsync_binary",
            "upload_rsync_binary",
            "rsync_binary",
            default="rsync",
        )
        executable = os.path.expanduser(str(configured))
        if os.path.isabs(executable) or os.path.sep in executable:
            return executable if os.path.exists(executable) else None
        return shutil.which(executable)

    def _rollout_rsync_remote_spec(self, remote_folder: str) -> str:
        host = str(self._upload_config_value("rollout_rsync_host", "upload_rsync_host", default=self.host))
        username = self._upload_config_value("rollout_rsync_username", "upload_rsync_username", default=self.username)
        remote_path = str(remote_folder).rstrip("/") + "/"
        login = f"{username}@{host}" if username else host
        return f"{login}:{remote_path}"

    def _rollout_rsync_ssh_command(self) -> str:
        ssh_parts = [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=30",
        ]
        if self.ssh_key_path:
            ssh_parts.extend(["-i", self.ssh_key_path])
        ssh_parts.extend(
            self._coerce_command_args(
                self._upload_config_value(
                    "rollout_rsync_ssh_options",
                    "upload_rsync_ssh_options",
                    "rsync_ssh_options",
                )
            )
        )
        return " ".join(shlex.quote(part) for part in ssh_parts)

    def _build_rollout_rsync_command(self, local_folder: str, remote_folder: str) -> List[str]:
        executable = self._rollout_rsync_executable()
        if not executable:
            raise FileNotFoundError("rsync executable was not found")

        source = str(Path(local_folder).expanduser()) + os.sep
        command = [
            executable,
            "-a",
            "--partial",
            "--append-verify",
            "--info=progress2",
            "-e",
            self._rollout_rsync_ssh_command(),
        ]
        command.extend(
            self._coerce_command_args(
                self._upload_config_value(
                    "rollout_rsync_extra_args",
                    "upload_rsync_extra_args",
                    "rsync_extra_args",
                )
            )
        )
        command.append(f"--bwlimit={int(self.upload_bandwidth_mbps * 1_000_000 / 1024)}")
        command.extend([source, self._rollout_rsync_remote_spec(remote_folder)])
        return command

    @staticmethod
    def _coerce_int_value(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _policy_generation_id_from_info(cls, policy_info: Dict[str, Any]) -> Optional[int]:
        """
        Extract the rollout/checkpoint generation id used for stale-upload checks.

        ``policy_id`` is preferred. If older metadata only recorded ckpt-style
        ids (for example ckpt_000012), use that number as the generation id.
        Avoid falling back to train_step because it is often a different scale.
        """
        if not isinstance(policy_info, dict):
            return None

        for key in ("policy_id", "current_policy_id"):
            value = cls._coerce_int_value(policy_info.get(key))
            if value is not None:
                return value

        for key in ("checkpoint_id", "current_checkpoint_id", "checkpoint_name"):
            raw_value = policy_info.get(key)
            value = cls._coerce_int_value(raw_value)
            if value is not None:
                return value
            inferred = infer_policy_id_from_text(raw_value)
            if inferred is not None:
                return inferred
            inferred = infer_checkpoint_index_from_text(raw_value)
            if inferred is not None:
                return inferred

        checkpoint_index = cls._coerce_int_value(policy_info.get("checkpoint_index"))
        if checkpoint_index is not None and "train_step" not in policy_info:
            return checkpoint_index
        return None

    def _policy_info_from_latest_registry(
        self,
        latest_path: str,
        latest_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        checkpoint_path = latest_info.get("checkpoint_path") or latest_info.get("params_path")
        checkpoint_dir = str(checkpoint_path or "")
        if checkpoint_dir:
            checkpoint_dir = self._checkpoint_dir_from_registry_path(latest_path, checkpoint_dir)
        checkpoint_name = (
            latest_info.get("checkpoint_name")
            or latest_info.get("checkpoint_id")
            or (posixpath.basename(checkpoint_dir.rstrip("/")) if checkpoint_dir else None)
        )
        return normalize_policy_info(
            {
                **latest_info,
                "checkpoint_id": latest_info.get("checkpoint_id") or checkpoint_name,
                "checkpoint_name": checkpoint_name,
                "remote_latest_path": latest_path,
                "remote_checkpoint_path": checkpoint_dir or latest_info.get("checkpoint_path"),
                "policy_source": "remote_latest",
            },
            ckpt_dir=checkpoint_dir or None,
        )

    def _record_latest_remote_policy_generation(
        self,
        policy_info: Dict[str, Any],
        *,
        source: str,
    ) -> Optional[int]:
        generation_id = self._policy_generation_id_from_info(policy_info)
        if generation_id is None:
            self._ckpt_log(
                f"Remote latest policy generation is unknown from {source}: {policy_info}",
                level="debug",
                throttle_key=f"CKPT:remote_generation_unknown:{source}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return None

        previous = self._latest_remote_policy_generation_id
        self._latest_remote_policy_generation_id = generation_id
        self._latest_remote_policy_info = dict(policy_info)
        self._latest_remote_policy_observed_at = time.monotonic()
        if previous != generation_id:
            self._ckpt_log(
                "Observed remote latest policy generation: "
                f"policy_id={generation_id}, checkpoint_id={policy_info.get('checkpoint_id')}, source={source}"
            )
        return generation_id

    # Checkpoint download support ---------------------------------------

    @staticmethod
    def _remote_join(*parts: str) -> str:
        normalized_parts = [str(part).strip("/") for part in parts if str(part)]
        if not normalized_parts:
            return ""
        first = str(parts[0])
        joined = posixpath.join(*normalized_parts)
        return "/" + joined if first.startswith("/") else joined

    @staticmethod
    def _remote_parent(path: str) -> str:
        parent = posixpath.dirname(path.rstrip("/"))
        return parent or "/"

    @staticmethod
    def _is_remote_dir_attr(attr: Any) -> bool:
        return bool(attr.st_mode is not None and stat.S_ISDIR(attr.st_mode))

    def _remote_exists(self, sftp: SFTPClient, remote_path: str) -> bool:
        try:
            sftp.stat(remote_path)
            return True
        except IOError:
            return False

    def _remote_is_dir(self, sftp: SFTPClient, remote_path: str) -> bool:
        try:
            return self._is_remote_dir_attr(sftp.stat(remote_path))
        except IOError:
            return False

    def _read_remote_json_safe(self, sftp: SFTPClient, remote_path: str) -> Dict[str, Any]:
        try:
            with sftp.open(remote_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.debug(f"Failed to read remote JSON {remote_path}: {exc}")
        return {}

    def _remote_checkpoint_base_root(self) -> str:
        return str(
            self.online_training_config.get(
                "remote_checkpoint_root",
                "/path/to/async",
            )
        )

    def _resolve_configured_remote_path(self, path_value: Any) -> Optional[str]:
        if not path_value:
            return None
        path = str(path_value)
        if path.startswith("/"):
            return path
        return self._remote_join(self._remote_checkpoint_base_root(), path)

    def _remote_checkpoint_scan_root(self) -> Optional[str]:
        checkpoint_path = (
            self.online_training_config.get("checkpoint_path")
            or self.online_training_config.get("checkpoint_dir")
            or self.online_training_config.get("remote_checkpoint_path")
            or self.online_training_config.get("remote_checkpoint_dir")
            or self.online_training_config.get("remote_policy_dir")
        )
        return self._resolve_configured_remote_path(checkpoint_path)

    def _remote_latest_checkpoint_path(self) -> Optional[str]:
        latest_path = (
            self.online_training_config.get("remote_latest_path")
            or self.online_training_config.get("remote_latest_checkpoint_path")
            or self.online_training_config.get("checkpoint_latest_path")
            or self.online_training_config.get("current_policy_registry")
        )
        return self._resolve_configured_remote_path(latest_path)

    def _local_latest_checkpoint_path(self) -> Path:
        latest_path = (
            self.online_training_config.get("local_latest_path")
            or self.online_training_config.get("local_latest_checkpoint_path")
            or self.online_training_config.get("checkpoint_local_latest_path")
        )
        if latest_path:
            return Path(latest_path).expanduser()
        return self.local_checkpoint_root / "latest.json"

    @staticmethod
    def _latest_train_step(info: Dict[str, Any]) -> Optional[int]:
        raw_value = info.get("train_step", info.get("checkpoint_index"))
        if raw_value is None:
            return None
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return None

    def _read_local_latest_checkpoint_info(self) -> Dict[str, Any]:
        latest_path = self._local_latest_checkpoint_path()
        if not latest_path.exists():
            return {}
        try:
            return read_json_file(latest_path)
        except Exception as exc:
            self._ckpt_log(f"Failed to read local latest checkpoint registry {latest_path}: {exc}", level="warning")
            return {}

    def _checkpoint_archive_dir(self, checkpoint_ref: Dict[str, Any]) -> Optional[Path]:
        checkpoint_id = checkpoint_ref.get("checkpoint_id")
        checkpoint_name = checkpoint_ref.get("checkpoint_name") or checkpoint_id
        if not checkpoint_name:
            return None
        return self.local_checkpoint_root / "archive" / str(checkpoint_name)

    def _checkpoint_archive_ready(self, checkpoint_ref: Dict[str, Any]) -> bool:
        archive_dir = self._checkpoint_archive_dir(checkpoint_ref)
        if archive_dir is None:
            return False
        return archive_dir.exists() and (archive_dir / "params").exists() and (archive_dir / "complete.txt").exists()

    def _write_local_latest_checkpoint_info(self, checkpoint_ref: Dict[str, Any]) -> None:
        archive_dir = self._checkpoint_archive_dir(checkpoint_ref)
        if archive_dir is None:
            return
        remote_latest = checkpoint_ref.get("remote_latest_info")
        payload = dict(remote_latest) if isinstance(remote_latest, dict) else dict(checkpoint_ref)
        remote_checkpoint_path = checkpoint_ref.get("remote_checkpoint_path")
        remote_params_path = checkpoint_ref.get("remote_params_path")
        payload.update(
            {
                "schema_version": payload.get("schema_version", "checkpoint_registry/v1"),
                "checkpoint_id": checkpoint_ref.get("checkpoint_id") or payload.get("checkpoint_id"),
                "checkpoint_path": str(archive_dir),
                "params_path": str(archive_dir / "params"),
                "local_checkpoint_path": str(archive_dir),
                "local_params_path": str(archive_dir / "params"),
                "remote_checkpoint_path": remote_checkpoint_path,
                "remote_params_path": remote_params_path,
                "downloaded_at": utc_now_iso(),
            }
        )
        if checkpoint_ref.get("checkpoint_name") is not None:
            payload["checkpoint_name"] = checkpoint_ref.get("checkpoint_name")
        atomic_write_json(self._local_latest_checkpoint_path(), payload)
        self._ckpt_log(
            "Updated local latest checkpoint registry: "
            f"{self._local_latest_checkpoint_path()} (train_step={payload.get('train_step')})"
        )

    def _write_observed_latest_checkpoint_info(self, checkpoint_ref: Dict[str, Any]) -> None:
        remote_latest = checkpoint_ref.get("remote_latest_info")
        payload = dict(remote_latest) if isinstance(remote_latest, dict) else dict(checkpoint_ref)
        remote_checkpoint_path = checkpoint_ref.get("remote_checkpoint_path")
        remote_params_path = checkpoint_ref.get("remote_params_path")
        for stale_local_key in ("downloaded_at", "local_checkpoint_path", "local_params_path"):
            payload.pop(stale_local_key, None)
        payload.update(
            {
                "schema_version": payload.get("schema_version", "checkpoint_registry/v1"),
                "checkpoint_id": checkpoint_ref.get("checkpoint_id") or payload.get("checkpoint_id"),
                "checkpoint_path": remote_checkpoint_path,
                "params_path": remote_params_path,
                "remote_checkpoint_path": remote_checkpoint_path,
                "remote_params_path": remote_params_path,
                "download_status": "pending_download",
                "observed_at": utc_now_iso(),
            }
        )
        for key in ("checkpoint_name", "policy_id", "train_step", "sha256"):
            if checkpoint_ref.get(key) is not None:
                payload[key] = checkpoint_ref.get(key)
        atomic_write_json(self._local_latest_checkpoint_path(), payload)
        self._ckpt_log(
            "Recorded observed latest checkpoint before download: "
            f"{self._local_latest_checkpoint_path()} "
            f"(policy_id={payload.get('policy_id')}, train_step={payload.get('train_step')})"
        )

    def _checkpoint_dir_from_registry_path(self, latest_path: str, checkpoint_path: str) -> str:
        checkpoint_path = str(checkpoint_path).rstrip("/")
        if not checkpoint_path.startswith("/"):
            checkpoint_path = self._remote_join(self._remote_parent(latest_path), checkpoint_path)
        if checkpoint_path.endswith("/params"):
            return self._remote_parent(checkpoint_path)
        return checkpoint_path

    def _build_remote_checkpoint_ref_from_latest(
        self,
        sftp: SFTPClient,
        latest_path: str,
        latest_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        checkpoint_path = latest_info.get("checkpoint_path") or latest_info.get("params_path")
        if not checkpoint_path:
            self._ckpt_log(
                f"Remote latest registry has no checkpoint_path: {latest_path}",
                level="warning",
                throttle_key=f"CKPT:latest_missing_path:{latest_path}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return None
        checkpoint_dir = self._checkpoint_dir_from_registry_path(latest_path, str(checkpoint_path))
        checkpoint_name = (
            latest_info.get("checkpoint_name")
            or latest_info.get("checkpoint_id")
            or posixpath.basename(checkpoint_dir.rstrip("/"))
        )
        ref = self._build_remote_checkpoint_ref(
            sftp,
            checkpoint_dir,
            latest_info,
            checkpoint_name=str(checkpoint_name),
            remote_checkpoint_root=self._remote_parent(checkpoint_dir),
        )
        if ref is None:
            return None
        ref["remote_latest_path"] = latest_path
        ref["remote_latest_info"] = dict(latest_info)
        return ref

    def _online_remote_model_root(self) -> Optional[str]:
        remote_root = self._remote_checkpoint_base_root()
        model_checkpoint_dir = (
            self.online_training_config.get("model_checkpoint_dir")
            or self.online_training_config.get("checkpoint_model_dir")
            or self.online_training_config.get("remote_model_checkpoint_dir")
        )
        if not model_checkpoint_dir:
            return None
        if str(model_checkpoint_dir).startswith("/"):
            return str(model_checkpoint_dir)
        return self._remote_join(str(remote_root), str(model_checkpoint_dir))

    def _resolve_remote_checkpoint(self, sftp: SFTPClient) -> Optional[Dict[str, Any]]:
        model_root = self._online_remote_model_root()
        explicit_latest_path = self._remote_latest_checkpoint_path()
        if explicit_latest_path and self._remote_exists(sftp, explicit_latest_path):
            latest = self._read_remote_json_safe(sftp, explicit_latest_path)
            if latest:
                return self._build_remote_checkpoint_ref_from_latest(sftp, explicit_latest_path, latest)
        if not model_root:
            logger.debug("Online checkpoint polling is enabled but model_checkpoint_dir is not configured")
            return None

        checkpoint_name = (
            self.online_training_config.get("checkpoint_name")
            or self.online_training_config.get("ckpt_name")
            or "latest.json"
        )
        checkpoint_name = str(checkpoint_name)

        registry_candidates = []
        if checkpoint_name in ("latest", "latest.json"):
            registry_candidates.append(self._remote_join(model_root, "checkpoints", "latest.json"))
            registry_candidates.append(self._remote_join(model_root, "latest.json"))
        elif checkpoint_name.endswith(".json"):
            registry_candidates.append(
                checkpoint_name if checkpoint_name.startswith("/") else self._remote_join(model_root, checkpoint_name)
            )

        for latest_path in registry_candidates:
            if not self._remote_exists(sftp, latest_path):
                continue
            latest = self._read_remote_json_safe(sftp, latest_path)
            checkpoint_ref = self._build_remote_checkpoint_ref_from_latest(sftp, latest_path, latest)
            if checkpoint_ref is not None:
                return checkpoint_ref

        candidates = []
        if checkpoint_name.startswith("/"):
            candidates.append(checkpoint_name)
        else:
            candidates.extend(
                [
                    self._remote_join(model_root, "checkpoints", checkpoint_name),
                    self._remote_join(model_root, checkpoint_name),
                ]
            )

        for checkpoint_dir in candidates:
            if checkpoint_dir.endswith("/params"):
                checkpoint_dir = self._remote_parent(checkpoint_dir)
            if self._remote_is_dir(sftp, checkpoint_dir):
                manifest = self._read_remote_json_safe(sftp, self._remote_join(checkpoint_dir, "manifest.json"))
                return self._build_remote_checkpoint_ref(sftp, checkpoint_dir, manifest)
        return None

    def _list_remote_checkpoint_refs(self, sftp: SFTPClient) -> Dict[str, Dict[str, Any]]:
        """
        Return complete remote checkpoints keyed by their direct child name.

        In the online split layout, ``checkpoint_path`` points at a registry
        directory whose children are policy/checkpoint directories. We compare
        these child names with local ``archive/<name>`` and download any local
        miss. If ``checkpoint_path`` itself is a checkpoint directory, this
        method returns a single entry for backward compatibility.
        """
        checkpoint_root = self._remote_checkpoint_scan_root()
        if not checkpoint_root:
            checkpoint_ref = self._resolve_remote_checkpoint(sftp)
            if checkpoint_ref is None:
                self._ckpt_log(
                    "No remote checkpoint registry or latest checkpoint resolved",
                    throttle_key="CKPT:no_registry",
                    throttle_seconds=self._status_log_interval("CKPT"),
                )
                return {}
            name = str(checkpoint_ref.get("checkpoint_name") or checkpoint_ref.get("checkpoint_id"))
            self._ckpt_log(f"Resolved single remote checkpoint: {name}")
            return {name: checkpoint_ref}

        if checkpoint_root.endswith("/params"):
            checkpoint_root = self._remote_parent(checkpoint_root)
        self._ckpt_log(
            f"Scanning remote checkpoint root: {checkpoint_root}",
            throttle_key=f"CKPT:scan:{checkpoint_root}",
            throttle_seconds=self._status_log_interval("CKPT"),
        )
        if not self._remote_is_dir(sftp, checkpoint_root):
            self._ckpt_log(
                f"Remote checkpoint root is not ready: {checkpoint_root}",
                throttle_key=f"CKPT:not_ready:{checkpoint_root}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return {}

        manifest = self._read_remote_json_safe(sftp, self._remote_join(checkpoint_root, "manifest.json"))
        if self._remote_is_dir(sftp, self._remote_join(checkpoint_root, "params")):
            name = posixpath.basename(checkpoint_root.rstrip("/"))
            ref = self._build_remote_checkpoint_ref(
                sftp,
                checkpoint_root,
                manifest,
                checkpoint_name=name,
                remote_checkpoint_root=self._remote_parent(checkpoint_root),
            )
            self._ckpt_log(f"checkpoint_path points to a checkpoint directory: name={name}, ready={ref is not None}")
            return {name: ref} if ref is not None else {}

        refs: Dict[str, Dict[str, Any]] = {}
        for entry in sftp.listdir_attr(checkpoint_root):
            if not self._is_remote_dir_attr(entry):
                continue
            if entry.filename.startswith(".") or entry.filename in {"current", "downloading", "registries"}:
                continue
            checkpoint_dir = self._remote_join(checkpoint_root, entry.filename)
            manifest = self._read_remote_json_safe(sftp, self._remote_join(checkpoint_dir, "manifest.json"))
            ref = self._build_remote_checkpoint_ref(
                sftp,
                checkpoint_dir,
                manifest,
                checkpoint_name=entry.filename,
                remote_checkpoint_root=checkpoint_root,
                remote_mtime=getattr(entry, "st_mtime", None),
            )
            if ref is not None:
                refs[entry.filename] = ref
        self._ckpt_log(
            f"Complete remote checkpoint candidates: {sorted(refs.keys(), key=str)}",
            throttle_key=f"CKPT:candidates:{checkpoint_root}:{','.join(sorted(refs.keys(), key=str))}",
            throttle_seconds=self._status_log_interval("CKPT"),
        )
        return refs

    @staticmethod
    def _checkpoint_sort_key(item: tuple[str, Dict[str, Any]]) -> tuple:
        name, ref = item
        policy_id = ref.get("policy_id")
        checkpoint_index = ref.get("checkpoint_index") or ref.get("train_step")
        try:
            policy_key = int(policy_id)
        except (TypeError, ValueError):
            inferred_policy_id = infer_policy_id_from_text(name)
            policy_key = inferred_policy_id if inferred_policy_id is not None else -1
        try:
            checkpoint_key = int(checkpoint_index)
        except (TypeError, ValueError):
            checkpoint_key = -1
        return (policy_key, checkpoint_key, float(ref.get("remote_mtime") or 0.0), name)

    def _local_complete_checkpoint_refs(self) -> Dict[str, Dict[str, Any]]:
        refs: Dict[str, Dict[str, Any]] = {}
        archive_root = self.local_checkpoint_root / "archive"
        if not archive_root.exists():
            return refs
        for checkpoint_dir in archive_root.iterdir():
            if not checkpoint_dir.is_dir() or checkpoint_dir.name.endswith(".partial"):
                continue
            if not (checkpoint_dir / "params").exists() or not (checkpoint_dir / "complete.txt").exists():
                continue
            manifest = {}
            manifest_path = checkpoint_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with manifest_path.open() as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        manifest = loaded
                except Exception as exc:
                    logger.debug(f"Failed to read local checkpoint manifest {manifest_path}: {exc}")
            refs[checkpoint_dir.name] = normalize_policy_info(
                {
                    **manifest,
                    "checkpoint_name": checkpoint_dir.name,
                    "checkpoint_dir": str(checkpoint_dir),
                    "params_dir": str(checkpoint_dir / "params"),
                },
                ckpt_dir=checkpoint_dir,
                default_policy_id=infer_policy_id_from_text(checkpoint_dir.name),
            )
        return refs

    def _newest_local_checkpoint_ref(self) -> Optional[Dict[str, Any]]:
        local_refs = self._local_complete_checkpoint_refs()
        if not local_refs:
            return None
        return sorted(local_refs.items(), key=self._checkpoint_sort_key)[-1][1]

    def _build_remote_checkpoint_ref(
        self,
        sftp: SFTPClient,
        checkpoint_dir: str,
        metadata: Dict[str, Any],
        *,
        checkpoint_name: Optional[str] = None,
        remote_checkpoint_root: Optional[str] = None,
        remote_mtime: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        params_dir = self._remote_join(checkpoint_dir, "params")
        if not self._remote_is_dir(sftp, params_dir):
            self._ckpt_log(
                f"Skipping remote checkpoint without params/: {checkpoint_dir}",
                throttle_key=f"CKPT:skip_no_params:{checkpoint_dir}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return None

        marker_names = self.online_training_config.get("complete_marker_names") or ["DONE", "complete.txt"]
        if isinstance(marker_names, str):
            marker_names = [marker_names]
        require_marker = self.online_training_config.get("require_complete_marker", True)
        has_marker = any(
            self._remote_exists(sftp, self._remote_join(checkpoint_dir, marker)) for marker in marker_names
        )
        if require_marker and not has_marker:
            self._ckpt_log(
                f"Skipping remote checkpoint without complete marker: {checkpoint_dir}, markers={marker_names}",
                throttle_key=f"CKPT:skip_no_marker:{checkpoint_dir}:{','.join(str(marker) for marker in marker_names)}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return None

        checkpoint_name = checkpoint_name or posixpath.basename(checkpoint_dir.rstrip("/"))
        checkpoint_id = metadata.get("checkpoint_id") or checkpoint_name
        policy_id = metadata.get("policy_id")
        if policy_id is None:
            policy_id = infer_policy_id_from_text(checkpoint_id)
        if policy_id is None:
            policy_id = infer_policy_id_from_text(checkpoint_name)

        ref = normalize_policy_info(
            {
                **metadata,
                "checkpoint_id": checkpoint_id,
                "policy_id": policy_id,
                "checkpoint_name": checkpoint_name,
                "remote_checkpoint_root": remote_checkpoint_root,
                "remote_checkpoint_path": checkpoint_dir,
                "remote_params_path": params_dir,
                "remote_mtime": remote_mtime,
                "policy_source": "remote",
            },
            ckpt_dir=checkpoint_dir,
        )
        ref["checkpoint_name"] = checkpoint_name
        if remote_checkpoint_root:
            ref["remote_checkpoint_root"] = remote_checkpoint_root
        ref["remote_checkpoint_path"] = checkpoint_dir
        ref["remote_params_path"] = params_dir
        if remote_mtime is not None:
            ref["remote_mtime"] = remote_mtime
        return ref

    @staticmethod
    def _format_bytes(size: Optional[int]) -> str:
        if size is None:
            return "unknown"
        value = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024.0 or unit == "TiB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024.0
        return f"{value:.1f} TiB"

    def _remote_dir_stats(self, sftp: SFTPClient, remote_dir: str) -> tuple[int, int]:
        file_count = 0
        total_bytes = 0
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = self._remote_join(remote_dir, entry.filename)
            if self._is_remote_dir_attr(entry):
                child_count, child_bytes = self._remote_dir_stats(sftp, remote_path)
                file_count += child_count
                total_bytes += child_bytes
            else:
                file_count += 1
                total_bytes += int(entry.st_size or 0)
        return file_count, total_bytes

    def _local_dir_size(self, local_dir: Path) -> int:
        if not local_dir.exists():
            return 0
        total = 0
        for path in local_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def _resume_checkpoint_download_enabled(self) -> bool:
        return bool(self.online_training_config.get("resume_checkpoint_download", True))

    def _checkpoint_download_progress_enabled(self) -> bool:
        return bool(self.online_training_config.get("checkpoint_download_progress", True))

    @staticmethod
    def _coerce_command_args(value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return shlex.split(value)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return [str(value)]

    def _checkpoint_download_method(self) -> str:
        method = (
            self.online_training_config.get("checkpoint_download_method")
            or self.online_training_config.get("checkpoint_transfer_method")
            or self.online_training_config.get("checkpoint_downloader")
            or "rsync"
        )
        method_key = str(method).strip().lower().replace("-", "_")
        aliases = {
            "paramiko": "sftp",
            "python": "sftp",
            "python_sftp": "sftp",
            "rsync_then_sftp": "rsync",
            "rsync_sftp": "rsync",
            "auto": "rsync",
        }
        method_key = aliases.get(method_key, method_key)
        if method_key not in {"rsync", "sftp"}:
            self._ckpt_log(
                f"Unknown checkpoint_download_method={method!r}; falling back to rsync",
                level="warning",
            )
            method_key = "rsync"
        return method_key

    def _checkpoint_rsync_fallback_enabled(self) -> bool:
        return bool(self.online_training_config.get("checkpoint_rsync_fallback_to_sftp", True))

    def _rsync_executable(self) -> Optional[str]:
        configured = (
            self.online_training_config.get("checkpoint_rsync_binary")
            or self.online_training_config.get("rsync_binary")
            or "rsync"
        )
        executable = os.path.expanduser(str(configured))
        if os.path.isabs(executable) or os.path.sep in executable:
            return executable if os.path.exists(executable) else None
        return shutil.which(executable)

    def _rsync_remote_spec(self, remote_dir: str) -> str:
        host = str(self.online_training_config.get("checkpoint_rsync_host") or self.host)
        username = self.online_training_config.get("checkpoint_rsync_username", self.username)
        remote_path = str(remote_dir).rstrip("/") + "/"
        login = f"{username}@{host}" if username else host
        return f"{login}:{remote_path}"

    def _rsync_ssh_command(self) -> str:
        ssh_parts = [
            "ssh",
            "-p",
            str(self.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=30",
        ]
        if self.ssh_key_path:
            ssh_parts.extend(["-i", self.ssh_key_path])
        ssh_parts.extend(
            self._coerce_command_args(
                self.online_training_config.get("checkpoint_rsync_ssh_options")
                or self.online_training_config.get("rsync_ssh_options")
            )
        )
        return " ".join(shlex.quote(part) for part in ssh_parts)

    def _build_rsync_command(self, remote_dir: str, local_dir: Path) -> List[str]:
        executable = self._rsync_executable()
        if not executable:
            raise FileNotFoundError("rsync executable was not found")

        local_dir = Path(local_dir).expanduser()
        local_dir.mkdir(parents=True, exist_ok=True)
        destination = str(local_dir) + os.sep
        command = [
            executable,
            "-a",
            "--partial",
            "--append-verify",
            "--info=progress2",
            "-e",
            self._rsync_ssh_command(),
        ]
        if bool(self.online_training_config.get("checkpoint_rsync_delete", False)):
            command.append("--delete")
        command.extend(
            self._coerce_command_args(
                self.online_training_config.get("checkpoint_rsync_extra_args")
                or self.online_training_config.get("rsync_extra_args")
            )
        )
        command.append(f"--bwlimit={int(self.download_bandwidth_mbps * 1_000_000 / 1024)}")
        command.extend([self._rsync_remote_spec(remote_dir), destination])
        return command

    def _run_external_transfer(self, command: List[str], context: str) -> None:
        display_command = " ".join(shlex.quote(part) for part in command)
        if str(context).lower().startswith("rollout") or str(context).lower().startswith("upload"):
            self._upload_log(f"Running {context}: {display_command}")
        else:
            self._ckpt_log(f"Running {context}: {display_command}")
        process = subprocess.Popen(command)
        while True:
            return_code = process.poll()
            if return_code is not None:
                break
            current_thread = threading.current_thread()
            managed_thread = current_thread in {self._checkpoint_thread, self._thread}
            if managed_thread and not self._running:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10.0)
                raise RuntimeError(f"{context} interrupted because DataUploadManager is stopping")
            time.sleep(1.0)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    def _promote_legacy_partial_files_for_rsync(self, local_dir: Path) -> None:
        if not local_dir.exists():
            return
        promoted = 0
        removed = 0
        for partial_path in sorted(local_dir.rglob("*.partial")):
            if not partial_path.is_file():
                continue
            target_name = partial_path.name[: -len(".partial")]
            if not target_name:
                continue
            target_path = partial_path.with_name(target_name)
            try:
                if target_path.exists() or target_path.is_symlink():
                    partial_size = partial_path.stat().st_size
                    target_size = target_path.stat().st_size if target_path.is_file() else -1
                    if partial_size > target_size:
                        if target_path.is_dir() and not target_path.is_symlink():
                            shutil.rmtree(target_path)
                        else:
                            target_path.unlink()
                        os.replace(partial_path, target_path)
                        promoted += 1
                    else:
                        partial_path.unlink()
                        removed += 1
                else:
                    os.replace(partial_path, target_path)
                    promoted += 1
            except Exception as exc:
                self._ckpt_log(f"Failed to prepare rsync resume file {partial_path}: {exc}", level="warning")
        if promoted or removed:
            self._ckpt_log(
                f"Prepared rsync resume files under {local_dir}: promoted={promoted}, removed={removed}",
                level="debug",
            )

    def _cleanup_legacy_partial_files(self, local_dir: Path) -> None:
        if not local_dir.exists():
            return
        removed = 0
        for partial_path in sorted(local_dir.rglob("*.partial")):
            try:
                if partial_path.is_file() or partial_path.is_symlink():
                    partial_path.unlink()
                    removed += 1
            except Exception as exc:
                self._ckpt_log(f"Failed to remove stale partial file {partial_path}: {exc}", level="debug")
        if removed:
            self._ckpt_log(f"Removed {removed} stale checkpoint partial files under {local_dir}", level="debug")

    def _download_remote_dir_via_rsync(self, remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        self._promote_legacy_partial_files_for_rsync(local_dir)
        command = self._build_rsync_command(remote_dir, local_dir)
        self._ckpt_log(
            "Downloading checkpoint with rsync "
            f"({remote_dir.rstrip('/')}/ -> {local_dir}/); rsync will resume existing files"
        )
        self._run_external_transfer(command, "checkpoint rsync download")
        self._cleanup_legacy_partial_files(local_dir)

    def _download_remote_file(
        self,
        sftp: SFTPClient,
        remote_path: str,
        local_path: Path,
        total_size: Optional[int],
        progress_bar: Any = None,
    ) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = local_path.with_name(f"{local_path.name}.partial")
        resume_enabled = self._resume_checkpoint_download_enabled()

        if local_path.exists() or local_path.is_symlink():
            local_size = local_path.stat().st_size if local_path.is_file() else None
            if total_size is not None and local_size == total_size:
                self._ckpt_log(f"File already complete, skipping: {local_path}", level="debug")
                return
            if resume_enabled and local_path.is_file():
                if partial_path.exists() or partial_path.is_symlink():
                    partial_size = partial_path.stat().st_size if partial_path.is_file() else -1
                    if local_size is not None and local_size > partial_size:
                        os.replace(local_path, partial_path)
                    else:
                        local_path.unlink()
                else:
                    os.replace(local_path, partial_path)
            elif local_path.is_dir() and not local_path.is_symlink():
                shutil.rmtree(local_path)
            else:
                local_path.unlink()

        if not resume_enabled and (partial_path.exists() or partial_path.is_symlink()):
            partial_path.unlink()

        resume_offset = 0
        if partial_path.exists():
            resume_offset = partial_path.stat().st_size
            if total_size is not None and resume_offset > total_size:
                self._ckpt_log(
                    "Discarding oversized checkpoint partial file: "
                    f"{partial_path} ({self._format_bytes(resume_offset)} > {self._format_bytes(total_size)})",
                    level="warning",
                )
                partial_path.unlink()
                resume_offset = 0
            elif total_size is not None and resume_offset == total_size:
                os.replace(partial_path, local_path)
                self._ckpt_log(f"File partial was already complete: {local_path}", level="debug")
                return

        self._ckpt_log(
            ("Resuming online checkpoint file: " if resume_offset else "Downloading online checkpoint file: ")
            + f"{remote_path} -> {local_path} "
            + f"({self._format_bytes(resume_offset)}/{self._format_bytes(total_size)})"
        )
        last_log_time = time.monotonic()
        last_log_bytes = resume_offset

        def log_progress(transferred: int) -> None:
            nonlocal last_log_time, last_log_bytes
            now = time.monotonic()
            total = total_size or 0
            should_log = (
                transferred == total or transferred - last_log_bytes >= 512 * 1024 * 1024 or now - last_log_time >= 30.0
            )
            if not should_log:
                return
            if total:
                percent = transferred / total * 100.0
                self._ckpt_log(
                    f"File progress: {local_path.name} "
                    f"{self._format_bytes(transferred)}/{self._format_bytes(total)} ({percent:.1f}%)"
                )
            else:
                self._ckpt_log(f"File progress: {local_path.name} {self._format_bytes(transferred)}")
            last_log_time = now
            last_log_bytes = transferred

        try:
            chunk_size = int(self.online_training_config.get("checkpoint_download_chunk_size", 8 * 1024 * 1024))
            chunk_size = self._download_limiter.chunk_size(chunk_size)
            transferred = resume_offset
            mode = "ab" if resume_offset else "wb"
            with sftp.open(remote_path, "rb") as remote_file, partial_path.open(mode) as local_file:
                if resume_offset:
                    remote_file.seek(resume_offset)
                while True:
                    if self._checkpoint_thread is threading.current_thread() and not self._running:
                        raise RuntimeError("Checkpoint download interrupted because DataUploadManager is stopping")
                    read_size = chunk_size
                    if total_size is not None:
                        # Read one byte at EOF to detect a growing remote file.
                        read_size = min(read_size, max(1, total_size - transferred))
                    # Pace requests before receiving data; do not enable prefetch.
                    self._download_limiter.acquire(read_size)
                    data = remote_file.read(read_size)
                    if not data:
                        break
                    local_file.write(data)
                    transferred += len(data)
                    if progress_bar is not None:
                        progress_bar.update(len(data))
                    log_progress(transferred)
            if total_size is not None and partial_path.stat().st_size != total_size:
                raise IOError(
                    f"Downloaded size mismatch for {remote_path}: {partial_path.stat().st_size} != {total_size}"
                )
            os.replace(partial_path, local_path)
        except Exception:
            self._ckpt_log(f"File download left partial file: {partial_path}", level="warning")
            raise

    def _download_remote_dir(
        self,
        sftp: SFTPClient,
        remote_dir: str,
        local_dir: Path,
        progress_bar: Any = None,
    ) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir):
            remote_path = self._remote_join(remote_dir, entry.filename)
            local_path = local_dir / entry.filename
            if self._is_remote_dir_attr(entry):
                self._download_remote_dir(sftp, remote_path, local_path, progress_bar=progress_bar)
            else:
                self._download_remote_file(
                    sftp,
                    remote_path,
                    local_path,
                    getattr(entry, "st_size", None),
                    progress_bar=progress_bar,
                )

    def _default_local_companion_path(self, remote_path: str) -> Path:
        remote_path = str(remote_path).rstrip("/")
        remote_root = self._remote_checkpoint_base_root().rstrip("/")
        if remote_path.startswith(remote_root + "/"):
            rel_path = remote_path[len(remote_root) + 1 :]
        else:
            rel_path = posixpath.basename(remote_path)
        return self.policy_directory / Path(rel_path)

    def _configured_checkpoint_companions(self) -> List[Dict[str, Any]]:
        companions: List[Dict[str, Any]] = []

        def add_companion(
            remote_path: Any,
            *,
            local_path: Any = None,
            policy_info_key: Optional[str] = None,
            remote_info_key: Optional[str] = None,
            overwrite: Optional[bool] = None,
        ) -> None:
            resolved_remote = self._resolve_configured_remote_path(remote_path)
            if not resolved_remote:
                return
            resolved_local = (
                Path(local_path).expanduser() if local_path else self._default_local_companion_path(resolved_remote)
            )
            companions.append(
                {
                    "remote_path": resolved_remote,
                    "local_path": resolved_local,
                    "policy_info_key": policy_info_key,
                    "remote_info_key": remote_info_key,
                    "overwrite": bool(
                        self.online_training_config.get("overwrite_checkpoint_companions", False)
                        if overwrite is None
                        else overwrite
                    ),
                }
            )

        add_companion(
            self.online_training_config.get("remote_assets_path"),
            local_path=self.online_training_config.get("local_assets_path"),
            policy_info_key="assets_dir",
            remote_info_key="remote_assets_path",
        )

        raw_companions = (
            self.online_training_config.get("checkpoint_companion_paths")
            or self.online_training_config.get("extra_checkpoint_paths")
            or []
        )
        if isinstance(raw_companions, dict):
            raw_companions = [raw_companions]
        if isinstance(raw_companions, str):
            raw_companions = [raw_companions]
        for item in raw_companions:
            if isinstance(item, str):
                add_companion(item)
                continue
            if not isinstance(item, dict):
                self._ckpt_log(f"Ignoring invalid checkpoint companion path config: {item!r}", level="warning")
                continue
            add_companion(
                item.get("remote_path") or item.get("source") or item.get("path"),
                local_path=item.get("local_path") or item.get("target"),
                policy_info_key=item.get("policy_info_key") or item.get("info_key"),
                remote_info_key=item.get("remote_info_key"),
                overwrite=item.get("overwrite"),
            )
        return companions

    def _local_companion_ready(self, local_path: Path) -> bool:
        if not local_path.exists():
            return False
        if local_path.is_dir():
            try:
                return any(local_path.iterdir())
            except OSError:
                return False
        return True

    def _download_remote_path_if_needed(
        self,
        sftp: SFTPClient,
        remote_path: str,
        local_path: Path,
        *,
        overwrite: bool = False,
    ) -> bool:
        attr = sftp.stat(remote_path)
        is_dir = self._is_remote_dir_attr(attr)
        if self._local_companion_ready(local_path) and not overwrite:
            return False

        if local_path.exists() or local_path.is_symlink():
            if local_path.is_dir() and not local_path.is_symlink():
                shutil.rmtree(local_path)
            else:
                local_path.unlink()

        self._ckpt_log(f"Downloading checkpoint companion: {remote_path} -> {local_path}")
        if is_dir:
            self._download_remote_dir(sftp, remote_path, local_path)
        else:
            self._download_remote_file(sftp, remote_path, local_path, getattr(attr, "st_size", None))
        return True

    def _ensure_checkpoint_companions(
        self,
        sftp: SFTPClient,
        checkpoint_ref: Dict[str, Any],
    ) -> Dict[str, Any]:
        companion_info: Dict[str, Any] = {}
        for companion in self._configured_checkpoint_companions():
            remote_path = str(companion["remote_path"])
            local_path = Path(companion["local_path"]).expanduser()
            try:
                if not self._remote_exists(sftp, remote_path):
                    self._ckpt_log(f"Checkpoint companion does not exist remotely: {remote_path}", level="warning")
                    continue
                self._download_remote_path_if_needed(
                    sftp,
                    remote_path,
                    local_path,
                    overwrite=bool(companion.get("overwrite", False)),
                )
                if not self._local_companion_ready(local_path):
                    self._ckpt_log(f"Checkpoint companion was not downloaded: {local_path}", level="warning")
                    continue
                policy_info_key = companion.get("policy_info_key")
                remote_info_key = companion.get("remote_info_key")
                if policy_info_key:
                    companion_info[str(policy_info_key)] = str(local_path)
                if remote_info_key:
                    companion_info[str(remote_info_key)] = remote_path
            except Exception as exc:
                self._ckpt_log(
                    "Failed to download checkpoint companion "
                    f"{remote_path} for {checkpoint_ref.get('checkpoint_name')}: {exc}",
                    level="warning",
                )
        return companion_info

    def _update_current_symlink(self, checkpoint_dir: Path) -> None:
        if not bool(self.online_training_config.get("update_current_symlink", True)):
            return
        current_link = self.local_checkpoint_root / "current"
        tmp_link = self.local_checkpoint_root / ".current.tmp"
        try:
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
            tmp_link.symlink_to(checkpoint_dir)
            os.replace(tmp_link, current_link)
        except Exception as exc:
            self._ckpt_log(f"Failed to update current checkpoint symlink {current_link}: {exc}", level="debug")

    def _download_checkpoint_if_needed(self, sftp: SFTPClient, checkpoint_ref: Dict[str, Any]) -> bool:
        checkpoint_id = checkpoint_ref.get("checkpoint_id")
        checkpoint_name = checkpoint_ref.get("checkpoint_name") or checkpoint_id
        remote_checkpoint_path = checkpoint_ref.get("remote_checkpoint_path")
        if not checkpoint_id or not checkpoint_name or not remote_checkpoint_path:
            return False
        checkpoint_id = str(checkpoint_id)
        checkpoint_name = str(checkpoint_name)

        archive_dir = self.local_checkpoint_root / "archive" / checkpoint_name
        complete_marker = archive_dir / "complete.txt"
        if archive_dir.exists() and (archive_dir / "params").exists() and complete_marker.exists():
            self._ckpt_log(
                "Checkpoint already local; refreshing registry/companions: "
                f"name={checkpoint_name}, checkpoint_id={checkpoint_id}, local_dir={archive_dir}",
                throttle_key=f"CKPT:already_local:{checkpoint_name}:{checkpoint_id}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            companion_info = self._ensure_checkpoint_companions(sftp, checkpoint_ref)
            local_policy_info = normalize_policy_info(
                {
                    **checkpoint_ref,
                    **companion_info,
                    "checkpoint_dir": str(archive_dir),
                    "params_dir": str(archive_dir / "params"),
                    "checkpoint_name": checkpoint_name,
                    "policy_source": "downloaded",
                }
            )
            write_current_policy_info(self.current_policy_path, local_policy_info)
            self._update_current_symlink(archive_dir)
            self._downloaded_checkpoints.add(checkpoint_name)
            return False

        downloading_dir = self.local_checkpoint_root / "downloading"
        tmp_dir = downloading_dir / f"{checkpoint_name}.partial"
        resume_download = self._resume_checkpoint_download_enabled()
        if tmp_dir.exists() and tmp_dir.is_dir():
            if resume_download:
                self._ckpt_log(f"Resuming partial checkpoint download: {tmp_dir}")
            else:
                shutil.rmtree(tmp_dir)
        elif tmp_dir.exists():
            tmp_dir.unlink()
        tmp_dir.mkdir(parents=True, exist_ok=True)

        self._ckpt_log(
            f"Downloading checkpoint {checkpoint_name} ({checkpoint_id}): {remote_checkpoint_path} -> {tmp_dir}"
        )
        download_method = self._checkpoint_download_method()
        total_bytes: Optional[int] = None
        stat_before_download = bool(self.online_training_config.get("checkpoint_stat_before_download", False))
        if download_method == "sftp" or stat_before_download:
            try:
                file_count, total_bytes = self._remote_dir_stats(sftp, str(remote_checkpoint_path))
                self._ckpt_log(
                    f"Download size: name={checkpoint_name}, files={file_count}, total={self._format_bytes(total_bytes)}"
                )
            except Exception as exc:
                self._ckpt_log(
                    f"Failed to stat remote checkpoint size for {remote_checkpoint_path}: {exc}", level="debug"
                )
        elif download_method == "rsync":
            self._ckpt_log(
                "Skipping recursive SFTP size scan because checkpoint_download_method=rsync; "
                "rsync --info=progress2 will report transfer progress",
                level="debug",
            )
        progress_bar = None
        try:
            if download_method == "rsync":
                try:
                    self._download_remote_dir_via_rsync(str(remote_checkpoint_path), tmp_dir)
                except Exception as exc:
                    if self._checkpoint_thread is threading.current_thread() and not self._running:
                        raise
                    if not self._checkpoint_rsync_fallback_enabled():
                        raise
                    self._ckpt_log(
                        f"rsync checkpoint download failed; falling back to Python SFTP: {exc}",
                        level="warning",
                    )
                    download_method = "sftp"

            if download_method == "sftp":
                if HAS_TQDM and self._checkpoint_download_progress_enabled() and total_bytes:
                    local_bytes = min(self._local_dir_size(tmp_dir), total_bytes)
                    progress_bar = tqdm(
                        total=total_bytes,
                        initial=local_bytes,
                        desc=f"Downloading ckpt {checkpoint_name}",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                    )
                self._download_remote_dir(sftp, str(remote_checkpoint_path), tmp_dir, progress_bar=progress_bar)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if not (tmp_dir / "params").exists():
            self._ckpt_log(f"Downloaded checkpoint is missing params directory: {tmp_dir}", level="warning")
            return False

        companion_info = self._ensure_checkpoint_companions(sftp, checkpoint_ref)
        tmp_complete_marker = tmp_dir / "complete.txt"
        tmp_complete_marker.write_text("downloaded\n")

        archive_dir.parent.mkdir(parents=True, exist_ok=True)
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        os.replace(tmp_dir, archive_dir)

        local_policy_info = normalize_policy_info(
            {
                **checkpoint_ref,
                **companion_info,
                "checkpoint_dir": str(archive_dir),
                "params_dir": str(archive_dir / "params"),
                "checkpoint_name": checkpoint_name,
                "policy_source": "downloaded",
            }
        )
        write_current_policy_info(self.current_policy_path, local_policy_info)
        self._update_current_symlink(archive_dir)
        self._downloaded_checkpoints.add(checkpoint_name)
        self._ckpt_log(f"Downloaded checkpoint {checkpoint_name} ({checkpoint_id}) to {archive_dir}")
        return True

    def _poll_latest_checkpoint_registry(self, sftp: SFTPClient, latest_path: str) -> None:
        local_latest_path = self._local_latest_checkpoint_path()
        self._ckpt_log(
            f"Poll start; remote_latest_path={latest_path}, local_latest_path={local_latest_path}",
            throttle_key="CKPT:latest_poll_start",
            throttle_seconds=self._status_log_interval("CKPT"),
        )

        if not self._remote_exists(sftp, latest_path):
            self._ckpt_log(
                f"Remote latest checkpoint registry is not ready: {latest_path}",
                throttle_key=f"CKPT:latest_missing:{latest_path}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return

        remote_latest = self._read_remote_json_safe(sftp, latest_path)
        if not remote_latest:
            self._ckpt_log(
                f"Remote latest checkpoint registry is empty or invalid: {latest_path}",
                level="warning",
                throttle_key=f"CKPT:latest_invalid:{latest_path}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return
        self._record_latest_remote_policy_generation(
            self._policy_info_from_latest_registry(latest_path, remote_latest),
            source=latest_path,
        )

        remote_train_step = self._latest_train_step(remote_latest)
        if remote_train_step is None:
            self._ckpt_log(
                f"Remote latest checkpoint registry has no valid train_step: {latest_path}",
                level="warning",
                throttle_key=f"CKPT:latest_no_train_step:{latest_path}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return

        local_latest = self._read_local_latest_checkpoint_info()
        local_train_step = self._latest_train_step(local_latest)
        checkpoint_ref = self._build_remote_checkpoint_ref_from_latest(sftp, latest_path, remote_latest)
        if checkpoint_ref is None:
            self._ckpt_log(
                "Latest checkpoint is not ready yet; waiting for params/ and complete marker",
                throttle_key=f"CKPT:latest_checkpoint_not_ready:{latest_path}:{remote_train_step}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            return

        self._write_observed_latest_checkpoint_info(checkpoint_ref)
        archive_ready = self._checkpoint_archive_ready(checkpoint_ref)
        if local_train_step is not None and remote_train_step <= local_train_step and archive_ready:
            self._ckpt_log(
                "No newer checkpoint in latest registry "
                f"(remote_train_step={remote_train_step}, local_train_step={local_train_step})",
                throttle_key=f"CKPT:latest_no_new:{latest_path}:{remote_train_step}:{local_train_step}",
                throttle_seconds=self._status_log_interval("CKPT"),
            )
            self._write_local_latest_checkpoint_info(checkpoint_ref)
            return

        if local_train_step is not None and remote_train_step <= local_train_step and not archive_ready:
            self._ckpt_log(
                "Latest registry matches local latest JSON, but local checkpoint archive is missing or incomplete; "
                f"downloading checkpoint_id={checkpoint_ref.get('checkpoint_id')}"
            )
        else:
            self._ckpt_log(
                "Latest registry advanced: "
                f"remote_train_step={remote_train_step}, local_train_step={local_train_step}, "
                f"checkpoint_path={remote_latest.get('checkpoint_path')}"
            )

        self._download_checkpoint_if_needed(sftp, checkpoint_ref)
        if self._checkpoint_archive_ready(checkpoint_ref):
            self._write_local_latest_checkpoint_info(checkpoint_ref)
        else:
            self._ckpt_log(
                "Latest checkpoint download did not produce a complete local archive; "
                f"local latest registry will not be updated (checkpoint_id={checkpoint_ref.get('checkpoint_id')})",
                level="warning",
            )

    def _poll_and_download_checkpoint(self) -> None:
        if not self.online_training_enabled:
            return
        try:
            ssh = self._get_ssh_client(context="CKPT")
            sftp = ssh.open_sftp()
            try:
                latest_path = self._remote_latest_checkpoint_path()
                if latest_path:
                    self._poll_latest_checkpoint_registry(sftp, latest_path)
                    return

                self._ckpt_log(
                    "Poll start; "
                    f"remote_checkpoint_path={self._remote_checkpoint_scan_root()}, "
                    f"local_archive={self.local_checkpoint_root / 'archive'}",
                    throttle_key="CKPT:poll_start",
                    throttle_seconds=self._status_log_interval("CKPT"),
                )
                remote_refs = self._list_remote_checkpoint_refs(sftp)
                if not remote_refs:
                    self._ckpt_log(
                        "No complete remote checkpoints found",
                        throttle_key="CKPT:no_complete",
                        throttle_seconds=self._status_log_interval("CKPT"),
                    )
                    return
                local_names = set(self._local_complete_checkpoint_refs())
                sorted_remote_items = sorted(remote_refs.items(), key=self._checkpoint_sort_key)
                remote_names = [name for name, _ in sorted_remote_items]
                missing_names = [name for name, _ in sorted_remote_items if name not in local_names]
                self._record_latest_remote_policy_generation(
                    sorted_remote_items[-1][1],
                    source=str(self._remote_checkpoint_scan_root() or "remote_checkpoint_scan"),
                )
                self._ckpt_log(
                    f"Poll status; remote={remote_names}, local={sorted(local_names, key=str)}, missing={missing_names}",
                    throttle_key=(
                        "CKPT:status:"
                        f"remote={','.join(remote_names)}:"
                        f"local={','.join(sorted(local_names, key=str))}:"
                        f"missing={','.join(missing_names)}"
                    ),
                    throttle_seconds=self._status_log_interval("CKPT") if not missing_names else None,
                )
                if not missing_names:
                    self._ckpt_log(
                        "No new checkpoints to download",
                        throttle_key="CKPT:no_new",
                        throttle_seconds=self._status_log_interval("CKPT"),
                    )
                for name, checkpoint_ref in sorted_remote_items:
                    self._download_checkpoint_if_needed(sftp, checkpoint_ref)
            finally:
                sftp.close()
                ssh.close()
        except Exception as exc:
            self._ckpt_log(f"Failed to poll/download checkpoint: {exc}", level="warning")

    def _checkpoint_poll_loop(self) -> None:
        self._ckpt_log("Checkpoint polling thread started")
        self.local_checkpoint_root.mkdir(parents=True, exist_ok=True)
        latest_local = self._newest_local_checkpoint_ref()
        if latest_local is not None and not self.current_policy_path.exists():
            write_current_policy_info(self.current_policy_path, latest_local)
            self._update_current_symlink(Path(str(latest_local["checkpoint_dir"])))

        poll_interval = float(
            self.online_training_config.get("checkpoint_poll_interval")
            or self.online_training_config.get("checkpoint_poll_seconds")
            or self.poll_interval
        )
        while self._running:
            self._poll_and_download_checkpoint()
            self._stop_event.wait(max(1.0, poll_interval))
        self._ckpt_log("Checkpoint polling thread stopped")

    def _poll_and_upload(self) -> None:
        """
        Poll the data directory for new folders and upload them.
        """
        payload_format = self.upload_config.get("payload_format")
        if payload_format == "lerobot_collect_v1":
            from utils.lerobot_collect_upload import poll_collect

            poll_collect(self)
            return
        if payload_format == "lerobot_direct_v1":
            from utils.lerobot_direct_upload import upload_direct

            ready = Path(self.data_dir)
            if ready.name != "ready":
                raise ValueError("Direct LeRobot data_dir must point to ready/")
            for folder in sorted(ready.glob("*")):
                if self._stop_event.is_set():
                    break
                if folder.is_symlink() or not (folder / "manifest.json").is_file():
                    continue
                try:
                    upload_direct(self, folder)
                except Exception as exc:  # noqa: BLE001
                    self._upload_log(f"Direct LeRobot upload failed for {folder.name}: {exc}", level="warning")
            return
        raise ValueError(f"Unsupported LeRobot payload: {payload_format}")

    def run(self) -> None:
        """
        Main loop for the upload manager thread.
        """
        self._upload_log("Upload polling thread started")

        while self._running:
            try:
                self._poll_and_upload()
                self._stop_event.wait(self.poll_interval)
            except Exception as e:
                self._upload_log(f"Error in poll loop: {e}", level="warning")
                logger.exception("Poll loop traceback:")

        self._upload_log("Upload polling thread stopped")

    def start(self) -> None:
        """
        Start the upload manager thread.
        """
        if self._running:
            logger.warning("DataUploadManager is already running")
            return

        logger.info(
            "Transfer payload limits per manager: "
            f"upload={self.upload_bandwidth_mbps:g} MBps, "
            f"download={self.download_bandwidth_mbps:g} MBps; "
            "one stream per direction"
        )
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        if self.online_training_enabled:
            self._checkpoint_thread = threading.Thread(
                target=self._checkpoint_poll_loop,
                daemon=True,
                name="CheckpointDownloadThread",
            )
            self._checkpoint_thread.start()
        logger.info("DataUploadManager started")

    def stop(self) -> None:
        """
        Stop the upload manager thread.
        """
        if not self._running:
            logger.warning("DataUploadManager is not running")
            return

        logger.info("Stopping DataUploadManager...")
        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                logger.warning("DataUploadManager thread did not stop gracefully")
            else:
                logger.info("DataUploadManager stopped")

        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=10.0)
            if self._checkpoint_thread.is_alive():
                logger.warning("DataUploadManager checkpoint thread did not stop gracefully")
            else:
                logger.info("DataUploadManager checkpoint thread stopped")

    def get_status(self) -> dict:
        """
        Get the current status of the upload manager.

        Returns:
            Dictionary containing status information
        """
        return {
            "running": self._running,
            "upload_bandwidth_mbps": self.upload_bandwidth_mbps,
            "download_bandwidth_mbps": self.download_bandwidth_mbps,
            "data_dir": self.data_dir,
            "host": self.host,
            "port": self.port,
            "remote_path": self.remote_path,
            "poll_interval": self.poll_interval,
            "uploaded_folders_count": len(self._uploaded_folders),
            "uploaded_folders": list(self._uploaded_folders),
            "online_training_enabled": self.online_training_enabled,
            "policy_directory": str(self.policy_directory),
            "current_policy_path": str(self.current_policy_path),
            "local_checkpoint_root": str(self.local_checkpoint_root),
            "downloaded_checkpoints": sorted(self._downloaded_checkpoints),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Upload Manager")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config_name", type=str, help="Name of the Hydra config")
    source.add_argument("--config_file", type=str, help="Resolved session config from run.py")

    args = parser.parse_args()

    config_dict: Dict[str, Any] = {}

    if args.config_file:
        from omegaconf import OmegaConf

        resolved = OmegaConf.to_container(OmegaConf.load(args.config_file), resolve=True)
        config_dict = resolved["augmental_data_upload"]
    else:
        config_dict = load_config_from_hydra(args.config_name)
    logger.info(f"Loaded configuration from: {args.config_file or args.config_name}")
    logger.info(f"Config: {config_dict}")

    data_dir = config_dict.get("data_dir")
    remote_path = config_dict.get("remote_path")

    assert data_dir is not None, "data_dir is required in config"
    assert remote_path is not None, "remote_path is required in config"

    host = config_dict.get("host")
    port = config_dict.get("port")
    username = config_dict.get("username")
    poll_interval = config_dict.get("poll_interval")
    ssh_key_path = config_dict.get("ssh_key_path")
    online_training_config = config_dict.get("online_training", {})

    logger.info("Data Upload Manager - CLI Mode")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Remote server: {username}@{host}:{port}")
    logger.info(f"Remote path: {remote_path}")
    logger.info(f"Poll interval: {poll_interval}s")

    uploader = DataUploadManager(
        data_dir=data_dir,
        host=host,
        port=port,
        username=username,
        remote_path=remote_path,
        poll_interval=poll_interval,
        ssh_key_path=ssh_key_path,
        online_training_config=online_training_config,
        upload_config=config_dict,
    )

    uploader.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        uploader.stop()
        logger.info("DataUploadManager shut down")
