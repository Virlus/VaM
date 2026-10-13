"""
Policy Connector - runs a policy and writes action chunking to policy SHM.

This module:
1. Reads observations from summary SHM (via BaseConsumer)
2. Runs a policy to compute action chunking
3. Writes action chunking into policy SHM for ActionExecutor to consume

Policy SHM is connect-only, created by ActionExecutor.

Author: Jun Lv, Han Xue, Zheng Wang
"""

import time
import struct
import numpy as np
from typing import Dict, List, Any, Optional, Union
import sys
import os
import subprocess
import threading
import queue
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from consumer.base import BaseConsumer
from policy import POLICY_CLASSES
from utils.shm_utils import (
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    unpack_buffer_header,
    unpack_frame_header,
    pack_frame_header,
    get_dtype,
    POLICY_HEADER_SIZE,
    unpack_policy_header,
    unpack_device_header,
    pack_policy_header,
    get_policy_data_offset,
    connect_to_policy_shm,
    get_policy_action_index_offset,
    ACTION_INDEX_FORMAT,
    ACTION_INDEX_SIZE,
    connect_to_shared_memory,
    get_control_frame_offset,
    PolicyOutputSHM,
    get_device_info_offset,
)
from utils.logger_config import logger
from utils.time_control import precise_loop_timing
from utils.human_rtc import build_human_rtc_prefix
from utils.online_training import (
    current_policy_path_from_config,
    find_latest_complete_checkpoint,
    list_complete_checkpoints,
    load_current_policy_info,
    load_policy_rollout_state,
    normalize_policy_info,
    policy_directory_from_config,
    policy_rollout_state_path_from_config,
    should_count_policy_episode_result,
    write_current_policy_info,
    write_policy_rollout_state,
)

ActionType = Union[np.ndarray, Dict[str, np.ndarray]]


class PolicyConnector(BaseConsumer):
    """
    Connector that:
      - Reads observations from summary SHM (via BaseConsumer)
      - Runs a policy to compute action chunking
      - Writes action chunking into policy SHM for ActionExecutor to consume

    Policy SHM is connect-only, created by ActionExecutor.
    """

    def __init__(
        self,
        summary_shm_name: str = "device_summary_data",
        policy_class: str = "VAMPolicy",
        policy_params: Optional[Dict[str, Any]] = None,
        obs_devices: Optional[List[str]] = None,
        controls: Optional[List[Dict[str, Any]]] = None,
        reset_cfg: Optional[List[Dict[str, Any]]] = None,
        halt_and_home_cfg: Optional[List[Dict[str, Any]]] = None,
        fps: float = 50.0,
        control_fps: float = 50.0,
        value_prediction_query_fps: float = 100.0,
        history_with_value: bool = False,
        enable_value_prediction: bool = False,
        chunk_length: int = 8,
        retry_connect_secs: float = 0.5,
        master_device: Optional[str] = None,
        infer_rollout_config: Optional[Dict[str, Any]] = None,
        config_file_path: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
        online_training_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(summary_shm_name)
        self.config_file_path = config_file_path
        self.policy_class = policy_class
        self.policy_params = dict(policy_params or {})
        self.online_training_config: Dict[str, Any] = dict(online_training_config or {})
        self.online_training_enabled = bool(self.online_training_config.get("enabled", False))
        self.max_current_policy_rollouts = self._configured_max_current_policy_rollouts(self.online_training_config)
        self.policy_rollout_limit_wait_interval = float(
            self.online_training_config.get("policy_rollout_limit_wait_interval")
            or self.online_training_config.get("max_policy_rollout_wait_interval")
            or self.online_training_config.get("current_policy_wait_interval")
            or 10.0
        )
        startup_ckpt_dir = (
            self.online_training_config.get("startup_checkpoint_path")
            or self.online_training_config.get("startup_checkpoint_dir")
            or self.online_training_config.get("base_checkpoint_path")
            or self.online_training_config.get("base_checkpoint_dir")
            or self.online_training_config.get("initial_checkpoint_path")
            or self.online_training_config.get("initial_checkpoint_dir")
        )
        if self.online_training_enabled and startup_ckpt_dir:
            self.policy_params["ckpt_dir"] = str(Path(startup_ckpt_dir).expanduser())
        self.obs_devices = obs_devices or []
        assert master_device is None or master_device in obs_devices, (
            f"master_device is not in obs_devices {obs_devices}"
        )
        self.master_device = master_device
        self.chunk_length = chunk_length
        self.controls_config: List[Dict[str, Any]] = []
        if controls:
            for c in controls:
                item = {
                    "shm": c["shm"],
                    "shape": tuple(c["shape"]) if "shape" in c and c["shape"] is not None else None,
                    "dtype": c.get("dtype", "float64"),
                }
                self.controls_config.append(item)
        self.reset_config: List[Dict[str, Any]] = reset_cfg or []
        self.halt_and_home_cfg: List[Dict[str, Any]] = halt_and_home_cfg or []
        self.fps = fps
        self.control_fps = control_fps
        self.history_with_value = history_with_value
        self.value_prediction_query_fps = value_prediction_query_fps
        self.update_interval = 1.0 / max(self.fps, 1e-06)
        self.value_prediction_query_interval = 1.0 / max(self.value_prediction_query_fps, 1e-06)
        self.retry_connect_secs = retry_connect_secs
        self.enable_value_prediction = enable_value_prediction
        self.policy_shm = None
        if self.policy_class not in POLICY_CLASSES:
            raise ValueError(f"Unknown policy_class: {self.policy_class}")
        policy_cls = POLICY_CLASSES[self.policy_class]
        if policy_cls is None:
            raise ValueError(f"Policy class '{self.policy_class}' is registered but not available (import failed)")
        self.policy = policy_cls(**self.policy_params)
        self.policy.load()
        self._policy_lock = threading.RLock()
        self._human_rtc_shms: Dict[str, Any] = {}
        self._human_rtc_active = False
        self.action_step_interval_s: float = 1.0 / self.control_fps
        self.action_step_interval_ns: int = int(self.action_step_interval_s * 1000000000.0)
        self.action_configs: List[Dict[str, Any]] = self._build_action_configs()
        self.infer_rollout_config: Dict[str, Any] = dict(infer_rollout_config or {})
        self.infer_rollout_process: Optional[subprocess.Popen] = None
        self._infer_rollout_started: bool = False
        default_policy_dir = Path(self.policy_params.get("ckpt_dir", ".")).expanduser().parent / "online_training"
        self.policy_directory = policy_directory_from_config(self.online_training_config, default_policy_dir)
        self.current_policy_path = current_policy_path_from_config(self.online_training_config, self.policy_directory)
        rollout_data_dir = (
            self.infer_rollout_config.get("output_dir")
            or self.online_training_config.get("rollout_data_dir")
            or self.policy_directory
        )
        self.policy_rollout_state_path = policy_rollout_state_path_from_config(
            self.online_training_config, rollout_data_dir
        )
        self._persisted_policy_rollout_state = (
            load_policy_rollout_state(self.policy_rollout_state_path) if self.online_training_enabled else {}
        )
        self.local_checkpoint_root = Path(
            self.online_training_config.get("local_checkpoint_root")
            or self.online_training_config.get("local_checkpoint_dir")
            or self.policy_directory / "checkpoints"
        ).expanduser()
        if self.online_training_enabled:
            self.current_policy_info = self._initial_policy_info()
            self.current_policy_info["current_policy_episode_cnt"] = self._restored_policy_rollout_count(
                self.current_policy_info
            )
            self._current_policy_checkpoint_id = self.current_policy_info.get("checkpoint_id")
            self._current_policy_checkpoint_dir = self._normalize_checkpoint_path(
                self.current_policy_info.get("checkpoint_dir")
            )
            self.policy_directory.mkdir(parents=True, exist_ok=True)
            if not self.current_policy_path.exists():
                write_current_policy_info(
                    self.current_policy_path, self._policy_registry_info(self.current_policy_info)
                )
            self._online_policy_log(
                f"Online training enabled: current_policy_path={self.current_policy_path}, policy_rollout_state_path={self.policy_rollout_state_path}, local_checkpoint_root={self.local_checkpoint_root}"
            )
            self._log_online_policy_scan("startup")
        else:
            self.current_policy_info = {}
            self._current_policy_checkpoint_id = None
            self._current_policy_checkpoint_dir = None
        self.rollout_python_path = self.infer_rollout_config.get(
            "python_path",
            os.path.join(str(Path(__file__).resolve().parents[2]), ".venv", "bin", "python"),
        )
        signal_files = {}
        if runtime_config is not None:
            signal_files = runtime_config.get("signal_files", {})
        required_signals = [
            "start_recording",
            "stop_recording",
            "reset_signal",
            "robot_home_index",
            "episode_result",
            "policy_runner_reset_command",
            "teleoperator_reset_command",
            "rollout_ready",
            "robot_halt_command_pattern",
            "robot_halt_reply_pattern",
        ]
        for sig in required_signals:
            if sig not in signal_files:
                raise ValueError(f"signal_files.{sig} is required in runtime config")
        self.signal_files = signal_files
        self.inference_enabled = True
        if self.online_training_enabled:
            self._maybe_switch_policy_at_episode_boundary()
            self._persist_policy_rollout_state(reason="startup")
        self.last_reset_signal_time = 0.0
        episode_control_config = self.infer_rollout_config.get("episode_control", {})
        self.episode_control_enabled = episode_control_config.get("enabled", False)
        self.episode_success_key = episode_control_config.get("success_key", "s")
        self.episode_failure_key = episode_control_config.get("failure_key", "f")
        self.episode_discard_key = episode_control_config.get("discard_key", "d")
        self.episode_auto_restart = episode_control_config.get("auto_restart", True)
        self.episode_home_wait_time = episode_control_config.get("home_wait_time", 3.0)
        self._episode_result_queue: queue.Queue = queue.Queue()
        self._keyboard_listener = None
        self._episode_in_transition = False
        self._episode_count = 0
        self.policy_outputs_config: Dict[str, Any] = dict(infer_rollout_config.get("policy_outputs", {}))
        self.value_shm: PolicyOutputSHM = PolicyOutputSHM(
            config=self.policy_outputs_config.get("value", {}), fps=self.fps
        )
        self.action_chunk_shm: PolicyOutputSHM = PolicyOutputSHM(
            config=self.policy_outputs_config.get("action_chunk", {}), fps=self.fps
        )
        self.value_prediction_thread: Optional[threading.Thread] = None
        self.value_prediction_running: bool = False

    def _close_human_rtc_shms(self) -> None:
        for channel in self._human_rtc_shms.values():
            try:
                channel.close()
            except Exception:
                pass
        self._human_rtc_shms.clear()

    def _read_direct_shm_frame(self, shm_name: str, frames_back: int = 0) -> Optional[tuple[int, np.ndarray]]:
        """Read a recent frame from a standard device/control ring-buffer SHM."""
        channel = self._human_rtc_shms.get(shm_name)
        if channel is None:
            try:
                channel = connect_to_shared_memory(shm_name, read_only=True)
                self._human_rtc_shms[shm_name] = channel
                logger.info(f"PolicyConnector: Connected to human RTC SHM {shm_name}")
            except FileNotFoundError:
                return None
        try:
            device_info = unpack_device_header(bytes(channel.buf[:DEVICE_HEADER_SIZE]))
            buffer_info = unpack_buffer_header(
                bytes(channel.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            )
            current_count = int(buffer_info["current_frames_count"])
            buffer_size = int(buffer_info["buffer_size"])
            frames_back = max(0, int(frames_back))
            if current_count <= frames_back or buffer_size <= 0:
                return None
            frame_size = int(device_info["frame_size"])
            frame_index = (int(buffer_info["write_index"]) - 1 - frames_back) % buffer_size
            frame_offset = get_control_frame_offset(frame_index, frame_size)
            timestamp_ns = unpack_frame_header(bytes(channel.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE]))[
                "timestamp_ns"
            ]
            if timestamp_ns == 0:
                return None
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = frame_offset + frame_size
            dtype = get_dtype(device_info["data_dtype"])
            data = np.frombuffer(bytes(channel.buf[data_start:data_end]), dtype=dtype)
            return (int(timestamp_ns), data.reshape(device_info["shape"]).copy())
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to read human RTC SHM {shm_name}: {exc}")
            try:
                channel.close()
            except Exception:
                pass
            self._human_rtc_shms.pop(shm_name, None)
            return None

    def _prepare_human_rtc_prefix(self) -> Optional[np.ndarray]:
        if not bool(getattr(self.policy, "enable_human_rtc", False)):
            return None
        flag_shm = getattr(self.policy, "human_rtc_control_flag_shm", None)
        action_shm = getattr(self.policy, "human_rtc_action_shm", None)
        if not flag_shm or not action_shm:
            return None
        flag_frame = self._read_direct_shm_frame(flag_shm)
        human_in_control = flag_frame is not None and float(np.asarray(flag_frame[1]).reshape(-1)[0]) > 0.5
        if not human_in_control:
            if self._human_rtc_active:
                logger.info("PolicyConnector: Human control ended; continuing with shadow RTC chunk")
            self._human_rtc_active = False
            return None
        if not self._human_rtc_active:
            logger.info("PolicyConnector: Human control active; seeding shadow RTC from human command SHM")
        self._human_rtc_active = True
        latest = self._read_direct_shm_frame(action_shm, frames_back=0)
        previous = self._read_direct_shm_frame(action_shm, frames_back=1)
        if latest is None:
            return None
        if previous is None or previous[0] >= latest[0]:
            previous = (latest[0] - self.action_step_interval_ns, latest[1])
        sample_dt_s = (latest[0] - previous[0]) / 1000000000.0
        return build_human_rtc_prefix(
            latest_action=latest[1],
            previous_action=previous[1],
            sample_dt_s=sample_dt_s,
            chunk_length=self.chunk_length,
            control_fps=self.control_fps,
            velocity_decay_steps=int(getattr(self.policy, "human_rtc_velocity_decay_steps", 20)),
            max_velocity=float(getattr(self.policy, "human_rtc_max_velocity", 2.0)),
        )

    def _online_policy_log(self, message: str, *, level: str = "info") -> None:
        try:
            logger.opt(colors=True).log(level.upper(), f"<cyan>[POLICY]</cyan> {message}")
        except Exception:
            getattr(logger, level.lower())(f"[POLICY] {message}")

    @staticmethod
    def _configured_max_current_policy_rollouts(config: Dict[str, Any]) -> Optional[int]:
        for key in (
            "max_current_policy_rollouts",
            "maximum_current_policy_rollouts",
            "max_current_policy_episode_cnt",
            "maximum_current_policy_episode_cnt",
            "max_rollouts_per_policy",
            "maximum_rollouts_per_policy",
        ):
            value = config.get(key)
            if value is None or value == "":
                continue
            try:
                limit = int(value)
            except (TypeError, ValueError):
                logger.warning(f"PolicyConnector: Ignoring invalid {key}={value!r}")
                return None
            return limit if limit > 0 else None
        return None

    @staticmethod
    def _coerce_episode_count(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _format_policy_info_for_log(self, policy_info: Dict[str, Any]) -> str:
        checkpoint_name = policy_info.get("checkpoint_name") or Path(str(policy_info.get("checkpoint_dir", ""))).name
        return f"{checkpoint_name}(policy_id={policy_info.get('policy_id')}, checkpoint_id={policy_info.get('checkpoint_id')})"

    @staticmethod
    def _normalize_checkpoint_path(checkpoint_dir: Any) -> Optional[str]:
        if not checkpoint_dir:
            return None
        return str(Path(str(checkpoint_dir)).expanduser().resolve(strict=False))

    def _is_loaded_checkpoint_dir(self, checkpoint_dir: Any) -> bool:
        normalized_dir = self._normalize_checkpoint_path(checkpoint_dir)
        return bool(normalized_dir and normalized_dir == self._current_policy_checkpoint_dir)

    @staticmethod
    def _policy_registry_info(policy_info: Dict[str, Any]) -> Dict[str, Any]:
        registry_info = dict(policy_info)
        registry_info.pop("current_policy_episode_cnt", None)
        return registry_info

    def _policy_rollout_state_matches(self, policy_info: Dict[str, Any]) -> bool:
        state = self._persisted_policy_rollout_state
        if not state:
            return False
        state_checkpoint_id = state.get("checkpoint_id")
        policy_checkpoint_id = policy_info.get("checkpoint_id")
        if state_checkpoint_id and policy_checkpoint_id:
            if str(state_checkpoint_id) != str(policy_checkpoint_id):
                return False
            state_step = state.get("train_step", state.get("checkpoint_index"))
            policy_step = policy_info.get("train_step", policy_info.get("checkpoint_index"))
            if state_step is not None and policy_step is not None:
                return str(state_step) == str(policy_step)
            return True
        state_step = state.get("train_step", state.get("checkpoint_index"))
        policy_step = policy_info.get("train_step", policy_info.get("checkpoint_index"))
        if state_step is not None and policy_step is not None:
            return str(state_step) == str(policy_step)
        state_dir = self._normalize_checkpoint_path(state.get("checkpoint_dir"))
        policy_dir = self._normalize_checkpoint_path(policy_info.get("checkpoint_dir"))
        return bool(state_dir and policy_dir and (state_dir == policy_dir))

    def _restored_policy_rollout_count(self, policy_info: Dict[str, Any]) -> int:
        if not self._persisted_policy_rollout_state:
            return self._coerce_episode_count(policy_info.get("current_policy_episode_cnt"))
        if not self._policy_rollout_state_matches(policy_info):
            return 0
        return self._coerce_episode_count(self._persisted_policy_rollout_state.get("current_policy_episode_cnt"))

    def _current_policy_episode_count(self) -> int:
        return self._coerce_episode_count(
            self.current_policy_info.get("current_policy_episode_cnt") if self.current_policy_info else 0
        )

    def _persist_policy_rollout_state(self, *, reason: str) -> None:
        if not self.online_training_enabled:
            return
        policy_info = dict(self.current_policy_info)
        policy_info["current_policy_episode_cnt"] = self._current_policy_episode_count()
        policy_info["max_current_policy_rollouts"] = self.max_current_policy_rollouts
        try:
            self._persisted_policy_rollout_state = write_policy_rollout_state(
                self.policy_rollout_state_path, policy_info
            )
            self._online_policy_log(
                f"Updated policy rollout state: count={self._persisted_policy_rollout_state.get('current_policy_episode_cnt')}, checkpoint_id={self.current_policy_info.get('checkpoint_id')}, reason={reason}"
            )
        except Exception as exc:
            self._online_policy_log(
                f"Failed to persist current policy rollout count during {reason}: {exc}", level="warning"
            )

    def _increment_current_policy_episode_count(self, result: str) -> None:
        if not self.online_training_enabled:
            return
        if not should_count_policy_episode_result(result):
            self._online_policy_log(
                f"Episode result does not count toward current policy rollout budget: result={result}, count={self._current_policy_episode_count()}, checkpoint_id={self.current_policy_info.get('checkpoint_id')}"
            )
            return
        count = self._current_policy_episode_count() + 1
        self.current_policy_info["current_policy_episode_cnt"] = count
        self._persist_policy_rollout_state(reason=f"episode_end:{result}")

    def _current_policy_rollout_limit_reached(self) -> bool:
        if not self.online_training_enabled or self.max_current_policy_rollouts is None:
            return False
        return self._current_policy_episode_count() >= self.max_current_policy_rollouts

    def _log_online_policy_scan(self, reason: str, *, selected_policy_info: Optional[Dict[str, Any]] = None) -> None:
        if not self.online_training_enabled:
            return
        try:
            checkpoints = list_complete_checkpoints(self.local_checkpoint_root)
        except Exception as exc:
            self._online_policy_log(f"Failed to scan local checkpoints during {reason}: {exc}", level="warning")
            checkpoints = []
        checkpoint_summary = (
            ", ".join((self._format_policy_info_for_log(info) for info in checkpoints)) if checkpoints else "none"
        )
        active_policy_info = selected_policy_info or self.current_policy_info
        self._online_policy_log(
            f"Local checkpoint scan during {reason}: detected=[{checkpoint_summary}], active_policy_id={active_policy_info.get('policy_id')}, active_checkpoint_id={active_policy_info.get('checkpoint_id')}, active_checkpoint_dir={active_policy_info.get('checkpoint_dir')}"
        )

    def _initial_policy_info(self) -> Dict[str, Any]:
        """
        Build policy metadata for the policy loaded during __init__.

        The loaded startup/base policy object is the source of truth before the
        first hot-reload check. Existing current_policy.json is intentionally
        left untouched so _maybe_switch_policy_at_episode_boundary() can adopt
        the current downloaded policy immediately after startup.
        """
        default_policy_id = self.online_training_config.get("initial_policy_id", 0)
        ckpt_dir = self.policy_params.get("ckpt_dir")
        return normalize_policy_info(
            {
                "policy_id": default_policy_id,
                "checkpoint_id": self.online_training_config.get("initial_checkpoint_id"),
                "checkpoint_index": self.online_training_config.get("initial_checkpoint_index"),
                "checkpoint_dir": ckpt_dir,
                "current_policy_episode_cnt": 0,
                "policy_source": "startup_config",
            },
            ckpt_dir=ckpt_dir,
            default_policy_id=default_policy_id,
        )

    def _candidate_policy_info_from_local_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not self.online_training_enabled:
            return None
        current_info: Optional[Dict[str, Any]] = None
        try:
            current_info = load_current_policy_info(self.current_policy_path)
            current_ckpt_dir = current_info.get("checkpoint_dir")
            if (
                current_ckpt_dir
                and Path(str(current_ckpt_dir)).exists()
                and (not self._is_loaded_checkpoint_dir(current_ckpt_dir))
            ):
                candidate = normalize_policy_info(current_info)
                self._log_online_policy_scan("episode boundary", selected_policy_info=candidate)
                return candidate
        except Exception as exc:
            logger.debug(f"PolicyConnector: Failed to read current policy registry: {exc}")
        latest_local = find_latest_complete_checkpoint(self.local_checkpoint_root)
        if latest_local is not None and (not self._is_loaded_checkpoint_dir(latest_local.get("checkpoint_dir"))):
            self._log_online_policy_scan("episode boundary", selected_policy_info=latest_local)
            return latest_local
        if current_info is not None:
            current_ckpt_dir = current_info.get("checkpoint_dir")
            if current_ckpt_dir and Path(str(current_ckpt_dir)).exists():
                candidate = normalize_policy_info(current_info)
                self._log_online_policy_scan("episode boundary", selected_policy_info=candidate)
                return candidate
        self._log_online_policy_scan("episode boundary")
        return latest_local

    def _reload_policy_from_info(self, policy_info: Dict[str, Any]) -> bool:
        checkpoint_dir = policy_info.get("checkpoint_dir")
        _checkpoint_id = policy_info.get("checkpoint_id")
        if not checkpoint_dir:
            logger.warning("PolicyConnector: New policy info does not include checkpoint_dir")
            return False
        checkpoint_path = Path(str(checkpoint_dir)).expanduser()
        if not checkpoint_path.exists():
            logger.warning(f"PolicyConnector: New checkpoint directory does not exist: {checkpoint_path}")
            return False
        if not (checkpoint_path / "params").exists() and checkpoint_path.name != "params":
            logger.warning(f"PolicyConnector: New checkpoint directory has no params/: {checkpoint_path}")
            return False
        if self._is_loaded_checkpoint_dir(checkpoint_path):
            return False
        normalized_info = normalize_policy_info(policy_info, ckpt_dir=checkpoint_path)
        normalized_info["checkpoint_dir"] = str(checkpoint_path)
        normalized_info["params_dir"] = (
            str(checkpoint_path) if checkpoint_path.name == "params" else str(checkpoint_path / "params")
        )
        normalized_info["policy_source"] = normalized_info.get("policy_source") or "local_checkpoint"
        normalized_info["current_policy_episode_cnt"] = self._restored_policy_rollout_count(normalized_info)
        self._online_policy_log(
            f"Switching policy checkpoint {self._current_policy_checkpoint_id} -> {normalized_info.get('checkpoint_id')}"
        )
        was_inference_enabled = self.inference_enabled
        self.inference_enabled = False
        with self._policy_lock:
            try:
                if hasattr(self.policy, "reload_checkpoint"):
                    self.policy.reload_checkpoint(
                        str(checkpoint_path),
                        policy_id=normalized_info.get("policy_id"),
                        checkpoint_id=str(normalized_info.get("checkpoint_id")),
                    )
                else:
                    self.policy_params["ckpt_dir"] = str(checkpoint_path)
                    policy_cls = POLICY_CLASSES[self.policy_class]
                    if policy_cls is None:
                        raise ValueError(f"Policy class '{self.policy_class}' is not available")
                    self.policy = policy_cls(**self.policy_params)
                    self.policy.load()
                    if hasattr(self.policy, "reset"):
                        self.policy.reset()
                registry_info = write_current_policy_info(
                    self.current_policy_path, self._policy_registry_info(normalized_info)
                )
                self.current_policy_info = dict(registry_info)
                self.current_policy_info["current_policy_episode_cnt"] = normalized_info["current_policy_episode_cnt"]
                self._current_policy_checkpoint_id = self.current_policy_info.get("checkpoint_id")
                self._current_policy_checkpoint_dir = self._normalize_checkpoint_path(
                    self.current_policy_info.get("checkpoint_dir")
                )
                self.policy_params["ckpt_dir"] = str(checkpoint_path)
            except Exception as exc:
                logger.error(f"PolicyConnector: Failed to switch policy checkpoint: {exc}")
                return False
            finally:
                self.inference_enabled = was_inference_enabled
        self._signal_policy_runner_reset()
        self._persist_policy_rollout_state(reason="checkpoint_reload")
        self._online_policy_log(
            f"Policy checkpoint switched: policy_id={self.current_policy_info.get('policy_id')}, checkpoint_id={self.current_policy_info.get('checkpoint_id')}, checkpoint_dir={self.current_policy_info.get('checkpoint_dir')}, current_policy_episode_cnt={self._current_policy_episode_count()}"
        )
        return True

    def _maybe_switch_policy_at_episode_boundary(self) -> bool:
        policy_info = self._candidate_policy_info_from_local_checkpoint()
        if policy_info is None:
            return False
        return self._reload_policy_from_info(policy_info)

    def _maybe_switch_or_wait_for_policy_budget(self) -> None:
        if not self.online_training_enabled:
            return
        if self._maybe_switch_policy_at_episode_boundary():
            return
        if not self._current_policy_rollout_limit_reached():
            return
        limit = self.max_current_policy_rollouts
        count = self._current_policy_episode_count()
        resume_inference_after_wait = self.inference_enabled
        self.inference_enabled = False
        self._signal_policy_runner_reset()
        self._set_rollout_display_phase("wait_policy")
        self._online_policy_log(
            f"Current policy rollout limit reached; waiting for a new policy before next episode. checkpoint_id={self.current_policy_info.get('checkpoint_id')}, count={count}, limit={limit}"
        )
        last_log_time = time.monotonic()
        while self.running:
            time.sleep(max(1.0, self.policy_rollout_limit_wait_interval))
            if self._maybe_switch_policy_at_episode_boundary():
                self.inference_enabled = resume_inference_after_wait
                self._set_rollout_display_phase("idle")
                self._online_policy_log(
                    f"New current policy loaded after rollout-limit wait; continuing to next episode. checkpoint_id={self.current_policy_info.get('checkpoint_id')}, count={self.current_policy_info.get('current_policy_episode_cnt')}"
                )
                return
            now = time.monotonic()
            if now - last_log_time >= max(30.0, self.policy_rollout_limit_wait_interval):
                self._online_policy_log(
                    f"Still waiting for a new policy because current policy rollout limit is reached. checkpoint_id={self.current_policy_info.get('checkpoint_id')}, count={self._current_policy_episode_count()}, limit={limit}, policy_rollout_state_path={self.policy_rollout_state_path}"
                )
                last_log_time = now
        self._online_policy_log(
            "Stopped waiting for a new policy because PolicyConnector is shutting down", level="warning"
        )

    def _check_reset_signal(self) -> bool:
        """Check if reset signal file exists and has been updated."""
        try:
            reset_signal_file = Path(self.signal_files.get("reset_signal"))
            if reset_signal_file.exists():
                current_mtime = reset_signal_file.stat().st_mtime
                if current_mtime > self.last_reset_signal_time:
                    self.last_reset_signal_time = current_mtime
                    return True
            return False
        except Exception as e:
            logger.debug(f"Failed to check reset signal: {e}")
            return False

    def find_value_by_timestamp_with_history_too_old_detection(
        self, target_timestamp_ns: int, tolerance_ms: float = 50.0
    ) -> Optional[Union[np.ndarray, str]]:
        """Find the closest value prediction and detect when the requested timestamp is too old."""
        try:
            if not self.value_shm.created or self.value_shm.shm is None or self.value_shm.shape is None:
                logger.warning("PolicyConnector: Value SHM not created or invalid")
                return None
            buf = self.value_shm.shm.buf
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            buffer_size = buffer_header["buffer_size"]
            current_frames_count = buffer_header["current_frames_count"]
            write_index = buffer_header["write_index"]
            if current_frames_count == 0:
                logger.warning("PolicyConnector: Value SHM is empty")
                return None
            tolerance_ns = int(tolerance_ms * 1000000)
            if current_frames_count < buffer_size:
                start_idx = 0
                end_idx = current_frames_count - 1
            else:
                start_idx = write_index
                end_idx = write_index + buffer_size - 1

            def _value_frame_timestamp(frame_idx: int) -> int:
                frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_idx * self.value_shm.frame_size
                return unpack_frame_header(buf[frame_offset : frame_offset + FRAME_HEADER_SIZE])["timestamp_ns"]

            best_frame_index = None
            best_timestamp_diff = float("inf")
            left, right = (start_idx, end_idx)
            while left <= right:
                mid = (left + right) // 2
                actual_frame_idx = mid % buffer_size
                frame_timestamp_ns = _value_frame_timestamp(actual_frame_idx)
                timestamp_diff = abs(frame_timestamp_ns - target_timestamp_ns)
                if timestamp_diff < best_timestamp_diff:
                    best_timestamp_diff = timestamp_diff
                    best_frame_index = actual_frame_idx
                if frame_timestamp_ns < target_timestamp_ns:
                    left = mid + 1
                elif frame_timestamp_ns > target_timestamp_ns:
                    right = mid - 1
                else:
                    best_timestamp_diff = 0
                    break
            if best_frame_index is not None and best_timestamp_diff <= tolerance_ns:
                frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + best_frame_index * self.value_shm.frame_size
                data_start = frame_offset + FRAME_HEADER_SIZE
                data_end = data_start + self.value_shm.frame_data_size
                value = np.frombuffer(
                    buf[data_start:data_end], dtype=get_dtype(self.value_shm.dtype_str, np.float64)
                ).copy()
                return value.reshape(self.value_shm.shape)
            logger.warning(
                f"No value prediction found within tolerance. best_diff={best_timestamp_diff / 1000000:.2f}ms, tolerance={tolerance_ms}ms"
            )
            return None
        except Exception as e:
            logger.error(f"Error finding value by timestamp with no-history detection: {e}")
            return None

    def _value_prediction_worker(self) -> None:
        """Background thread worker for value prediction."""
        wait_for_next_value_prediction = precise_loop_timing(self.value_prediction_query_interval)
        while self.value_prediction_running:
            _loop_start_time = time.time()
            try:
                obs_dict, obs_time = self._assemble_observation_dict()
                if not obs_dict or obs_time is None:
                    logger.warning("PolicyConnector: Observation is None")
                    wait_for_next_value_prediction()
                    continue
                if not self.inference_enabled:
                    wait_for_next_value_prediction()
                    continue
                policy_input = self._prepare_policy_input(obs_dict, obs_time, history_with_value=False)
                action_dim = self.action_configs[0]["action_dim"]
                predicted_action = np.zeros((self.chunk_length, action_dim))
                if not self.enable_value_prediction or not hasattr(self.policy, "predict_value"):
                    logger.warning(
                        "PolicyConnector: Value prediction is disabled or policy does not have predict_value method"
                    )
                    continue
                try:
                    with self._policy_lock:
                        value_result = self.policy.predict_value(policy_input, predicted_action)
                    value_data = value_result.get("value")
                    if value_data is not None:
                        value_array = np.asarray(value_data, dtype=np.float64)
                        if value_array.size != 1:
                            raise ValueError("VAM must predict one scalar value per observation")
                        if value_array.ndim == 0:
                            value_array = value_array.reshape(1)
                        if not self.value_shm.created:
                            self.value_shm.create(value_array.shape)
                        if self.value_shm.created:
                            self.value_shm.write(value_array, obs_time)
                        else:
                            logger.warning(f"Value SHM not created: value={self.value_shm.created}")
                    else:
                        logger.warning("PolicyConnector: Value prediction result is None")
                except Exception as e:
                    logger.warning(f"Value prediction failed: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in value prediction worker: {e}")
                continue

    def _reconnect_all_shms(self) -> None:
        """Reconnect to all SHMs (summary SHM and policy SHM) after reset signal."""
        max_retries = 50
        retry_delay = 0.5
        self._close_human_rtc_shms()
        old_summary_shm = None
        old_policy_shm = None
        if hasattr(self, "summary_shm") and self.summary_shm is not None:
            try:
                _ = self.summary_shm.buf
                old_summary_shm = self.summary_shm
            except (ValueError, AttributeError, OSError, BufferError):
                old_summary_shm = None
                self.summary_shm = None
        if self.policy_shm is not None:
            try:
                _ = self.policy_shm.buf
                old_policy_shm = self.policy_shm
            except (ValueError, AttributeError, OSError, BufferError):
                old_policy_shm = None
                self.policy_shm = None
        summary_reconnected = False
        for attempt in range(max_retries):
            try:
                logger.info(f"PolicyConnector: Reconnecting summary SHM... (attempt {attempt + 1}/{max_retries})")
                if old_summary_shm is not None and self.summary_shm == old_summary_shm:
                    try:
                        old_summary_shm.close()
                    except Exception:
                        pass
                    self.summary_shm = None
                    old_summary_shm = None
                if self.connect():
                    summary_reconnected = True
                    logger.info("PolicyConnector: Successfully reconnected to summary SHM")
                    break
                else:
                    logger.warning(
                        f"PolicyConnector: Failed to reconnect to summary SHM (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                logger.warning(
                    f"PolicyConnector: Error reconnecting summary SHM (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        if not summary_reconnected:
            logger.error("PolicyConnector: Failed to reconnect to summary SHM after all retries")
        policy_reconnected = False
        for attempt in range(max_retries):
            try:
                logger.info(f"PolicyConnector: Reconnecting policy SHM... (attempt {attempt + 1}/{max_retries})")
                if old_policy_shm is not None and self.policy_shm == old_policy_shm:
                    try:
                        old_policy_shm.close()
                    except Exception:
                        pass
                    self.policy_shm = None
                    old_policy_shm = None
                self._connect_policy_shm()
                policy_reconnected = True
                logger.info("PolicyConnector: Successfully reconnected to policy SHM")
                break
            except FileNotFoundError:
                logger.warning(
                    f"PolicyConnector: Policy SHM not found (attempt {attempt + 1}/{max_retries}), waiting..."
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
            except Exception as e:
                logger.warning(
                    f"PolicyConnector: Error reconnecting policy SHM (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        if not policy_reconnected:
            logger.error("PolicyConnector: Failed to reconnect to policy SHM after all retries")
        if summary_reconnected and policy_reconnected:
            logger.info("PolicyConnector: Successfully reconnected all SHMs after reset")
        else:
            logger.warning("PolicyConnector: Partial reconnection - some SHMs may need retry in next cycle")

    def _merge_action_chunks(self, action_chunk_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Merge multi-device action chunks into a single array for recording.
        Chunks are concatenated along the action dimension in sorted device order.
        """
        if not action_chunk_dict:
            return None
        try:
            device_names = sorted(action_chunk_dict.keys())
            chunks = [action_chunk_dict[name] for name in device_names]
            base_len = chunks[0].shape[0]
            if any((c.shape[0] != base_len for c in chunks)):
                logger.warning("Action chunk merge skipped due to mismatched chunk lengths")
                return None
            merged = np.concatenate(chunks, axis=1)
            return merged
        except Exception as exc:
            logger.warning(f"Failed to merge action chunks: {exc}")
            return None

    def _connect_policy_shm(self) -> None:
        """Connect to policy SHM created by ActionExecutor."""
        try:
            start_time = time.time()
            timeout_sec = max(10.0, 60.0)
            while True:
                try:
                    self.policy_shm = connect_to_policy_shm(read_only=False)
                    break
                except Exception as exc:
                    if time.time() - start_time < timeout_sec:
                        logger.info(
                            f"PolicyConnector: Waiting for policy SHM 'policy_data'... retry in {self.retry_connect_secs}s"
                        )
                        time.sleep(self.retry_connect_secs)
                        continue
                    raise exc
            buf = self.policy_shm.buf
            manager_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
            device_count = manager_header["device_count"]
            self.policy_devices = []
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset : device_offset + DEVICE_HEADER_SIZE])
                device_name = device_header["device_type"]
                shape = device_header["shape"]
                frame_size = device_header["frame_size"]
                self.policy_devices.append(
                    {"device_name": device_name, "shape": shape, "frame_size": frame_size, "data_offset": None}
                )
            for device_info in self.policy_devices:
                logger.debug(f"Device {device_info['device_name']}: expected_shape={device_info['shape']}")
        except Exception as e:
            logger.error(f"PolicyConnector: Failed to connect to policy SHM: {e}")
            raise

    def _read_policy_action_index(self) -> Optional[int]:
        """Read action_index stored at the tail of policy SHM."""
        if not self.policy_shm or not self.policy_devices:
            return None
        try:
            device_count = len(self.policy_devices)
            frame_sizes = [device["frame_size"] for device in self.policy_devices]
            offset = get_policy_action_index_offset(device_count, frame_sizes)
            return struct.unpack(ACTION_INDEX_FORMAT, self.policy_shm.buf[offset : offset + ACTION_INDEX_SIZE])[0]
        except Exception as exc:
            logger.warning(f"PolicyConnector: Failed to read action_index from policy SHM: {exc}")
            return None

    def _build_action_configs(self) -> List[Dict[str, Any]]:
        action_configs: List[Dict[str, Any]] = []
        for cfg in self.controls_config:
            shm_name = cfg["shm"]
            device_name = shm_name.replace("_control", "") if shm_name.endswith("_control") else shm_name
            shape = cfg.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) > 0:
                action_dim = int(shape[0])
            else:
                raise ValueError(f"Invalid shape for control device {device_name}: {shape}")
            action_configs.append({"device_name": device_name, "action_dim": action_dim, "shm_name": shm_name})
        return action_configs

    def _get_action_chunk_shape(self) -> tuple:
        """
        Compute merged action chunk shape based on action_configs.
        Returns (chunk_length, total_action_dim).
        """
        total_dim = 0
        for cfg in self.action_configs:
            total_dim += int(cfg.get("action_dim", 0))
        if total_dim <= 0:
            total_dim = 1
        return (self.chunk_length, total_dim)

    def _should_start_infer_rollout(self) -> bool:
        if not self.infer_rollout_config or not self.infer_rollout_config.get("enabled", False):
            return False
        if self.infer_rollout_config.get("format", "lerobot") != "lerobot":
            raise ValueError("VAM rollout supports only LeRobot")
        saver = Path(__file__).with_name("async_infer_data_saver_lerobot.py")
        if not saver.is_file() or not Path(self.rollout_python_path).is_file():
            raise FileNotFoundError("Configure the LeRobot recorder and its Python interpreter")
        return True

    def _start_infer_rollout(self) -> None:
        if self._infer_rollout_started:
            return
        from utils.lerobot_rollout import start_saver

        start_saver(self)

    def _stop_infer_rollout(self) -> None:
        """Stop the inference rollout data saver subprocess."""
        from utils.lerobot_rollout import drain_saver

        drain_saver(self)
        return

    def _set_rollout_display_phase(self, phase: str) -> None:
        progress = getattr(self, "_lerobot_progress", None)
        if progress is not None:
            progress.set_phase(phase)

    def _signal_start_recording(self) -> None:
        """Signal the rollout saver to start recording."""
        if not self._infer_rollout_started:
            return
        from utils.lerobot_rollout import start_recording

        start_recording(self)
        return

    def _signal_stop_recording(self) -> None:
        """Signal the rollout saver to stop recording."""
        if not self._infer_rollout_started:
            return
        from utils.lerobot_rollout import stop_recording

        stop_recording(self)
        return

    def _setup_keyboard_listener(self) -> None:
        """Setup non-blocking keyboard listener for episode control."""
        if not self.episode_control_enabled:
            return
        try:
            from pynput import keyboard as pynput_keyboard

            def on_press(key: Any) -> None:
                try:
                    if hasattr(key, "char") and key.char:
                        char = key.char.lower()
                        if char == self.episode_success_key:
                            self._episode_result_queue.put("success")
                            logger.info("PolicyConnector: 'S' pressed - marking episode as SUCCESS")
                        elif char == self.episode_failure_key:
                            self._episode_result_queue.put("failure")
                            logger.info("PolicyConnector: 'F' pressed - marking episode as FAILURE")
                        elif char == self.episode_discard_key:
                            self._episode_result_queue.put("discard")
                            logger.info("PolicyConnector: 'D' pressed - discarding episode data")
                except AttributeError:
                    pass

            self._keyboard_listener = pynput_keyboard.Listener(on_press=on_press)
            self._keyboard_listener.start()
            logger.info(
                f"PolicyConnector: Keyboard listener started. Press '{self.episode_success_key.upper()}' for success, '{self.episode_failure_key.upper()}' for failure, '{self.episode_discard_key.upper()}' to discard episode data."
            )
        except Exception as e:
            logger.error(f"PolicyConnector: Failed to setup keyboard listener: {e}")
            self._keyboard_listener = None

    def _stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener."""
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
                self._keyboard_listener = None
                logger.info("PolicyConnector: Keyboard listener stopped")
            except Exception as e:
                logger.warning(f"PolicyConnector: Failed to stop keyboard listener: {e}")

    def _write_episode_result(self, result: str) -> None:
        """
        Write episode result to signal file for data saver to read.

        Args:
            result: 'success', 'failure', or 'discard'
        """
        try:
            episode_result_signal = Path(self.signal_files.get("episode_result"))
            episode_result_signal.parent.mkdir(parents=True, exist_ok=True)
            episode_result_signal.write_text(result)
            logger.info(f"PolicyConnector: Wrote episode result '{result}' to signal file")
        except Exception as e:
            logger.error(f"PolicyConnector: Failed to write episode result: {e}")

    def _signal_policy_runner_reset(self) -> None:
        """Signal the policy runner to clear its action cache and SHM buffers."""
        try:
            policy_runner_reset_signal = Path(self.signal_files.get("policy_runner_reset_command"))
            policy_runner_reset_signal.parent.mkdir(parents=True, exist_ok=True)
            policy_runner_reset_signal.touch()
            logger.info("PolicyConnector: Signaled policy runner to reset")
        except Exception as e:
            logger.warning(f"PolicyConnector: Failed to signal policy runner reset: {e}")

    def _signal_teleoperator_reset(self) -> None:
        """
        Signal the teleoperator to switch to policy control mode.

        This is called at the start of a new episode to ensure policy control
        is active regardless of how the previous episode ended (human or policy).
        """
        try:
            teleoperator_reset_signal = Path(self.signal_files.get("teleoperator_reset_command"))
            teleoperator_reset_signal.parent.mkdir(parents=True, exist_ok=True)
            teleoperator_reset_signal.touch()
            logger.info("PolicyConnector: Signaled teleoperator to reset to policy control mode")
        except Exception as e:
            logger.warning(f"PolicyConnector: Failed to signal teleoperator reset: {e}")

    def _read_robot_home_index(self) -> int:
        """
        Read robot home index from shared file written by device_starter.py.

        Returns:
            Home preset index (default 0 if file not found or invalid)
        """
        try:
            robot_home_index_file = Path(self.signal_files.get("robot_home_index"))
            if robot_home_index_file.exists():
                content = robot_home_index_file.read_text().strip()
                if content:
                    home_index = int(content)
                    logger.debug(f"PolicyConnector: Read robot home index {home_index} from shared file")
                    return home_index
        except Exception as e:
            logger.warning(f"PolicyConnector: Failed to read robot home index: {e}")
        logger.debug("PolicyConnector: Using default robot home index 0")
        return 0

    def _signal_robot_home(self, home_index: int = 0) -> None:
        """
        Signal all robots in controls config to return to home pose.

        Args:
            home_index: Home preset index to use (default 0)
        """
        if not self.controls_config:
            logger.warning("PolicyConnector: No controls config, cannot signal robot home")
            return
        for control in self.controls_config:
            shm_name = control.get("shm", "")
            if shm_name.endswith("_control"):
                device_full_name = shm_name.replace("_control", "")
            else:
                device_full_name = shm_name
            parts = device_full_name.rsplit("_", 1)
            if len(parts) == 2:
                device_name, device_id = parts
            else:
                device_name = device_full_name
                device_id = "0"
            home_command_file = Path(
                self.signal_files.get("robot_home_command_pattern").format(device_name=device_name, device_id=device_id)
            )
            try:
                home_command_file.parent.mkdir(parents=True, exist_ok=True)
                home_command_file.write_text(str(home_index))
                logger.info(
                    f"PolicyConnector: Signaled {device_name}_{device_id} to return to home pose (index={home_index})"
                )
            except Exception as e:
                logger.error(f"PolicyConnector: Failed to signal robot home for {device_name}_{device_id}: {e}")
        for reset_device in self.reset_config:
            device_name = reset_device.get("device_name", "")
            device_id = reset_device.get("device_id", 0)
            home_index = reset_device.get("home_index", 0)
            reset_command_file = Path(
                self.signal_files.get("robot_home_command_pattern").format(device_name=device_name, device_id=device_id)
            )
            try:
                reset_command_file.parent.mkdir(parents=True, exist_ok=True)
                reset_command_file.write_text(str(home_index))
                logger.info(f"PolicyConnector: Signaled {device_name}_{device_id} to reset")
            except Exception as e:
                logger.error(f"PolicyConnector: Failed to signal robot reset for {device_name}_{device_id}: {e}")

    def _signal_and_wait_for_robot_halt(self) -> None:
        """Signal the robot to halt and wait for the robot to report its "do halt state"."""
        try:
            for halt_device in self.halt_and_home_cfg:
                device_name = halt_device.get("device_name", "")
                device_id = halt_device.get("device_id", 0)
                robot_halt_signal = Path(
                    self.signal_files.get("robot_halt_command_pattern").format(
                        device_name=device_name, device_id=device_id
                    )
                )
                robot_halt_signal.parent.mkdir(parents=True, exist_ok=True)
                robot_halt_signal.touch()
                logger.info(f"PolicyConnector: Signaled robot to halt, device: {device_name}_{device_id}")
        except Exception as e:
            logger.error(f"PolicyConnector: Failed to signal robot halt: {e}")
            return
        all_robot_halt_reply_files_exist = False
        while not all_robot_halt_reply_files_exist:
            all_robot_halt_reply_files_exist_list = []
            for reply_device in self.halt_and_home_cfg:
                device_name = reply_device.get("device_name", "")
                device_id = reply_device.get("device_id", 0)
                robot_halt_reply_file = Path(
                    self.signal_files.get("robot_halt_reply_pattern").format(
                        device_name=device_name, device_id=device_id
                    )
                )
                if robot_halt_reply_file.exists():
                    logger.info(f"PolicyConnector: {robot_halt_reply_file} exists")
                    now_robot_halt_reply_file_exists = True
                else:
                    logger.info(f"PolicyConnector: {robot_halt_reply_file} does not exist")
                    now_robot_halt_reply_file_exists = False
                all_robot_halt_reply_files_exist_list.append(now_robot_halt_reply_file_exists)
            all_robot_halt_reply_files_exist = all(all_robot_halt_reply_files_exist_list)
            if all_robot_halt_reply_files_exist:
                break
            else:
                time.sleep(0.1)
        logger.info("PolicyConnector: Robot reported its 'do halt state'")
        for reply_device in self.halt_and_home_cfg:
            device_name = reply_device.get("device_name", "")
            device_id = reply_device.get("device_id", 0)
            robot_halt_reply_file = Path(
                self.signal_files.get("robot_halt_reply_pattern").format(device_name=device_name, device_id=device_id)
            )
            robot_halt_reply_file.unlink()
            logger.info(f"PolicyConnector: Consumed robot halt reply file for {device_name}_{device_id}")

    def _handle_episode_transition(self, result: str) -> None:
        """
        Handle episode end: save result, reset, return home, and optionally start new episode.

        Args:
            result: 'success', 'failure', or 'discard'
        """
        if self._episode_in_transition:
            logger.warning("PolicyConnector: Episode transition already in progress, ignoring")
            return
        self._episode_in_transition = True
        logger.info(f"PolicyConnector: ========== EPISODE {self._episode_count} ENDED: {result.upper()} ==========")
        try:
            self._write_episode_result(result)
            self._signal_stop_recording()
            self._increment_current_policy_episode_count(result)
            self._set_rollout_display_phase("halting")
            self._signal_and_wait_for_robot_halt()
            logger.info("PolicyConnector: Robot halted, sleeping for 2 seconds to avoid state misreading")
            time.sleep(2)
            self.inference_enabled = False
            self._signal_policy_runner_reset()
            self._signal_teleoperator_reset()
            time.sleep(0.1)
            home_index = self._read_robot_home_index()
            self._set_rollout_display_phase("homing")
            self._signal_robot_home(home_index=home_index)
            logger.info(f"PolicyConnector: Waiting {self.episode_home_wait_time}s for robot to return home...")
            time.sleep(self.episode_home_wait_time)
            self._set_rollout_display_phase("idle")
            self._maybe_switch_or_wait_for_policy_budget()
            if result != "discard":
                self._episode_count += 1
            if self.episode_auto_restart:
                logger.info(f"PolicyConnector: ========== STARTING EPISODE {self._episode_count} ==========")
                self._signal_start_recording()
                self.inference_enabled = True
            else:
                logger.info("PolicyConnector: Auto-restart disabled. Waiting for manual restart...")
        except Exception as e:
            self._set_rollout_display_phase("ERROR")
            logger.error(f"PolicyConnector: Error during episode transition: {e}")
        finally:
            self._episode_in_transition = False

    def _check_episode_end_request(self) -> Optional[str]:
        """
        Check if there's an episode end request in the queue.

        Returns:
            'success', 'failure', 'discard', or None if no request
        """
        try:
            result = self._episode_result_queue.get_nowait()
            return result
        except queue.Empty:
            return None

    def _requires_history(self) -> bool:
        obs_horizon = getattr(self.policy, "obs_horizon", 1)
        return (
            isinstance(obs_horizon, int)
            and obs_horizon > 1
            and hasattr(self.policy, "obs_time_interval")
            and hasattr(self.policy, "downsample")
        )

    def _collect_observation_history(
        self, latest_obs: Dict[str, np.ndarray], latest_timestamp: int, history_with_value: bool = False
    ) -> Optional[Dict[str, List[np.ndarray]]]:
        with self.summary_shm_lock:
            return self._collect_observation_history_unlocked(
                latest_obs, latest_timestamp, history_with_value=history_with_value
            )

    def _collect_observation_history_unlocked(
        self, latest_obs: Dict[str, np.ndarray], latest_timestamp: int, history_with_value: bool = False
    ) -> Optional[Dict[str, List[np.ndarray]]]:
        """
        Collect observation history from SHM while holding the summary SHM lock.
        """
        obs_horizon = getattr(self.policy, "obs_horizon", 1)
        if not isinstance(obs_horizon, int) or obs_horizon <= 1:
            return {k: [np.array(v, copy=True)] for k, v in latest_obs.items()}
        interval_sec = float(getattr(self.policy, "obs_time_interval", 0.0))
        downsample = int(getattr(self.policy, "downsample", 1))
        if interval_sec <= 0.0 or downsample <= 0:
            logger.warning(
                "PolicyConnector: Invalid observation interval configuration; falling back to single observation"
            )
            return {k: [np.array(v, copy=True)] for k, v in latest_obs.items()}
        dt_ns = int(interval_sec * downsample * 1000000000.0)
        value_prediction_key = "value_prediction" if history_with_value else None
        latest_obs = {key: np.array(value, copy=True) for key, value in latest_obs.items()}
        obs_list: List[Dict[str, np.ndarray]] = [latest_obs]
        if history_with_value:
            obs_list[0][value_prediction_key] = np.zeros(1, dtype=np.float32)
        is_pad_list: Optional[List[bool]] = [False]
        history_obs_times = [latest_timestamp - (i + 1) * dt_ns for i in range(obs_horizon - 1)]
        for target_timestamp in history_obs_times:
            past_frame_invalid = False
            past_value_invalid = False
            past_obs = None
            past_value = None
            past_frame = self.find_frame_by_timestamp(target_timestamp, tolerance_ms=50.0)
            if past_frame is None:
                logger.warning(
                    "PolicyConnector: the requested history frame is not found, too old, or the shm is empty, we pad it"
                )
                past_frame_invalid = True
            if history_with_value:
                past_value = self.find_value_by_timestamp_with_history_too_old_detection(
                    target_timestamp, tolerance_ms=50.0
                )
                if past_value is None:
                    logger.warning("PolicyConnector: past_value is not found, too old, or the shm is empty, we pad it")
                    past_value_invalid = True
            if past_frame_invalid or past_value_invalid:
                past_obs = {key: np.array(value, copy=True) for key, value in obs_list[-1].items()}
                if history_with_value:
                    past_obs[value_prediction_key] = np.zeros(1, dtype=np.float32)
                obs_list.append(past_obs)
                is_pad_list.append(True)
            else:
                past_obs = {key: np.array(value[1], copy=True) for key, value in past_frame["data"].items()}
                if history_with_value:
                    assert isinstance(past_value, np.ndarray), (
                        f"past_value must be a numpy array, got {type(past_value)}"
                    )
                    past_obs[value_prediction_key] = np.array(past_value, copy=True)
                obs_list.append(past_obs)
                is_pad_list.append(False)
        if len(obs_list) != obs_horizon:
            logger.warning(
                f"PolicyConnector: Requested history is not collected; obs_list: {len(obs_list)}, obs_horizon: {obs_horizon}"
            )
            return None
        if is_pad_list is None or len(is_pad_list) != obs_horizon:
            logger.warning(
                f"PolicyConnector: Requested stride history, but is_pad_list: {(len(is_pad_list) if is_pad_list is not None else None)}, obs_horizon: {obs_horizon}"
            )
            return None
        obs_list.reverse()
        if is_pad_list is not None:
            is_pad_list.reverse()
        obs_batch: Dict[str, List[np.ndarray]] = {}
        for key in latest_obs.keys():
            obs_batch[key] = [step[key] for step in obs_list]
            obs_batch[f"{key}_is_pad"] = [np.asarray(is_pad, dtype=bool) for is_pad in is_pad_list]
        if history_with_value:
            obs_batch[value_prediction_key] = [step[value_prediction_key] for step in obs_list]
            obs_batch[f"{value_prediction_key}_is_pad"] = [np.asarray(is_pad, dtype=bool) for is_pad in is_pad_list]
        return obs_batch

    def _prepare_policy_input(
        self, latest_obs: Dict[str, np.ndarray], latest_timestamp: int, history_with_value: bool = False
    ) -> Optional[Dict[str, Any]]:
        if not latest_obs:
            return None
        if self._requires_history():
            history = self._collect_observation_history(
                latest_obs, latest_timestamp, history_with_value=history_with_value
            )
            if history is None:
                return None
            return history
        return latest_obs

    def _write_action_chunk_dict(
        self, action_chunk_dict: Dict[str, np.ndarray], observation_timestamp: int = -1
    ) -> None:
        """Write dict-based action chunking to policy SHM."""
        if not self.policy_shm:
            logger.warning("PolicyConnector: policy_shm is None, cannot write action chunks")
            return
        try:
            buf = self.policy_shm.buf
            timestamp_ns = time.time_ns()
            device_count = len(self.policy_devices)
            manager_header = pack_policy_header(device_count, timestamp_ns)
            buf[:POLICY_HEADER_SIZE] = manager_header
            frame_sizes = [device["frame_size"] for device in self.policy_devices]
            for i, device_info in enumerate(self.policy_devices):
                device_name = device_info["device_name"]
                expected_shape = device_info["shape"]
                if device_name not in action_chunk_dict:
                    logger.warning(f"No action chunk for device: {device_name}")
                    continue
                action_chunk = action_chunk_dict[device_name]
                if len(action_chunk.shape) == 1:
                    action_chunk = action_chunk.reshape(1, -1)
                if action_chunk.shape != tuple(expected_shape):
                    logger.error(
                        f"Shape mismatch for {device_name}: got {action_chunk.shape}, expected {expected_shape}"
                    )
                    continue
                data_offset = get_policy_data_offset(i, device_count, frame_sizes)
                frame_header = pack_frame_header(observation_timestamp)
                buf[data_offset : data_offset + FRAME_HEADER_SIZE] = frame_header
                data_start = data_offset + FRAME_HEADER_SIZE
                chunk_bytes = action_chunk.astype(np.float64).tobytes()
                data_end = data_start + len(chunk_bytes)
                buf[data_start:data_end] = chunk_bytes
            logger.debug(f"Wrote action chunks to policy SHM: devices={list(action_chunk_dict.keys())}")
        except (OSError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to write action chunk dict (SHM may have been recreated): {e}")
            logger.info("PolicyConnector: Attempting to reconnect to policy SHM...")
            try:
                self._connect_policy_shm()
                logger.info("PolicyConnector: Successfully reconnected to policy SHM")
            except Exception as reconnect_error:
                logger.error(f"PolicyConnector: Failed to reconnect to policy SHM: {reconnect_error}")
        except Exception as e:
            logger.error(f"Failed to write action chunk dict: {e}")

    def _assemble_observation_dict(self) -> Optional[Dict[str, np.ndarray]]:
        """Assemble observation dictionary using atomic read operation."""
        try:
            all_data = self.read_all_device_data()
            if all_data is None:
                return (None, None)
            obs_dict = {}
            all_obs_time = []
            for device_name, (timestamp, data) in all_data.items():
                obs_dict[device_name] = data
                if self.master_device is None:
                    all_obs_time.append(timestamp)
            if self.master_device is None:
                obs_time = int(np.array(all_obs_time).mean().item())
            else:
                obs_time = all_data[self.master_device][0]
            return (obs_dict, obs_time)
        except Exception as e:
            logger.error(f"Error assembling observation dict: {e}")
            return (None, None)

    def _signal_handler(self, signum, frame) -> None:
        self.running = False
        self.inference_enabled = False
        if not getattr(self, "_lerobot_shutting_down", False):
            self._lerobot_shutting_down = True
            raise KeyboardInterrupt

    def run(self) -> None:
        self._lerobot_shutting_down = False
        try:
            self._run_policy()
        except KeyboardInterrupt:
            logger.info("Rollout acquisition interrupted")
        finally:
            self._lerobot_shutting_down = True
            self.running = False
            self.inference_enabled = False
            if getattr(self, "infer_rollout_process", None) is not None:
                self._stop_infer_rollout()
            self.disconnect()
        if getattr(self, "_lerobot_saver_exit_code", 0):
            raise RuntimeError("LeRobot conversion did not finish successfully; inspect the retained caches and logs")

    def _run_policy(self) -> None:
        self.running = True
        self._maybe_switch_or_wait_for_policy_budget()
        if not self.running:
            return
        if self._should_start_infer_rollout():
            self._start_infer_rollout()
        if not self.connect():
            logger.error("PolicyConnector failed to connect to summary SHM")
            self.running = False
            return
        if self.inference_enabled:
            logger.info("PolicyConnector: Inference is enabled; policy inference will begin now.")
        else:
            logger.info("PolicyConnector: Inference is disabled; waiting for inference to be enabled.")
        self._connect_policy_shm()
        cached_policy_input: Optional[Dict[str, Any]] = None
        initial_obs_dict, initial_obs_time = self._assemble_observation_dict()
        if initial_obs_dict and initial_obs_time is not None:
            cached_policy_input = self._prepare_policy_input(
                initial_obs_dict, initial_obs_time, history_with_value=self.history_with_value
            )
            if cached_policy_input is None:
                logger.warning("PolicyConnector: Failed to prepare initial policy input for warm-up")
        else:
            logger.warning("PolicyConnector: Failed to assemble initial observation for warm-up")
        enable_rtc = getattr(self.policy, "enable_rtc", False)
        if enable_rtc and hasattr(self.policy, "warm_up"):
            try:
                logger.info("PolicyConnector: Starting policy warm-up with RTC...")
                self.policy.warm_up(self.action_configs, num_observations=10, sample_observation=cached_policy_input)
                logger.info("PolicyConnector: Policy warm-up completed")
            except Exception as e:
                logger.warning(f"PolicyConnector: Warm-up failed, continuing anyway: {e}")
        logger.info(f"PolicyConnector running with {self.fps} Hz inference rate. Press Ctrl+C to stop...")
        logger.info(f"PolicyConnector: enable_value_prediction: {self.enable_value_prediction}")
        logger.info(f"PolicyConnector: policy has predict_value: {hasattr(self.policy, 'predict_value')}")
        if self.enable_value_prediction and hasattr(self.policy, "predict_value"):
            self.value_prediction_running = True
            self.value_prediction_thread = threading.Thread(
                target=self._value_prediction_worker, daemon=True, name="ValuePredictionThread"
            )
            self.value_prediction_thread.start()
            logger.info("PolicyConnector: Started value prediction thread")
        if self.episode_control_enabled:
            self._setup_keyboard_listener()
            logger.info(f"PolicyConnector: Episode control enabled. Episode {self._episode_count} starting...")
        wait_for_next_iteration = precise_loop_timing(self.update_interval)
        iteration_count = 0
        last_inference_state = self.inference_enabled
        first_action_written = False
        try:
            while self.running:
                if self._check_reset_signal():
                    logger.info("PolicyConnector: Reset signal detected, reconnecting all SHMs...")
                    self._reconnect_all_shms()
                    if hasattr(self.policy, "reset"):
                        with self._policy_lock:
                            self.policy.reset()
                        logger.info("PolicyConnector: Reset policy's RTC state")
                    self._signal_policy_runner_reset()
                    logger.info("PolicyConnector: Signaled policy runner to reset action cache")
                    iteration_count = 0
                    logger.info("PolicyConnector: Reset iteration_count to 0")
                if self.episode_control_enabled and (not self._episode_in_transition):
                    episode_result = self._check_episode_end_request()
                    if episode_result is not None:
                        self._handle_episode_transition(episode_result)
                        first_action_written = False
                        wait_for_next_iteration()
                        continue
                if self.inference_enabled != last_inference_state:
                    if self.inference_enabled:
                        logger.info("PolicyConnector: Inference ENABLED - model will perform predictions")
                    else:
                        logger.info(
                            "PolicyConnector: Inference DISABLED - model remains loaded but predictions are paused"
                        )
                        if self._infer_rollout_started and first_action_written:
                            self._signal_stop_recording()
                            first_action_written = False
                    last_inference_state = self.inference_enabled
                obs_dict, obs_time = self._assemble_observation_dict()
                if not obs_dict or obs_time is None:
                    wait_for_next_iteration()
                    continue
                if not self.inference_enabled:
                    wait_for_next_iteration()
                    continue
                policy_input = self._prepare_policy_input(obs_dict, obs_time, self.history_with_value)
                if policy_input is None:
                    wait_for_next_iteration()
                    continue
                enable_rtc = getattr(self.policy, "enable_rtc", False)
                if enable_rtc:
                    action_index_value = self._read_policy_action_index()
                    action_index = max(0, int(action_index_value)) if action_index_value is not None else None
                    if iteration_count == 0:
                        action_index = None
                else:
                    action_index = None
                human_rtc_prefix = self._prepare_human_rtc_prefix()
                if human_rtc_prefix is not None:
                    action_index = 0
                with self._policy_lock:
                    if human_rtc_prefix is not None and hasattr(self.policy, "seed_human_rtc_prefix"):
                        self.policy.seed_human_rtc_prefix(human_rtc_prefix)
                    action_chunk_dict = self.policy.predict(
                        policy_input,
                        action_configs=self.action_configs,
                        chunk_length=self.chunk_length,
                        action_index=action_index,
                    )
                for device_name in action_chunk_dict:
                    action_chunk_dict[device_name] = action_chunk_dict[device_name][: self.chunk_length, :]
                merged_chunk = self._merge_action_chunks(action_chunk_dict)
                if merged_chunk is not None:
                    self.action_chunk_shm.write(merged_chunk, obs_time)
                if self._infer_rollout_started and (not first_action_written):
                    logger.info("PolicyConnector: First action being written - starting rollout recording NOW")
                    self._signal_start_recording()
                    first_action_written = True
                self._write_action_chunk_dict(action_chunk_dict, obs_time)
                iteration_count += 1
                wait_for_next_iteration()
        except KeyboardInterrupt:
            logger.info("Stopping PolicyConnector...")
        finally:
            self._lerobot_shutting_down = True
            self.running = False
            self.inference_enabled = False
            if self.episode_control_enabled:
                self._stop_keyboard_listener()
            if self.value_prediction_running:
                self.value_prediction_running = False
                if self.value_prediction_thread is not None:
                    try:
                        self.value_prediction_thread.join(timeout=2.0)
                        if self.value_prediction_thread.is_alive():
                            logger.warning("Value prediction thread did not stop gracefully")
                    except Exception as e:
                        logger.warning(f"Error stopping value prediction thread: {e}")
            if self._infer_rollout_started:
                try:
                    pass
                except (RuntimeError, TimeoutError) as exc:
                    logger.error(f"Recorder stop acknowledgement failed: {exc}")
                finally:
                    self._stop_infer_rollout()
            self.stop()
            if self.policy_shm:
                try:
                    self.policy_shm.close()
                except Exception:
                    pass
            self.value_shm.cleanup()
            self.action_chunk_shm.cleanup()
            self._close_human_rtc_shms()
