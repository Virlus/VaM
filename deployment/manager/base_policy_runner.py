import signal
from utils.config_parser import load_config, ensure_config_dict

"""
Policy Runner - reads action chunking from policy and dispatches to robot control SHMs.

Author: Jun Lv, Zheng Wang
"""
import time
import struct
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from pathlib import Path
from utils.shm_utils import (
    pack_device_header,
    pack_buffer_header,
    pack_frame_header,
    unpack_device_header,
    unpack_buffer_header,
    unpack_frame_header,
    DEVICE_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    pack_policy_header,
    POLICY_HEADER_SIZE,
    unpack_policy_header,
    calculate_policy_shm_size,
    get_device_info_offset,
    get_policy_data_offset,
    get_policy_action_index_offset,
    create_policy_shm,
    connect_to_shared_memory,
    ACTION_INDEX_SIZE,
    ACTION_INDEX_FORMAT,
)
from utils.logger_config import logger
from manager.utils import load_chunk_manager


class BasePolicyRunner:
    """
    Policy Runner for reading action chunking and dispatching to robot control SHMs.

    This class supports device-specific action delays to synchronize execution across
    devices with different latencies. Actions are queued per device, and devices with
    larger delays have their actions held longer before dispatch.

    Args:
        config: Configuration dictionary (alternative to config_path)
        config_path: Path to configuration YAML file
        robot_control_shms: List of robot control SHM configurations
        execution_fps: Execution frequency in Hz
        chunk_length: Length of action chunks
        chunk_manager: Chunk manager configuration
        device_delays: Dict mapping device names to delay frames. Devices with higher
                      latency should have smaller delays, while devices with lower
                      latency should have larger delays to ensure synchronized execution.
                      Example: {'I2RTRobot_1': 0, 'I2RTRobot_2': 10}
                      This means I2RTRobot_2 actions will be delayed by 10 frames.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_path: str = "config.yaml",
        robot_control_shms: Optional[List[Dict[str, Any]]] = None,
        execution_fps: float = 100.0,
        chunk_length: int = 10,
        signal_files: Optional[Dict[str, str]] = None,
        chunk_manager: Optional[Dict[str, Any]] = None,
        device_delays: Optional[Dict[str, int]] = None,
        device_dimension_delays: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        self.config_path = config_path
        self.robot_control_shms = robot_control_shms or []
        self.execution_fps = execution_fps
        self.chunk_length = chunk_length
        self.running = False
        self.policy_shm = None
        self.policy_device_count: int = 0
        self.policy_frame_sizes: List[int] = []
        self.robot_states = []
        self.current_step = 0
        self.current_action_chunk_dict = None
        self.last_read_timestamp = 0
        self.last_reset_timestamp = 0
        self.chunk_manager = load_chunk_manager(chunk_manager or {}, execution_fps=self.execution_fps)
        self.retry_connect_secs: float = 0.5
        self.connect_timeout_secs: float = 60.0
        self.device_delays = device_delays or {}
        self.device_action_queues: Dict[str, deque] = {}
        self.device_dimension_delays: Dict[str, List[Tuple[List[int], int]]] = {}
        self.device_first_actions: Dict[str, Optional[np.ndarray]] = {}
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")
        self._policy_runner_reset_command_path = signal_files.get("policy_runner_reset_command")
        if self._policy_runner_reset_command_path is None:
            raise ValueError("policy_runner_reset_command_path is required in signal_files")
        if device_dimension_delays:
            for device_name, delay_configs in device_dimension_delays.items():
                parsed_delays = []
                for delay_config in delay_configs:
                    indices = delay_config.get("indices", [])
                    delay = delay_config.get("delay", 0)
                    parsed_delays.append((indices, delay))
                self.device_dimension_delays[device_name] = parsed_delays

        if config is None:
            self.config = load_config(self.config_path)
        else:
            self.config = ensure_config_dict(config)
        self.enable_rtc: bool = self._get_enable_rtc_from_config(self.config)

        # Initialize device action queues based on device delays
        self._initialize_device_queues()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if self.device_delays:
            logger.info(f"Device delays configured: {self.device_delays}")
        logger.info(f"RTC enabled: {self.enable_rtc}")

    def _get_enable_rtc_from_config(self, config: Dict[str, Any]) -> bool:
        """Extract enable_rtc from policy params."""
        policy_config = config.get("policy", {})
        params_config = policy_config.get("params", {})
        return bool(params_config.get("enable_rtc", False))

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        self.stop()

    def _initialize_device_queues(self) -> None:
        """
        Initialize action queues for each device based on configured delays.

        For devices with dimension-level delays, we create separate queues for each
        dimension group. For devices with only whole-action delays, we create a single queue.
        """
        for robot_cfg in self.robot_control_shms:
            shm_name = robot_cfg.get("shm", "")
            device_name = shm_name.replace("_control", "") if shm_name.endswith("_control") else shm_name
            self.device_first_actions[device_name] = None
            if device_name in self.device_dimension_delays:
                dimension_queues = {}
                for idx, (indices, delay) in enumerate(self.device_dimension_delays[device_name]):
                    dimension_queues[idx] = deque()
                dimension_queues["default"] = deque()
                self.device_action_queues[device_name] = dimension_queues
                logger.info(
                    f"Initialized dimension-level queues for {device_name}: {len(dimension_queues) - 1} groups + default"
                )
            else:
                self.device_action_queues[device_name] = deque()
        logger.info(f"Initialized action queues for devices: {list(self.device_action_queues.keys())}")

    def start(self) -> bool:
        """Start the action executor."""
        if not self.connect():
            logger.error("Failed to connect, cannot start executor")
            return False
        logger.info("Starting ActionExecutor")
        self.running = True
        try:
            while self.running:
                self._update_execution()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        return True

    def stop(self) -> None:
        """Stop the action executor."""
        self.running = False
        for robot_state in self.robot_states:
            robot_shm = robot_state.get("shm")
            robot_state["shm"] = None
            try:
                if robot_shm is not None:
                    robot_shm.close()
            except Exception:
                pass
        policy_shm = self.policy_shm
        self.policy_shm = None
        if policy_shm is not None:
            try:
                policy_shm.close()
            except Exception:
                pass
        logger.info("ActionExecutor stopped")

    def connect(self) -> bool:
        """Connect to robot control SHMs and create policy SHM."""
        try:
            for robot_cfg in self.robot_control_shms:
                robot_state = self._connect_robot_control(robot_cfg)
                if robot_state is None:
                    logger.error(f"Failed to connect to robot control SHM: {robot_cfg['shm']}")
                    return False
                self.robot_states.append(robot_state)
            self._create_policy_shm()
            if hasattr(self.chunk_manager, "last_read_time"):
                self.chunk_manager.last_read_time = -1
                logger.info("ActionExecutor: Reset chunk manager state after restart")
            logger.info("ActionExecutor connected successfully")
            return True
        except Exception as e:
            logger.error(f"ActionExecutor failed to connect: {e}")
            return False

    def _connect_robot_control(self, robot_cfg: Dict) -> Optional[Dict]:
        """Connect to a single robot control SHM using unified interface (read-write for policy runner)."""
        control_shm_name = robot_cfg.get("shm")
        if not control_shm_name:
            return None
        start_time = time.time()
        while True:
            try:
                robot_shm = connect_to_shared_memory(control_shm_name, read_only=False)
                buf = robot_shm.buf
                device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
                buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
                action_dim = robot_cfg.get("shape", [1])[0]
                control_frame_size = device_header["frame_size"]
                robot_state = {
                    "device_info": robot_cfg,
                    "shm": robot_shm,
                    "action_dim": action_dim,
                    "frame_size": control_frame_size,
                    "buffer_size": buffer_header["buffer_size"],
                }
                logger.info(f"Connected to robot control SHM: {control_shm_name} (READ-WRITE)")
                return robot_state
            except Exception as e:
                _last_error = e
                if time.time() - start_time < self.connect_timeout_secs:
                    logger.info(
                        f"ActionExecutor: Waiting for control SHM '{control_shm_name}'... retry in {self.retry_connect_secs}s"
                    )
                    time.sleep(self.retry_connect_secs)
                    continue
                logger.error(f"Failed to connect to robot control SHM after retries: {e}")
                return None

    def _create_policy_shm(self) -> None:
        """Create or connect to policy SHM for PolicyConnector to write action chunking using unified interface."""
        try:
            device_count = len(self.robot_states)
            frame_sizes = []
            for robot_state in self.robot_states:
                action_dim = robot_state["action_dim"]
                chunk_data_size = self.chunk_length * action_dim * 8
                frame_size = FRAME_HEADER_SIZE + chunk_data_size
                frame_sizes.append(frame_size)
            self.policy_device_count = device_count
            self.policy_frame_sizes = frame_sizes
            self.policy_shm = create_policy_shm(device_count, frame_sizes)
            _total_size = calculate_policy_shm_size(device_count, frame_sizes)
            logger.info(
                f"Policy header: {POLICY_HEADER_SIZE} bytes, Device headers: {device_count * DEVICE_HEADER_SIZE} bytes, Data: {sum(frame_sizes)} bytes, action_index: {ACTION_INDEX_SIZE} bytes (tail)"
            )
            self._initialize_policy_header()
            self._write_action_index(0)
        except Exception as e:
            logger.error(f"Failed to create/connect policy SHM: {e}")
            raise

    def _initialize_policy_header(self) -> None:
        """Initialize policy SHM header with manager and device headers."""
        device_count = len(self.robot_states)
        manager_header = pack_policy_header(device_count, 0)
        self.policy_shm.buf[:POLICY_HEADER_SIZE] = manager_header
        for i, robot_state in enumerate(self.robot_states):
            robot_cfg = robot_state["device_info"]
            shm_name = robot_cfg.get("shm", "")
            device_name = shm_name.replace("_control", "") if shm_name.endswith("_control") else shm_name
            action_dim = robot_state["action_dim"]
            chunk_data_size = self.chunk_length * action_dim * 8
            frame_size = FRAME_HEADER_SIZE + chunk_data_size
            device_header = pack_device_header(
                device_type=device_name,
                device_id=0,
                fps=self.execution_fps,
                data_dtype="float64",
                shape=[self.chunk_length, action_dim],
                frame_size=frame_size,
                hardware_latency_ms=0.0,
            )
            device_offset = get_device_info_offset(i)
            self.policy_shm.buf[device_offset : device_offset + DEVICE_HEADER_SIZE] = device_header

    def _write_action_index(self, action_index: int) -> None:
        if not self.policy_shm or not self.policy_frame_sizes:
            return
        try:
            offset = get_policy_action_index_offset(self.policy_device_count, self.policy_frame_sizes)
            action_index_bytes = struct.pack(ACTION_INDEX_FORMAT, int(action_index))
            self.policy_shm.buf[offset : offset + ACTION_INDEX_SIZE] = action_index_bytes
        except Exception as exc:
            logger.error(f"Failed to write action_index to policy SHM: {exc}")

    def _update_execution(self) -> None:
        """
        Update execution loop - read action chunk, manage delays, and dispatch actions.

        This method handles the complete execution pipeline:
        1. Read new action chunks from policy SHM
        2. Get next action from chunk manager
        3. Enqueue actions to per-device delay queues
        4. Dequeue actions that have reached required delay
        5. Dispatch ready actions to robot control SHMs
        """
        start_time = time.time()
        try:
            self._check_and_handle_reset_command()
            chunk_result = self._read_latest_policy_frame()
            if chunk_result is not None:
                latest_chunk_dict, timestamp_ns = chunk_result
                if self.last_reset_timestamp > 0 and timestamp_ns <= self.last_reset_timestamp:
                    logger.debug(
                        f"Skipping stale action chunk (obs_ts={timestamp_ns} <= reset_ts={self.last_reset_timestamp})"
                    )
                elif self.chunk_manager.is_empty() or self.chunk_manager.is_new_chunk(timestamp_ns):
                    self.chunk_manager.put(latest_chunk_dict, timestamp_ns)
            if not self.chunk_manager.is_empty():
                action_dict = self.chunk_manager.get()
                if action_dict is not None:
                    if self.enable_rtc and (not hasattr(self.chunk_manager, "get_original_current_step")):
                        raise RuntimeError(
                            "RTC enabled but chunk_manager lacks get_original_current_step; cannot write action_index."
                        )
                    if hasattr(self.chunk_manager, "get_original_current_step"):
                        action_index = self.chunk_manager.get_original_current_step()
                        if action_index is not None:
                            self._write_action_index(action_index)
                    for device_name, action in action_dict.items():
                        if device_name in self.device_action_queues:
                            self._enqueue_action_with_delays(device_name, action)
                    ready_actions = {}
                    for device_name in self.device_action_queues.keys():
                        ready_action = self._dequeue_action_with_delays(device_name)
                        if ready_action is not None:
                            ready_actions[device_name] = ready_action
                    if ready_actions:
                        self._dispatch_action_to_robots(ready_actions)
        except Exception as e:
            logger.error(f"Error in execution update: {e}")
        elapsed = time.time() - start_time
        target_interval = 1.0 / self.execution_fps
        sleep_time = max(0.0, target_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _check_and_handle_reset_command(self) -> None:
        """Check for reset command file and clear action cache and SHM if present."""
        try:
            reset_command_file = Path(self._policy_runner_reset_command_path)
            if reset_command_file.exists():
                try:
                    reset_command_file.unlink()
                    self.last_reset_timestamp = time.time_ns()
                    logger.info(f"Reset timestamp set to {self.last_reset_timestamp}")
                    if hasattr(self.chunk_manager, "reset"):
                        self.chunk_manager.reset()
                    else:
                        self.chunk_manager.current_action_chunk_dict = None
                        self.chunk_manager.current_step = 0
                        if hasattr(self.chunk_manager, "last_read_time"):
                            self.chunk_manager.last_read_time = 0
                    logger.debug(
                        f"Before clearing all device action queues, the chunk manager is empty: {self.chunk_manager.is_empty()}"
                    )
                    for device_name in self.device_action_queues.keys():
                        queue_structure = self.device_action_queues[device_name]
                        if isinstance(queue_structure, dict):
                            for group_id in queue_structure.keys():
                                queue_structure[group_id].clear()
                        else:
                            queue_structure.clear()
                    logger.info(f"Cleared all device action queues for {len(self.device_action_queues)} devices")
                    for device_name in self.device_first_actions.keys():
                        self.device_first_actions[device_name] = None
                    logger.info("Cleared all device first actions cache")
                    self._clear_all_shm()
                    self._write_action_index(0)
                    logger.info("Reset action_index to 0 for RTC")
                    logger.debug("PolicyRunner: Action cache, queues, and all SHM buffers cleared due to robot reset")
                except Exception as e:
                    logger.error(f"Error processing reset command file: {e}")
                    try:
                        reset_command_file.unlink()
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Error checking reset command: {e}")

    def _clear_all_shm(self) -> None:
        """Clear all SHM buffers: policy SHM and robot control SHMs."""
        try:
            if self.policy_shm:
                try:
                    buf = self.policy_shm.buf
                    policy_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
                    device_count = policy_header["device_count"]
                    reset_policy_header = pack_policy_header(device_count, 0)
                    buf[:POLICY_HEADER_SIZE] = reset_policy_header
                    frame_sizes = []
                    for i in range(device_count):
                        device_offset = get_device_info_offset(i)
                        device_header = unpack_device_header(buf[device_offset : device_offset + DEVICE_HEADER_SIZE])
                        frame_sizes.append(device_header["frame_size"])
                    for i in range(device_count):
                        data_offset = get_policy_data_offset(i, device_count, frame_sizes)
                        frame_size = frame_sizes[i]
                        zero_timestamp_header = pack_frame_header(0)
                        buf[data_offset : data_offset + FRAME_HEADER_SIZE] = zero_timestamp_header
                        data_start = data_offset + FRAME_HEADER_SIZE
                        data_end = data_offset + frame_size
                        buf[data_start:data_end] = b"\x00" * (frame_size - FRAME_HEADER_SIZE)
                    logger.info("Cleared policy SHM buffer")
                except Exception as e:
                    logger.error(f"Failed to clear policy SHM: {e}")
            for robot_state in self.robot_states:
                try:
                    robot_shm = robot_state["shm"]
                    buf = robot_shm.buf
                    buffer_header = unpack_buffer_header(
                        buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
                    )
                    buffer_size = buffer_header["buffer_size"]
                    frame_size = robot_state["frame_size"]
                    for i in range(buffer_size):
                        frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + i * frame_size
                        zero_timestamp_header = pack_frame_header(0)
                        buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = zero_timestamp_header
                        data_start = frame_offset + FRAME_HEADER_SIZE
                        data_end = frame_offset + frame_size
                        buf[data_start:data_end] = b"\x00" * (frame_size - FRAME_HEADER_SIZE)
                    reset_buffer_header = pack_buffer_header(buffer_size, 0, 0)
                    buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = reset_buffer_header
                    shm_name = robot_state["device_info"].get("shm", "unknown")
                    logger.info(f"Cleared robot control SHM buffer: {shm_name}")
                except Exception as e:
                    logger.error(f"Failed to clear robot control SHM: {e}")
        except Exception as e:
            logger.error(f"Error clearing SHM buffers: {e}")

    def _enqueue_action_with_delays(self, device_name: str, action: np.ndarray) -> None:
        """
        Enqueue action with dimension-level delay support.

        If device has dimension-level delays configured, split the action and enqueue
        each dimension group separately. Otherwise, enqueue the whole action.

        Also stores the first action received for this device to use during delay period.
        """
        if self.device_first_actions[device_name] is None:
            self.device_first_actions[device_name] = action.copy()
            logger.info(f"Stored first action for {device_name} to use during delay period")
        queue_structure = self.device_action_queues[device_name]
        if isinstance(queue_structure, dict):
            dimension_delays = self.device_dimension_delays[device_name]
            assigned_dims = set()
            for group_id, (indices, delay) in enumerate(dimension_delays):
                dim_values = action[indices].copy()
                queue_structure[group_id].append(dim_values)
                assigned_dims.update(indices)
                logger.debug(
                    f"Enqueued {device_name} dims {indices} to group {group_id}, queue_len={len(queue_structure[group_id])}, delay={delay}"
                )
            all_dims = set(range(len(action)))
            default_dims = sorted(list(all_dims - assigned_dims))
            if default_dims:
                default_values = action[default_dims].copy()
                queue_structure["default"].append(default_values)
                logger.debug(
                    f"Enqueued {device_name} dims {default_dims} to default queue, queue_len={len(queue_structure['default'])}"
                )
        else:
            queue_structure.append(action.copy())
            delay_frames = self.device_delays.get(device_name, 0)
            logger.debug(
                f"Enqueued action for {device_name}, queue length: {len(queue_structure)}, delay: {delay_frames}"
            )

    def _dequeue_action_with_delays(self, device_name: str) -> Optional[np.ndarray]:
        """
        Dequeue action with dimension-level delay support.

        If device has dimension-level delays configured, dequeue from each dimension
        queue and reconstruct the full action. Otherwise, dequeue the whole action.

        During the delay period (when queues are not ready), dimensions that need delay
        will use values from the first action received, while other dimensions use current values.
        """
        queue_structure = self.device_action_queues[device_name]
        first_action = self.device_first_actions.get(device_name)
        if isinstance(queue_structure, dict):
            dimension_delays = self.device_dimension_delays[device_name]
            queues_ready = {}
            for group_id, (indices, delay) in enumerate(dimension_delays):
                queues_ready[group_id] = len(queue_structure[group_id]) > delay
                if not queues_ready[group_id]:
                    logger.debug(
                        f"Queue not ready for {device_name} group {group_id} (dims {indices}): current={len(queue_structure[group_id])}, required={delay + 1}"
                    )
            default_delay = self.device_delays.get(device_name, 0)
            default_ready = len(queue_structure["default"]) > default_delay
            if not default_ready:
                logger.debug(
                    f"Default queue not ready for {device_name}: current={len(queue_structure['default'])}, required={default_delay + 1}"
                )
            if not any(queues_ready.values()) and (not default_ready):
                return None
            if first_action is not None:
                total_dims = len(first_action)
            else:
                total_dims = 0
                for group_id, (indices, _) in enumerate(dimension_delays):
                    total_dims = max(total_dims, max(indices) + 1)
                if queue_structure["default"]:
                    assigned_dims = set()
                    for indices, _ in dimension_delays:
                        assigned_dims.update(indices)
                    default_count = len(queue_structure["default"][0])
                    total_dims = max(total_dims, len(assigned_dims) + default_count)
            if first_action is not None:
                reconstructed_action = first_action.copy()
            else:
                reconstructed_action = np.zeros(total_dims, dtype=np.float64)
            assigned_dims = set()
            for group_id, (indices, delay) in enumerate(dimension_delays):
                if queues_ready[group_id]:
                    dim_values = queue_structure[group_id].popleft()
                    reconstructed_action[indices] = dim_values
                    assigned_dims.update(indices)
                    logger.debug(
                        f"Dequeued {device_name} group {group_id} (dims {indices}), remaining={len(queue_structure[group_id])}"
                    )
                else:
                    assigned_dims.update(indices)
                    logger.debug(
                        f"Using first action for {device_name} group {group_id} (dims {indices}) - queue not ready"
                    )
            all_dims = set(range(total_dims))
            default_dims = sorted(list(all_dims - assigned_dims))
            if default_dims:
                if default_ready:
                    default_values = queue_structure["default"].popleft()
                    reconstructed_action[default_dims] = default_values
                    logger.debug(
                        f"Dequeued {device_name} default (dims {default_dims}), remaining={len(queue_structure['default'])}"
                    )
                else:
                    logger.debug(
                        f"Using first action for {device_name} default (dims {default_dims}) - queue not ready"
                    )
            return reconstructed_action
        else:
            delay_frames = self.device_delays.get(device_name, 0)
            if len(queue_structure) > delay_frames:
                ready_action = queue_structure.popleft()
                logger.debug(
                    f"Dequeued action for {device_name} (delay={delay_frames}, remaining in queue={len(queue_structure)})"
                )
                return ready_action
            elif first_action is not None and len(queue_structure) > 0:
                logger.debug(
                    f"Using first action for {device_name} during delay period (current: {len(queue_structure)}, required: {delay_frames + 1})"
                )
                return first_action.copy()
            else:
                logger.debug(
                    f"Waiting for {device_name} queue to fill (current: {len(queue_structure)}, required: {delay_frames + 1})"
                )
                return None

    def _read_latest_policy_frame(self) -> Optional[Tuple[Dict[str, np.ndarray], int]]:
        """Read latest action chunk dict from policy SHM using standard manager/device structure."""
        try:
            if not self.policy_shm:
                return None
            buf = self.policy_shm.buf
            manager_header = unpack_policy_header(buf[:POLICY_HEADER_SIZE])
            device_count = manager_header["device_count"]
            update_timestamp = manager_header["update_timestamp"]
            if update_timestamp == 0:
                return None
            frame_sizes = []
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset : device_offset + DEVICE_HEADER_SIZE])
                frame_sizes.append(device_header["frame_size"])
            action_chunk_dict = {}
            latest_timestamp = 0
            for i in range(device_count):
                device_offset = get_device_info_offset(i)
                device_header = unpack_device_header(buf[device_offset : device_offset + DEVICE_HEADER_SIZE])
                device_name = device_header["device_type"]
                shape = device_header["shape"]
                frame_size = device_header["frame_size"]
                data_offset = get_policy_data_offset(i, device_count, frame_sizes)
                frame_header = unpack_frame_header(buf[data_offset : data_offset + FRAME_HEADER_SIZE])
                timestamp_ns = frame_header["timestamp_ns"]
                if timestamp_ns == 0:
                    continue
                data_start = data_offset + FRAME_HEADER_SIZE
                data_end = data_start + (frame_size - FRAME_HEADER_SIZE)
                sliced_buf = buf[data_start:data_end]
                chunk_data = np.frombuffer(sliced_buf, dtype=np.float64)
                action_chunk = chunk_data.reshape(shape)
                action_chunk_dict[device_name] = action_chunk
                latest_timestamp = max(latest_timestamp, timestamp_ns)
            if not action_chunk_dict:
                return None
            return (action_chunk_dict, latest_timestamp)
        except Exception as e:
            logger.error(f"Failed to read policy frame: {e}")
            return None

    def _dispatch_action_to_robots(self, action_dict: Dict[str, np.ndarray]) -> None:
        """
        Dispatch action dict to robot control SHMs.

        This function simply writes the provided actions to their corresponding robot control SHMs.
        Delay management is handled in _update_execution().
        """
        if not self.robot_states or not action_dict:
            return
        for robot_state in self.robot_states:
            robot_cfg = robot_state["device_info"]
            shm_name = robot_cfg.get("shm", "")
            device_name = shm_name.replace("_control", "") if shm_name.endswith("_control") else shm_name
            if device_name in action_dict:
                robot_action = action_dict[device_name]
                self._write_action_to_robot(robot_state, robot_action)
                logger.debug(
                    f"Dispatched action to {device_name}: {(robot_action[:3] if len(robot_action) > 3 else robot_action)}"
                )
            else:
                logger.debug(f"No action found for device {device_name} in this dispatch cycle")

    def _write_action_to_robot(self, robot_state: Dict[str, Any], action: np.ndarray) -> None:
        """Write action to robot control SHM."""
        try:
            robot_shm = robot_state["shm"]
            buf = robot_shm.buf
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            write_index = buffer_header["write_index"]
            buffer_size = buffer_header["buffer_size"]
            frame_size = robot_state["frame_size"]
            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + write_index * frame_size
            current_time_ns = time.time_ns()
            frame_header = pack_frame_header(current_time_ns)
            buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = frame_header
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + (frame_size - FRAME_HEADER_SIZE)
            action_bytes = action.tobytes()
            if len(action_bytes) != frame_size - FRAME_HEADER_SIZE:
                data_len = frame_size - FRAME_HEADER_SIZE
                if len(action_bytes) > data_len:
                    action_bytes = action_bytes[:data_len]
                else:
                    action_bytes = action_bytes + b"\x00" * (data_len - len(action_bytes))
            buf[data_start:data_end] = action_bytes
            new_write_index = (write_index + 1) % buffer_size
            new_current_frames_count = min(buffer_header["current_frames_count"] + 1, buffer_size)
            new_buffer_header = pack_buffer_header(buffer_size, new_current_frames_count, new_write_index)
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = new_buffer_header
        except Exception as e:
            logger.error(f"Failed to write action to robot: {e}")
