#!/usr/bin/env python3
"""
Robot Device - Base robot class and PyBullet-based simulated robot.

Author: Jun Lv,Zixi Ying
"""

import os
import time
import threading
import subprocess
import json
import numpy as np
import multiprocessing.shared_memory as shm
from typing import Optional, Dict, Any, Union, List, Literal
from devices.base import BaseDevice
from utils.shm_utils import (
    pack_device_header,
    pack_buffer_header,
    pack_frame_header,
    calculate_device_shm_size,
    DEVICE_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    get_control_frame_offset,
    get_control_frame_offsets,
    get_latest_control_frame_index,
    get_latest_control_frame_offset,
    get_dtype,
    create_control_shm,
    connect_to_shared_memory,
    unpack_device_header,
    unpack_buffer_header,
)
from utils.logger_config import logger
from utils.portal_compat import (
    import_portal_with_wake_pipe_fix,
    portal_wake_pipe_stats,
)
from scipy.spatial.transform import Rotation as R


class BaseRobot(BaseDevice):
    """
    Base robot class that extends BaseDevice with control capabilities.
    Provides dual SHM buffers: one for state data (inherited) and one for control commands.
    """

    def __init__(
        self,
        device_id=0,
        data_shape=(7,),
        control_shape=(7,),
        fps=100.0,
        data_dtype=np.float64,
        control_dtype=np.float64,
        command_fps=100.0,
        buffer_size=100,
        control_buffer_size=50,
        hardware_latency_ms=0.0,
        control_shm_name: Union[str, List[str], None] = None,
        control_flag_shm_name: Union[str, None] = None,
        dual_control_mode: bool = False,
        **kwargs,
    ):
        """
        Initialize the base robot.

        Args:
            device_id: Unique identifier for this robot instance
            data_shape: Shape of robot state data (e.g., joint positions, velocities)
            control_shape: Shape of control command data (e.g., target joint positions)
            fps: Control frequency in Hz
            data_dtype: Data type for state data (string or numpy dtype)
            control_dtype: Data type for control data (string or numpy dtype)
            buffer_size: Number of state frames to store in buffer
            control_buffer_size: Number of control commands to store in buffer
            hardware_latency_ms: Hardware latency in milliseconds (default: 0.0)
            control_shm_name: Name of existing control SHM to connect to (if None, will create new one)
            control_flag_shm_name: Name of existing control flag SHM to connect to (only effective in dual control mode)
            **kwargs: Additional arguments passed to BaseDevice
        """
        # Initialize base device for state data
        super().__init__(
            device_id=device_id,
            data_shape=data_shape,
            fps=fps,
            data_dtype=data_dtype,
            buffer_size=buffer_size,
            hardware_latency_ms=hardware_latency_ms,
            **kwargs,
        )
        # Robot command executer fps
        self.command_fps = command_fps

        # Robot-specific attributes
        # Convert control_shape to tuple if it's a list
        self.control_shape = tuple(control_shape) if isinstance(control_shape, list) else control_shape

        self.control_dtype = control_dtype

        self.control_buffer_size = control_buffer_size

        # Control buffer management
        self.control_shared_memory = None

        # Control flag SHM (Created by the teleoperator)
        self._control_flag_shm = None
        self._control_flag_shm_name = control_flag_shm_name

        # Calculate control buffer layout
        self._calculate_control_buffer_layout()

        self.control_shm_names = control_shm_name
        if isinstance(self.control_shm_names, str):
            self.control_shm_names = [self.control_shm_names]

        self.dual_control_mode = dual_control_mode
        assert not self.dual_control_mode or self._control_flag_shm_name is not None, (
            "Control flag SHM name is required in dual control mode"
        )

    def _calculate_control_buffer_layout(self) -> None:
        """Calculate the layout of the control command buffer."""
        # Calculate control data size per command
        numpy_control_dtype = get_dtype(self.control_dtype)
        bytes_per_element = np.dtype(numpy_control_dtype).itemsize
        control_data_size = np.prod(self.control_shape) * bytes_per_element

        # Control frame header: timestamp (8 bytes)
        self.control_frame_header_size = FRAME_HEADER_SIZE
        self.control_frame_data_size = control_data_size
        self.control_frame_size = self.control_frame_header_size + self.control_frame_data_size

        # Pre-calculate control frame offsets
        self.control_frame_offsets = get_control_frame_offsets(self.control_buffer_size, self.control_frame_size)

        # Total control SHM size: device_header + buffer_header + control_frames
        self.total_control_shm_size = calculate_device_shm_size(self.control_buffer_size, self.control_frame_data_size)

        # Control shared memory name
        self.default_control_shm_name = f"{self.device_name}_{self.device_id}_control"

    def _create_control_shared_memory(self) -> None:
        """Create shared memory for control commands using unified interface."""
        # Phase 1: Create writable SHM for initialization
        control_shm = create_control_shm(
            device_name=self.device_name,
            device_id=self.device_id,
            buffer_size=self.control_buffer_size,
            frame_data_size=self.control_frame_data_size,
            readonly_after_creation=False,  # Initially writable for initialization
        )

        # Store in control shared memory dict (don't overwrite user-specified control_shm_names)
        if self.control_shm_names is None:
            self.control_shm_names = [self.default_control_shm_name]
        elif self.default_control_shm_name not in self.control_shm_names:
            self.control_shm_names.append(self.default_control_shm_name)
        if self.control_shared_memory is None:
            self.control_shared_memory = {}
        self.control_shared_memory.update({self.default_control_shm_name: control_shm})

        logger.info(
            f"Created control shared memory: {self.default_control_shm_name} ({self.total_control_shm_size:,} bytes)"
        )

        # Phase 2: Initialize control buffer header
        self._initialize_control_buffer_header()
        logger.info("Initialized control buffer header")

        # Phase 3: Convert to read-only after initialization
        from utils.shm_utils import make_readonly

        readonly_control_shm = make_readonly(control_shm)
        self.control_shared_memory[self.default_control_shm_name] = readonly_control_shm

        logger.info("Control SHM converted to READ-ONLY after initialization")

    def _connect_to_existing_control_shm(self, shm_name: Optional[str] = None) -> bool:
        """
        Connect to an existing control shared memory in READ-ONLY mode using unified interface.

        Args:
            shm_name: Name of the existing control SHM (if None, uses default name)

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Try to connect to the SHM directly
            # Let the unified interface handle the connection logic
            self.control_shared_memory[shm_name] = connect_to_shared_memory(shm_name, read_only=True)
            logger.info(f"Connected to control SHM: {shm_name} (READ-ONLY)")

            # Validate the connected SHM
            if not self._validate_control_shm():
                logger.warning(f"Control SHM validation failed for: {shm_name}")
                self.control_shared_memory[shm_name].close()
                self.control_shared_memory[shm_name] = None
                return False

            logger.info(f"Successfully connected to control source: {shm_name} (READ-ONLY)")
            return True

        except FileNotFoundError:
            logger.warning(f"Control shared memory not found: {shm_name}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to control shared memory {shm_name}: {e}")
            return False

    def _connect_control_flag_shm(self) -> None:
        """
        Connect to the control flag SHM owned by I2RT leader.

        This SHM is owned by the teleoperator, so we connect as a read-only consumer
        and should NEVER close it - the teleoperator will handle cleanup.
        """
        try:
            from utils.shm_utils import connect_to_shared_memory

            self._control_flag_shm = connect_to_shared_memory(self._control_flag_shm_name, read_only=True)
            logger.info(f"Connected to control flag SHM: {self._control_flag_shm_name} (READ-ONLY)")

        except FileNotFoundError:
            logger.warning(f"Control flag SHM not found: {self._control_flag_shm_name}")
            self._control_flag_shm = None
        except Exception as e:
            logger.warning(f"Failed to connect to control flag SHM: {e}")
            self._control_flag_shm = None

    def _validate_control_shm(self) -> bool:
        """
        Validate the connected control shared memory.

        Returns:
            bool: True if validation successful, False otherwise
        """
        try:
            from utils.shm_utils import unpack_device_header, unpack_buffer_header

            for shm_name in self.control_shm_names:
                # Read and validate device header
                device_header_data = self.control_shared_memory[shm_name].buf[:DEVICE_HEADER_SIZE]
                device_info = unpack_device_header(device_header_data)

                # Check control shape
                shm_shape = device_info["shape"]
                if shm_shape != self.control_shape:
                    logger.error(f"Control shape mismatch: expected {self.control_shape}, got {shm_shape}")
                    return False

                # Check control dtype
                shm_dtype_str = device_info["data_dtype"]  # dtype is at index 3
                if shm_dtype_str != self.control_dtype:
                    logger.error(f"Control dtype mismatch: expected {self.control_dtype}, got {shm_dtype_str}")
                    return False

                # Read and validate buffer header
                buffer_header_data = self.control_shared_memory[shm_name].buf[
                    DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
                ]
                buffer_info = unpack_buffer_header(buffer_header_data)

                # Check buffer size
                shm_buffer_size = buffer_info["buffer_size"]
                if shm_buffer_size != self.control_buffer_size:
                    logger.error(
                        f"Control buffer size mismatch: expected {self.control_buffer_size}, got {shm_buffer_size}"
                    )
                    return False

                # Recalculate frame offsets based on connected SHM
                shm_frame_size = device_info["frame_size"]  # frame_size is at index 5
                self.control_frame_size = shm_frame_size
                self.control_frame_data_size = shm_frame_size - FRAME_HEADER_SIZE

                self.control_frame_offsets = get_control_frame_offsets(
                    self.control_buffer_size, self.control_frame_size
                )

                logger.info("Control SHM validation successful")
                logger.info(f"  - Shape: {shm_shape}")
                logger.info(f"  - Dtype: {shm_dtype_str}")
                logger.info(f"  - Buffer size: {shm_buffer_size}")
                logger.info(f"  - Current count: {buffer_info['current_frames_count']}")
                logger.info(f"  - Write index: {buffer_info['write_index']}")

            return True

        except Exception as e:
            logger.error(f"Error validating control SHM: {e}")
            return False

    def _get_control_buffer_info(self, shm_name: str) -> Optional[tuple]:
        """
        Get current control buffer information from SHM.

        Returns:
            tuple: (current_frames_count, write_index) or (None, None) if error
        """
        try:
            from utils.shm_utils import unpack_buffer_header

            # Read buffer header from SHM
            buffer_header_data = self.control_shared_memory[shm_name].buf[
                DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
            ]
            buffer_info = unpack_buffer_header(buffer_header_data)

            current_frames_count = buffer_info["current_frames_count"]
            write_index = buffer_info["write_index"]

            return current_frames_count, write_index

        except Exception as e:
            logger.error(f"Error reading control buffer info: {e}")
            return None, None

    def _initialize_control_buffer_header(self) -> None:
        """Initialize the control buffer header with metadata."""
        # Pack device header for control buffer
        device_header = pack_device_header(
            f"{self.device_name}_control",
            self.device_id,
            self.fps,
            self.control_dtype,
            self.control_shape,
            self.control_frame_size,
        )

        # Pack buffer header (initialize with zeros)
        buffer_header = pack_buffer_header(self.control_buffer_size, 0, 0)

        # Write headers to control SHM
        self.control_shared_memory[self.default_control_shm_name].buf[:DEVICE_HEADER_SIZE] = device_header
        self.control_shared_memory[self.default_control_shm_name].buf[
            DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE
        ] = buffer_header

    def read_latest_control_command(self, shm_name: str, min_timestamp_ns: int = 0) -> Optional[np.ndarray]:
        """
        Read the latest control command from the control buffer.

        Args:
            shm_name: Name of the shared memory to read from
            min_timestamp_ns: Minimum valid timestamp (in nanoseconds). If the action's
                           timestamp is less than this value, it is considered stale and
                           (None, None) is returned. Default is 0 (no filtering).

        Returns:
            tuple: (timestamp_ns, control_array) or (None, None) if no command available
                   or if the action is older than min_timestamp_ns
        """
        if not self.control_shared_memory[shm_name]:
            return None, None

        # Get current buffer info from SHM
        current_frames_count, write_index = self._get_control_buffer_info(shm_name)
        if current_frames_count is None or current_frames_count == 0:
            return None, None

        try:
            # Calculate frame offset for latest command
            latest_index = get_latest_control_frame_index(write_index, self.control_buffer_size)
            frame_offset = self.control_frame_offsets[latest_index]

            # Read frame header
            _frame_header = pack_frame_header(0)  # Placeholder
            frame_header_data = self.control_shared_memory[shm_name].buf[
                frame_offset : frame_offset + FRAME_HEADER_SIZE
            ]
            timestamp_ns = int.from_bytes(frame_header_data, byteorder="little")

            # Check if action is stale (written before the minimum valid timestamp)
            if min_timestamp_ns > 0 and timestamp_ns > 0 and timestamp_ns < min_timestamp_ns:
                return None, None

            # Read control data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + self.control_frame_data_size
            numpy_control_dtype = get_dtype(self.control_dtype)
            # Extract control data: DataChannel.buf returns raw buffer, slicing works directly
            sliced_buf = self.control_shared_memory[shm_name].buf[data_start:data_end]
            control_array = np.frombuffer(sliced_buf, dtype=numpy_control_dtype)
            # Reshape if needed
            if len(self.control_shape) > 0:
                control_array = control_array.reshape(self.control_shape)

            return timestamp_ns, control_array

        except Exception as e:
            logger.error(f"Error reading control command: {e}")
            return None, None

    def get_robot_state(self) -> np.ndarray:
        """
        Get current robot state. To be implemented by subclasses.

        Returns:
            numpy.ndarray: Current robot state
        """
        raise NotImplementedError("Subclasses must implement get_robot_state()")

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        """
        Execute a control command. To be implemented by subclasses.

        Args:
            control_array: Control command to execute
        """
        raise NotImplementedError("Subclasses must implement execute_control_command()")

    def _cleanup_control_shared_memory(self) -> None:
        """Clean up control shared memory resources."""
        for shm_name in self.control_shm_names:
            if self.control_shared_memory[shm_name]:
                try:
                    self.control_shared_memory[shm_name].close()
                    self.control_shared_memory[shm_name].unlink()
                    logger.info(f"Cleaned up control shared memory: {shm_name}")
                except (FileNotFoundError, PermissionError):
                    pass  # Already cleaned up or no permission
                except Exception as e:
                    logger.error(f"Error cleaning up control shared memory: {e}")
                finally:
                    self.control_shared_memory[shm_name] = None

    def _check_and_execute_home_command(self) -> None:
        """
        Check for external home command file and execute if present.

        Override in subclasses to implement robot-specific homing behavior.
        Base implementation does nothing.
        """
        pass

    def _clear_control_shm(self) -> None:
        """
        Clear all data in robot control SHM buffers.

        Override in subclasses to implement robot-specific SHM clearing.
        Base implementation does nothing.
        """
        pass

    def _notify_policy_runner_reset(self) -> None:
        """
        Notify policy runner to clear action cache and SHM when robot is resetting.

        Override in subclasses to implement notification mechanism.
        Base implementation does nothing.
        """
        pass

    def start_server(self) -> None:
        """Start the robot server and run control loop."""
        if self.running:
            logger.info(f"Robot {self.device_name}_{self.device_id} is already running")
            return

        logger.info(f"Starting {self.device_name}_{self.device_id} server...")
        self.running = True

        # Create state shared memory
        self._create_shared_memory()  # State buffer
        if self.control_shm_names is not None:
            self.control_shared_memory = {shm_name: None for shm_name in self.control_shm_names}
            for shm_name in self.control_shm_names:
                connected = self._connect_to_existing_control_shm(shm_name)
                # Give peer devices a short window to create their SHMs.
                if not connected:
                    max_retries = 20
                    retry_interval_s = 0.2
                    for _ in range(max_retries):
                        time.sleep(retry_interval_s)
                        connected = self._connect_to_existing_control_shm(shm_name)
                        if connected:
                            break
                    if not connected:
                        logger.warning(f"Failed to connect control SHM after retries: {shm_name}")
        if self.dual_control_mode or self.control_shm_names is None:
            self._create_control_shared_memory()  # Control buffer is currently used for policy rollout

        # Connect to control flag SHM (owned by I2RT leader) if in dual control mode
        if self.dual_control_mode and not getattr(self, "is_leader_role", False):
            self._connect_control_flag_shm()

        if not self.shared_memory or not any(self.control_shared_memory.values()):
            self.running = False
            raise RuntimeError("Failed to create/connect to shared memories")

        logger.info(f"Robot {self.device_name}_{self.device_id} server started successfully")

        # Main control loop
        try:
            assert self.fps >= self.command_fps
            iteration = 1
            command_iteration = int(self.fps // self.command_fps)
            while self.running:
                start_time = time.time()

                # Check for external home command (from GUI reset)
                # Only check periodically to avoid performance impact
                if hasattr(self, "_last_home_check_time"):
                    if time.time() - self._last_home_check_time > 0.1:  # Check every 100ms
                        self._check_and_execute_home_command()
                        self._last_home_check_time = time.time()
                else:
                    self._last_home_check_time = time.time()
                    self._check_and_execute_home_command()

                if hasattr(self, "_check_and_execute_teleop_start_command"):
                    if hasattr(self, "_last_teleop_start_check_time"):
                        if time.time() - self._last_teleop_start_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_start_command()
                            self._last_teleop_start_check_time = time.time()
                    else:
                        self._last_teleop_start_check_time = time.time()
                        self._check_and_execute_teleop_start_command()

                if hasattr(self, "_check_and_execute_teleop_discard_command"):
                    if hasattr(self, "_last_teleop_discard_check_time"):
                        if time.time() - self._last_teleop_discard_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_discard_command()
                            self._last_teleop_discard_check_time = time.time()
                    else:
                        self._last_teleop_discard_check_time = time.time()
                        self._check_and_execute_teleop_discard_command()
                # Check for control mode switch BEFORE reading commands
                # This ensures we set the timestamp filter before reading any actions
                if self.dual_control_mode:
                    if self._control_flag_shm is None and not getattr(self, "is_leader_role", False):
                        self._connect_control_flag_shm()
                    current_control_flag = self._read_control_flag()

                    # Detect transition from human to policy control
                    if not hasattr(self, "_last_control_flag"):
                        self._last_control_flag = 0.0
                    if self._last_control_flag > 0.5 and current_control_flag <= 0.5:
                        # Clear control SHM to prevent stale actions from being executed
                        self._clear_control_shm()
                        # Record when policy control started - used to filter stale actions
                        self._policy_control_start_timestamp = time.time_ns()
                        logger.info(
                            f"Control mode switch detected in main loop: cleared control SHM, timestamp={self._policy_control_start_timestamp}"
                        )

                    # Toggle control mode if control flag changes [Only for follower arms because leader toggles a.s.a the button is pressed]
                    if hasattr(self, "_toggle_control_mode") and getattr(self, "is_follower_role", False):
                        if (current_control_flag > 0.5) != (self._last_control_flag > 0.5):
                            self._toggle_control_mode()

                    self._last_control_flag = current_control_flag

                control_dict = {}
                for shm_name in self.control_shm_names:
                    # Read latest control command
                    # Pass minimum timestamp to filter stale actions after control mode switch
                    # Add grace period to allow fresh policy actions to arrive
                    policy_start_ts = getattr(self, "_policy_control_start_timestamp", 0)
                    grace_period_ns = getattr(self, "_policy_control_grace_period_ns", int(0.5 * 1e9))
                    last_home_command_timestamp = getattr(self, "_last_home_command_timestamp", 0)
                    min_timestamp_ns = policy_start_ts + grace_period_ns if policy_start_ts > 0 else 0
                    min_timestamp_ns = max(min_timestamp_ns, last_home_command_timestamp)
                    timestamp_ns, control_array = self.read_latest_control_command(shm_name, min_timestamp_ns)
                    control_dict[shm_name] = control_array

                if any([value is not None for value in control_dict.values()]):  # Any non-None values in control_dict
                    if iteration % command_iteration == 0:
                        # Execute control command
                        self.execute_control_command(control_dict)
                iteration = iteration % command_iteration + 1
                # Get current robot state
                state_array = self.get_robot_state()
                # Write state to shared memory
                if state_array is not None:
                    self._write_array_to_shm_with_timestamp(state_array, time.time_ns())

                # Maintain control frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info(f"Stopping {self.device_name}_{self.device_id} server...")
        except Exception as e:
            logger.error(f"Error in robot control loop: {e}")
        finally:
            self.stop_server()

    def stop_server(self) -> None:
        """Stop the robot server and clean up resources."""
        logger.info(
            f"=== Stopping {self.device_name}_{self.device_id} server... in BaseRobot self.running={self.running} ==="
        )
        if not self.running:
            return

        self.running = False
        self._cleanup_shared_memory()
        self._cleanup_control_shared_memory()

        # Close connection to control flag SHM (owned by teleoperator, don't unlink)
        if self._control_flag_shm is not None:
            try:
                self._control_flag_shm.close()
                logger.info(f"Closed control flag SHM connection: {self._control_flag_shm_name}")
            except Exception as e:
                logger.warning(f"Failed to close control flag SHM connection: {e}")
            finally:
                self._control_flag_shm = None

        logger.info(f"Stopped {self.device_name}_{self.device_id} server")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop_server()


class I2RTRobot(BaseRobot):
    """
    Inherits from BaseRobot and provides I2RT robot interface for VAM.
    """

    # Hardcoded YAM teaching-handle compensation measured on 2026-07-17.
    # Maps left leader trigger output onto the right leader trigger output range.
    _LEFT_LEADER_GRIPPER_SOURCE_RANGE = (0.046740510, 0.997808599)
    _LEFT_LEADER_GRIPPER_TARGET_RANGE = (0.0175, 1.0)
    _TAKEOVER_ALIGNMENT_FLAG = 0.5
    _CONTROL_MODE_AUTONOMOUS = "follower_autonomous"
    _CONTROL_MODE_ALIGNMENT = "takeover_alignment"
    _CONTROL_MODE_TELEOP = "leader_to_follower"

    def __init__(
        self,
        device_id=0,
        data_shape=(14,),
        control_shape=(14,),
        fps=100.0,
        data_dtype=np.float64,
        control_dtype=np.float64,
        command_fps=100.0,
        buffer_size=100,
        control_buffer_size=50,
        hardware_latency_ms=0.0,
        control_shm_name: Union[str, List[str], None] = None,
        control_flag_shm_name: Union[str, None] = None,
        dual_control_mode: bool = False,
        policy_control_only: bool = False,
        takeover_alignment: Optional[Dict[str, Any]] = None,
        signal_files: Optional[Dict[str, str]] = None,
        # I2RT-specific arguments
        gripper_type: Literal[
            "crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"
        ] = "yam_teaching_handle",
        mode: Literal["follower", "leader"] = "follower",
        server_host: str = "localhost",
        server_port: int = 11333,
        can_channel: str = "can0",
        bilateral_kp: float = 0.0,
        bilateral_kd: float = 0.0,
        home_joint_pos: Optional[Union[np.ndarray, List[float]]] = None,
        left_server_host: Optional[str] = None,
        right_server_host: Optional[str] = None,
        left_server_port: Optional[int] = None,
        right_server_port: Optional[int] = None,
        left_can_channel: Optional[str] = None,
        right_can_channel: Optional[str] = None,
        server_startup_wait_s: float = 3.0,
        server_launch_python: str = "python",
        minimum_gello_script: Optional[str] = None,
        portal_stats_interval_s: float = 0.0,
        random_init_pose: bool = False,
        random_init_pose_params: Optional[Dict[str, Any]] = None,
        random_init_pose_sync: bool = False,
        **kwargs,
    ):
        resolved_control_flag_name = control_flag_shm_name
        if "leader" in mode and resolved_control_flag_name is None:
            resolved_control_flag_name = f"{self.__class__.__name__}_{device_id}_control_flag"

        super().__init__(
            device_id=device_id,
            data_shape=data_shape,
            control_shape=control_shape,
            fps=fps,
            data_dtype=data_dtype,
            control_dtype=control_dtype,
            command_fps=command_fps,
            buffer_size=buffer_size,
            control_buffer_size=control_buffer_size,
            hardware_latency_ms=hardware_latency_ms,
            control_shm_name=control_shm_name,
            control_flag_shm_name=resolved_control_flag_name,
            dual_control_mode=dual_control_mode,
            **kwargs,
        )

        self.gripper_type = gripper_type
        self.mode = mode
        self.server_host = server_host
        self.server_port = server_port
        self.can_channel = can_channel
        self.bilateral_kp = bilateral_kp
        self.bilateral_kd = bilateral_kd

        self.home_joint_pos = None if home_joint_pos is None else np.asarray(home_joint_pos, dtype=np.float64).flatten()
        self.is_leader_role = "leader" in self.mode
        self.is_follower_role = not self.is_leader_role
        self.policy_control_only = bool(policy_control_only)
        self.takeover_alignment_config = dict(takeover_alignment or {})
        self.takeover_alignment_enabled = bool(self.takeover_alignment_config.get("enabled", False))
        self.takeover_alignment_duration_s = float(self.takeover_alignment_config.get("duration_s", 0.2))
        self.takeover_alignment_settle_timeout_s = float(self.takeover_alignment_config.get("settle_timeout_s", 0.15))
        if not np.isfinite(self.takeover_alignment_duration_s) or self.takeover_alignment_duration_s < 0.0:
            raise ValueError("takeover_alignment.duration_s must be finite and non-negative")
        if not np.isfinite(self.takeover_alignment_settle_timeout_s) or self.takeover_alignment_settle_timeout_s < 0.0:
            raise ValueError("takeover_alignment.settle_timeout_s must be finite and non-negative")
        if self.takeover_alignment_enabled and not self.is_leader_role:
            raise ValueError("takeover_alignment may only be enabled on the leader robot")
        if self.policy_control_only and self.dual_control_mode:
            raise ValueError("policy_control_only cannot be combined with dual_control_mode")
        if self.policy_control_only and self.is_leader_role:
            raise ValueError("policy_control_only is only supported for follower robots")

        self.random_init_pose = bool(random_init_pose)
        self.random_init_pose_sync = bool(random_init_pose_sync)
        self.random_init_pose_params = dict(random_init_pose_params or {})
        position_radius_m = float(self.random_init_pose_params.get("position_radius_m", 0.0))
        if position_radius_m < 0.0:
            raise ValueError("random_init_pose_params.position_radius_m must be non-negative")
        self.random_init_pose_params["position_radius_m"] = position_radius_m

        rotation_range = np.asarray(
            self.random_init_pose_params.get("rotation_euler_range_deg", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(-1)
        if rotation_range.size == 1:
            rotation_range = np.repeat(rotation_range, 3)
        if rotation_range.size != 3 or np.any(rotation_range < 0.0):
            raise ValueError("random_init_pose_params.rotation_euler_range_deg must contain three non-negative values")
        self.random_init_pose_params["rotation_euler_range_deg"] = rotation_range
        self.random_init_pose_params.setdefault("sync_target_path", "/tmp/vam_i2rt_random_home_target.json")
        self.random_init_pose_params.setdefault("sync_wait_timeout_s", 15.0)
        self.random_init_pose_params.setdefault("sync_wait_before_read_s", 0.5)

        seed = self.random_init_pose_params.get("seed")
        self._random_init_pose_rng = np.random.default_rng(seed)
        self._i2rt_kinematics = None
        self._i2rt_kinematics_backend = None

        self.left_server_host = left_server_host or server_host
        self.right_server_host = right_server_host or server_host
        self.left_server_port = left_server_port if left_server_port is not None else server_port
        self.right_server_port = right_server_port if right_server_port is not None else server_port + 1
        self.left_can_channel = left_can_channel or can_channel
        self.right_can_channel = right_can_channel or can_channel
        self.server_startup_wait_s = server_startup_wait_s
        self.server_launch_python = server_launch_python
        experiment_server_script = os.environ.get("VAM_I2RT_MINIMUM_GELLO_SCRIPT")
        if minimum_gello_script is None:
            from pathlib import Path

            repo_root = Path(__file__).resolve().parents[1]
            self.minimum_gello_script = experiment_server_script or str(
                repo_root.parent / "packages" / "i2rt" / "scripts" / "minimum_gello.py"
            )
        else:
            self.minimum_gello_script = minimum_gello_script

        portal_stats_env = os.environ.get("VAM_I2RT_PORTAL_STATS_INTERVAL_S")
        if portal_stats_env is not None:
            try:
                portal_stats_interval_s = float(portal_stats_env)
            except ValueError as exc:
                raise ValueError("VAM_I2RT_PORTAL_STATS_INTERVAL_S must be a number") from exc
        self._portal_stats_interval_s = max(0.0, float(portal_stats_interval_s))
        self._portal_stats_last_log_monotonic = 0.0
        self._diagnostic_run_id = os.environ.get("VAM_I2RT_DIAGNOSTIC_RUN_ID", "untracked")

        self._left_client = None
        self._right_client = None
        self._server_processes: List[subprocess.Popen] = []
        self._last_button_left = 0.0
        self._last_button_right = 0.0
        self._leader_flag_write_index = 0
        self._leader_flag_buffer_size = max(100, int(self.control_buffer_size))
        self._leader_flag_owner = False
        self._last_button_pressed = False
        self._button_down_latched = False
        self._button_release_samples = 0
        self._button_release_debounce_samples = 3
        self._control_mode = self._CONTROL_MODE_AUTONOMOUS
        # Serializes leader state publishing with takeover alignment.
        self._dagger_mode_transition_lock = threading.Lock()
        self._takeover_hold_joint_pos: Optional[np.ndarray] = None
        self._last_control_flag = 0.0 if self.dual_control_mode else 1.0
        self._load_default_bilateral_gains()

        self._home_command_active = False
        # Normal homing uses the filtered state cache independently of RPC recovery.
        self._trusted_state_check_interval = 0.01
        self._trusted_state_thread: Optional[threading.Thread] = None
        self._trusted_state_monitor_evt = threading.Event()
        self._main_control_loop_thread: Optional[threading.Thread] = None
        self._main_control_loop_generation: int = 0
        self._main_control_loop_lock = threading.Lock()

        # Episode signal tracking for button-controlled data collection
        if signal_files is None:
            raise ValueError("signal_files must be provided in config")
        self._episode_signal_pattern = signal_files.get("episode_signal_pattern")
        self._robot_home_command_pattern = signal_files.get("robot_home_command_pattern")
        self._robot_teleop_start_command_pattern = signal_files.get("robot_teleop_start_command_pattern")
        self._robot_teleop_discard_command_pattern = signal_files.get("robot_teleop_discard_command_pattern")
        self._policy_runner_reset_command = signal_files.get("policy_runner_reset_command")
        self._reset_signal = signal_files.get("reset_signal")
        self._robot_halt_command_pattern = signal_files.get("robot_halt_command_pattern")
        self._robot_halt_reply_pattern = signal_files.get("robot_halt_reply_pattern")
        self._is_recording_episode = False

        # self._last_get_info_state_left = None
        # self._last_get_info_state_right = None
        self._last_trusted_get_info_state_left = None
        self._last_trusted_get_info_state_right = None
        self._trusted_state_lock = threading.Lock()
        self.state_outlier_threshold = 0.3
        self._robot_halt_command_active = False

        self._ensure_rpc_servers_running()
        self._connect_rpc_clients()
        logger.info(
            f"Initialized I2RTRobot_{self.device_id} role="
            f"{'leader' if self.is_leader_role else 'follower'} "
            f"left={self.left_server_host}:{self.left_server_port} "
            f"right={self.right_server_host}:{self.right_server_port}"
        )

    def _connect_rpc_clients(self) -> None:
        """Connect to left/right I2RT RPC servers."""
        try:
            portal_module = import_portal_with_wake_pipe_fix()
            self._left_client = portal_module.Client(f"{self.left_server_host}:{self.left_server_port}")
            self._right_client = portal_module.Client(f"{self.right_server_host}:{self.right_server_port}")
            self._configure_portal_client_diagnostics(self._left_client)
            self._configure_portal_client_diagnostics(self._right_client)
            # minimum_gello leader/follower server exposes get_info()
            self._left_client.get_info().result(timeout=2.0)
            self._right_client.get_info().result(timeout=2.0)
        except Exception as e:
            logger.error(f"Failed to connect I2RT RPC clients: {e}")
            self._left_client = None
            self._right_client = None

    def _load_default_bilateral_gains(self) -> None:
        """Load default KP/KD from bilateral_control.py (with local fallback)."""
        try:
            import importlib.util
            from pathlib import Path

            repo_root = Path(__file__).resolve().parents[1]
            bilateral_path = repo_root.parent / "packages" / "i2rt" / "examples" / "bilateral_control.py"
            spec = importlib.util.spec_from_file_location("i2rt_bilateral_control_defaults", bilateral_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                leader_default_kp = np.asarray(module.LEADER_DEFAULT_KP, dtype=np.float64)
                leader_default_kd = np.asarray(module.LEADER_DEFAULT_KD, dtype=np.float64)
                follower_default_kp = np.asarray(module.FOLLOWER_DEFAULT_KP, dtype=np.float64)
                follower_default_kd = np.asarray(module.FOLLOWER_DEFAULT_KD, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Failed to import bilateral defaults, using fallback: {e}")

        self._leader_default_kp = leader_default_kp
        self._leader_default_kd = leader_default_kd
        self._leader_zero_kp = np.zeros_like(leader_default_kp)
        self._leader_zero_kd = np.zeros_like(leader_default_kd)
        self._follower_default_kp = follower_default_kp
        self._follower_default_kd = follower_default_kd
        self._follower_zero_kp = np.zeros_like(follower_default_kp)
        self._follower_zero_kd = np.zeros_like(follower_default_kd)

    def _apply_bilateral_gains_for_mode(self) -> None:
        """Apply KP/KD according to current toggled control mode."""
        if self._left_client is None or self._right_client is None:
            return
        try:
            if self.is_leader_role:
                if self._control_mode == "leader_to_follower":
                    kp = self._leader_zero_kp
                    kd = self._leader_zero_kd
                else:
                    kp = self._leader_default_kp * self.bilateral_kp
                    kd = self._leader_default_kd * self.bilateral_kd
            else:
                kp = self._follower_default_kp * self.bilateral_kp
                kd = self._follower_default_kd * self.bilateral_kd

            self._left_client.update_kp_kd(kp, kd)
            self._right_client.update_kp_kd(kp, kd)

        except Exception as e:
            logger.warning(f"Failed to update KP/KD gains in mode {self._control_mode}: {e}")

    @classmethod
    def _control_mode_from_flag(cls, control_flag: float) -> str:
        """Map the shared control flag onto the follower's three control modes."""
        if abs(float(control_flag) - cls._TAKEOVER_ALIGNMENT_FLAG) <= 1e-6:
            return cls._CONTROL_MODE_ALIGNMENT
        if float(control_flag) > cls._TAKEOVER_ALIGNMENT_FLAG:
            return cls._CONTROL_MODE_TELEOP
        return cls._CONTROL_MODE_AUTONOMOUS

    @staticmethod
    def _valid_takeover_joint_state(state: Optional[np.ndarray]) -> bool:
        if state is None:
            return False
        state_array = np.asarray(state, dtype=np.float64).reshape(-1)
        return state_array.size >= 14 and bool(np.all(np.isfinite(state_array[:14])))

    def _set_follower_control_mode_from_flag(self, control_flag: float) -> None:
        """Apply a control-flag transition, capturing a hold pose for flag 0.5."""
        desired_mode = self._control_mode_from_flag(control_flag)
        if desired_mode == self._control_mode:
            return

        if desired_mode == self._CONTROL_MODE_ALIGNMENT:
            hold_state = getattr(self, "_latest_joint_state", None)
            if not self._valid_takeover_joint_state(hold_state):
                hold_state = self.get_robot_state()
            if not self._valid_takeover_joint_state(hold_state):
                logger.error(
                    "Cannot enter takeover alignment: no valid follower joint state; keeping autonomous control"
                )
                return

            self._takeover_hold_joint_pos = np.asarray(hold_state, dtype=np.float64).reshape(-1)[:14].copy()
            self._command_joint_pos(
                self._left_client,
                self._takeover_hold_joint_pos[:7],
                "command_joint_pos_left_takeover_alignment_hold",
            )
            self._command_joint_pos(
                self._right_client,
                self._takeover_hold_joint_pos[7:14],
                "command_joint_pos_right_takeover_alignment_hold",
            )
            logger.info("I2RT follower holding current pose during leader takeover alignment")
        else:
            self._takeover_hold_joint_pos = None

        self._control_mode = desired_mode
        self._apply_bilateral_gains_for_mode()
        logger.info(f"I2RT follower control mode set from flag: {self._control_mode}")

    def _read_leader_joint_state_for_takeover(self) -> Optional[np.ndarray]:
        """Read the leader state after an alignment command without handling buttons."""
        try:
            left_pos, _ = self._leader_get_info(self._left_client, "get_info_left_leader_takeover_alignment")
            right_pos, _ = self._leader_get_info(self._right_client, "get_info_right_leader_takeover_alignment")
            state = np.concatenate([left_pos, right_pos], axis=0)
            if not self._valid_takeover_joint_state(state):
                return None
            return self._fit_state_shape(state)
        except Exception as exc:
            logger.warning(f"Failed to read leader state during takeover alignment: {exc}")
            return None

    def _align_leader_for_takeover(self, current_leader_state: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Hold the follower and move the leader onto its joint pose before takeover."""
        if not self._valid_takeover_joint_state(current_leader_state):
            logger.error("Cannot align leader for takeover: current leader state is invalid")
            return None

        follower_target = self._read_state_from_the_other_shm()
        if not self._valid_takeover_joint_state(follower_target):
            logger.error("Cannot align leader for takeover: follower state SHM is unavailable")
            return None

        # Publish the third state before waiting for the follower's loop to capture
        # and hold its current pose.  The rollout saver also uses this value to
        # omit the complete alignment interval from every recorded stream.
        self._write_leader_control_flag(self._TAKEOVER_ALIGNMENT_FLAG, time.time_ns())
        control_period_s = 1.0 / max(float(self.command_fps), 1.0)
        time.sleep(control_period_s)

        # Re-read after one control period so the target reflects the pose the
        # follower has just latched, rather than a pre-hold sample.
        held_follower_state = self._read_state_from_the_other_shm()
        if self._valid_takeover_joint_state(held_follower_state):
            follower_target = held_follower_state

        current = np.asarray(current_leader_state, dtype=np.float64).reshape(-1)
        target = np.asarray(follower_target, dtype=np.float64).reshape(-1)
        left_current = current[:6]
        right_current = current[7:13]
        left_target = target[:6]
        right_target = target[7:13]

        duration_s = self.takeover_alignment_duration_s
        interpolation_steps = max(1, int(np.ceil(duration_s * max(float(self.command_fps), 1.0))))
        interpolation_start = time.monotonic()
        for step in range(1, interpolation_steps + 1):
            alpha = step / interpolation_steps
            self._command_joint_pos(
                self._left_client,
                left_current * (1.0 - alpha) + left_target * alpha,
                "command_joint_pos_left_takeover_alignment",
            )
            self._command_joint_pos(
                self._right_client,
                right_current * (1.0 - alpha) + right_target * alpha,
                "command_joint_pos_right_takeover_alignment",
            )
            self._write_leader_control_flag(self._TAKEOVER_ALIGNMENT_FLAG, time.time_ns())
            step_deadline = interpolation_start + duration_s * (step / interpolation_steps)
            remaining_s = step_deadline - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)

        # Keep commanding the target for the configured settling period.
        # Joint error is logged only; it never gates or shortens human takeover.
        settle_deadline = time.monotonic() + self.takeover_alignment_settle_timeout_s
        aligned_state: Optional[np.ndarray] = None
        max_joint_error = float("inf")
        while True:
            measured_state = self._read_leader_joint_state_for_takeover()
            if measured_state is not None:
                aligned_state = measured_state
                measured = np.asarray(measured_state, dtype=np.float64).reshape(-1)
                joint_error = np.concatenate([measured[:6] - left_target, measured[7:13] - right_target])
                max_joint_error = float(np.max(np.abs(joint_error)))

            if time.monotonic() >= settle_deadline:
                break
            self._command_joint_pos(
                self._left_client,
                left_target,
                "command_joint_pos_left_takeover_alignment_settle",
            )
            self._command_joint_pos(
                self._right_client,
                right_target,
                "command_joint_pos_right_takeover_alignment_settle",
            )
            self._write_leader_control_flag(self._TAKEOVER_ALIGNMENT_FLAG, time.time_ns())
            time.sleep(control_period_s)

        if aligned_state is None:
            logger.error("Leader takeover alignment produced no valid measured state; keeping policy control")
            self._write_leader_control_flag(0.0, time.time_ns())
            return None

        # Make the aligned leader state visible before publishing flag=1.  This
        # ordering prevents the follower from consuming the pre-alignment leader
        # sample on its first teleoperation cycle.
        if getattr(self, "shared_memory", None) is not None:
            self._write_array_to_shm_with_timestamp(aligned_state, time.time_ns())
        logger.info(f"Leader takeover alignment phase finished: max_joint_error={max_joint_error:.6f} rad")
        return aligned_state

    def _handle_mode_transition(
        self,
        button_pressed: bool,
        current_leader_state: Optional[np.ndarray] = None,
        button_down: bool | None = None,
    ) -> Optional[np.ndarray]:
        """Handle button press and return an aligned leader-state override if used."""
        aligned_state = None
        # ``button_pressed`` is a rising-edge event.  ``button_down`` is the
        # current physical level and is used to keep the edge latched while a
        # takeover alignment call temporarily blocks this loop.
        physical_down = bool(button_pressed if button_down is None else button_down)
        button_down_latched = bool(getattr(self, "_button_down_latched", False))
        if button_pressed and not button_down_latched:
            # Set the latch before running alignment.  Alignment performs
            # blocking RPC calls and another state-reader thread can observe
            # the same physical press while it is in progress.
            self._button_down_latched = True
            self._button_release_samples = 0

            if not self.dual_control_mode:
                # Data collection mode: toggle episode recording on button press
                self._toggle_episode_recording()
            else:
                # Dual control mode: toggle between leader and autonomous mode
                entering_teleop = self._control_mode == self._CONTROL_MODE_AUTONOMOUS
                if entering_teleop and self.is_leader_role and self.takeover_alignment_enabled:
                    aligned_state = self._align_leader_for_takeover(current_leader_state)
                    if aligned_state is not None:
                        self._toggle_control_mode()
                else:
                    self._toggle_control_mode()
        elif not physical_down:
            release_samples = int(getattr(self, "_button_release_samples", 0)) + 1
            release_debounce_samples = int(getattr(self, "_button_release_debounce_samples", 3))
            self._button_release_samples = release_samples
            if release_samples >= release_debounce_samples:
                self._button_down_latched = False
                self._button_release_samples = 0

        self._last_button_pressed = self._button_down_latched
        return aligned_state

    def _toggle_episode_recording(self) -> None:
        """Toggle episode recording state and write appropriate signal file."""
        if not self._is_recording_episode:
            # Start episode
            self._is_recording_episode = True
            self._write_episode_signal("start")
            logger.info("Episode started via button press")
        else:
            # Stop episode
            self._is_recording_episode = False
            self._write_episode_signal("end")
            logger.info("Episode stopped via button press")

    def _write_episode_signal(self, signal_type: str) -> None:
        """Write episode signal file for data saver to detect."""
        signal_path = self._episode_signal_pattern.format(signal=signal_type)
        try:
            with open(signal_path, "w") as f:
                f.write(f"{time.time():.6f}")
            logger.info(f"Wrote episode {signal_type} signal to {signal_path}")
        except Exception as e:
            logger.error(f"Failed to write episode signal {signal_type}: {e}")

    def _toggle_control_mode(self) -> None:
        """Toggle between leader_to_follower and follower_autonomous modes."""
        self._control_mode = (
            self._CONTROL_MODE_AUTONOMOUS
            if self._control_mode == self._CONTROL_MODE_TELEOP
            else self._CONTROL_MODE_TELEOP
        )
        self._apply_bilateral_gains_for_mode()
        logger.info(f"I2RT control mode toggled to: {self._control_mode}")

    def _rpc_server_mode(self) -> str:
        return "leader_server" if self.is_leader_role else "follower_server"

    def _rpc_endpoints_ready(self) -> bool:
        """Check whether both RPC endpoints are reachable and serving."""
        try:
            portal_module = import_portal_with_wake_pipe_fix()
            left_client = portal_module.Client(f"{self.left_server_host}:{self.left_server_port}")
            right_client = portal_module.Client(f"{self.right_server_host}:{self.right_server_port}")
            left_client.get_info().result(timeout=1.0)
            right_client.get_info().result(timeout=1.0)
            return True
        except Exception:
            return False

    def _launch_rpc_server_process(self, can_channel: str, server_port: int) -> Optional[subprocess.Popen]:
        """Launch a minimum_gello RPC server process for one arm.

        Uses start_new_session=True so the server process does not receive
        SIGINT from the terminal when the user presses Ctrl+C.  This ensures
        the servers stay alive long enough for _go_rest() to send the final
        joint commands during shutdown.
        """
        cmd = [
            self.server_launch_python,
            self.minimum_gello_script,
            "--can_channel",
            can_channel,
            "--gripper",
            self.gripper_type,
            "--mode",
            self._rpc_server_mode(),
            "--server_port",
            str(server_port),
        ]
        logger.info(f"Launching I2RT RPC server: {' '.join(cmd)}")
        try:
            return subprocess.Popen(cmd, start_new_session=True)
        except Exception as e:
            logger.error(f"Failed to launch minimum_gello process ({can_channel}, {server_port}): {e}")
            return None

    def _ensure_rpc_servers_running(self) -> None:
        """
        Ensure minimum_gello server processes are running before client init.
        If endpoints are already up, skip launching.
        """
        if len(self._server_processes) == 2:
            return

        left_proc = self._launch_rpc_server_process(self.left_can_channel, self.left_server_port)
        right_proc = self._launch_rpc_server_process(self.right_can_channel, self.right_server_port)
        self._server_processes = [proc for proc in [left_proc, right_proc] if proc is not None]
        if len(self._server_processes) != 2:
            logger.error("Failed to launch both minimum_gello RPC server processes")
            return

        logger.info(f"Waiting {self.server_startup_wait_s:.1f}s for minimum_gello RPC servers")
        time.sleep(self.server_startup_wait_s)
        for _ in range(10):
            if self._rpc_endpoints_ready():
                logger.info("minimum_gello RPC servers are ready")
                return
            time.sleep(0.2)
        logger.error("minimum_gello RPC servers are not reachable after launch")

    def _fit_state_shape(self, state_array: np.ndarray) -> np.ndarray:
        """Pad/crop state to configured shape for robust SHM writes."""
        expected_size = int(np.prod(self.data_shape))
        flattened = state_array.astype(get_dtype(self.data_dtype), copy=False).reshape(-1)
        if flattened.size == expected_size:
            return flattened.reshape(self.data_shape)
        shaped = np.zeros(expected_size, dtype=get_dtype(self.data_dtype))
        copy_size = min(expected_size, flattened.size)
        shaped[:copy_size] = flattened[:copy_size]
        return shaped.reshape(self.data_shape)

    def _map_left_leader_gripper_to_right_range(self, joint_pos: np.ndarray, source: str) -> np.ndarray:
        """Compensate the left YAM teaching-handle trigger to match the right trigger range."""
        if not getattr(self, "is_leader_role", False) or "left_leader" not in source or joint_pos.size < 7:
            return joint_pos

        source_min, source_max = self._LEFT_LEADER_GRIPPER_SOURCE_RANGE
        target_min, target_max = self._LEFT_LEADER_GRIPPER_TARGET_RANGE
        source_span = source_max - source_min
        if source_span <= 0.0:
            return joint_pos

        mapped = target_min + (float(joint_pos[6]) - source_min) * (target_max - target_min) / source_span
        joint_pos = joint_pos.copy()
        joint_pos[6] = np.clip(mapped, 0.0, 1.0)
        return joint_pos

    def _leader_get_info(self, client, source: str) -> tuple[np.ndarray, np.ndarray]:
        """Read leader info from RPC and normalize return format."""
        result = client.get_info().result()

        if isinstance(result, (list, tuple)) and len(result) == 2:
            joint_pos = np.asarray(result[0], dtype=np.float64).reshape(-1)
            button_state = np.asarray(result[1], dtype=np.float64).reshape(-1)
            joint_pos = self._map_left_leader_gripper_to_right_range(joint_pos, source)
            return joint_pos, button_state
        joint_pos = np.asarray(result, dtype=np.float64).reshape(-1)
        joint_pos = self._map_left_leader_gripper_to_right_range(joint_pos, source)
        return joint_pos, np.zeros(1, dtype=np.float64)

    def _command_joint_pos(self, client, joint_pos: np.ndarray, source: str) -> None:
        """Send a joint command through the existing RPC client."""
        client.command_joint_pos(joint_pos)

    def _follower_get_info(self, client, source: str) -> np.ndarray:
        """Read follower joint+gripper state from RPC."""
        result = client.get_info().result()

        return np.asarray(result, dtype=np.float64).reshape(-1)

    def _smooth_joint_move(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        left_current: np.ndarray,
        right_current: np.ndarray,
        steps: int = 100,
        dt: float = 0.03,
    ) -> None:
        """Linearly interpolate both arms from current to target joint positions."""
        left_target = np.asarray(left_target, dtype=np.float64).reshape(-1)
        right_target = np.asarray(right_target, dtype=np.float64).reshape(-1)
        left_current = np.asarray(left_current, dtype=np.float64).reshape(-1)
        right_current = np.asarray(right_current, dtype=np.float64).reshape(-1)
        if steps <= 0:
            self._command_joint_pos(self._left_client, left_target, "command_joint_pos_left_smooth_joint_move")
            self._command_joint_pos(self._right_client, right_target, "command_joint_pos_right_smooth_joint_move")
            return
        for i in range(steps):
            alpha = i / steps
            left_interp = left_current * (1.0 - alpha) + left_target * alpha
            right_interp = right_current * (1.0 - alpha) + right_target * alpha
            self._command_joint_pos(self._left_client, left_interp, "command_joint_pos_left_smooth_joint_move")
            self._command_joint_pos(self._right_client, right_interp, "command_joint_pos_right_smooth_joint_move")
            time.sleep(dt)

    def _create_leader_control_flag_shm(self) -> None:
        """Create a leader-owned control flag SHM for shared-control takeover."""
        if not self.is_leader_role:
            return
        if self._control_flag_shm is not None:
            return
        if self._control_flag_shm_name is None:
            self._control_flag_shm_name = f"{self.device_name}_{self.device_id}_control_flag"
        try:
            control_flag_shape = (1,)
            numpy_dtype = get_dtype(self.data_dtype)
            bytes_per_element = np.dtype(numpy_dtype).itemsize
            data_size = np.prod(control_flag_shape) * bytes_per_element
            frame_size = FRAME_HEADER_SIZE + data_size
            total_size = calculate_device_shm_size(self._leader_flag_buffer_size, data_size)

            try:
                existing = shm.SharedMemory(name=self._control_flag_shm_name, create=False)
                existing.close()
                existing.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug(f"Failed to remove old leader control flag SHM: {e}")

            self._control_flag_shm = shm.SharedMemory(
                name=self._control_flag_shm_name,
                create=True,
                size=total_size,
            )
            self._leader_flag_owner = True
            self._leader_flag_write_index = 0

            device_header = pack_device_header(
                self.device_name,
                self.device_id,
                self.fps,
                self.data_dtype,
                control_flag_shape,
                frame_size,
                self.hardware_latency_ms,
            )
            self._control_flag_shm.buf[:DEVICE_HEADER_SIZE] = device_header
            buffer_header = pack_buffer_header(self._leader_flag_buffer_size, 0, 0)
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header
            logger.info(f"Created leader control flag SHM: {self._control_flag_shm_name}")
        except Exception as e:
            logger.error(f"Failed to create leader control flag SHM: {e}")
            self._control_flag_shm = None
            self._leader_flag_owner = False

    def _write_leader_control_flag(self, flag_value: float, timestamp_ns: int) -> None:
        """Write leader button ownership flag to SHM."""
        if not self.is_leader_role:
            return
        if self._control_flag_shm is None:
            self._create_leader_control_flag_shm()
        if self._control_flag_shm is None:
            return
        try:
            device_header_data = bytes(self._control_flag_shm.buf[:DEVICE_HEADER_SIZE])
            device_info = unpack_device_header(device_header_data)
            frame_size = device_info["frame_size"]
            frame_offset = get_control_frame_offset(self._leader_flag_write_index, frame_size)
            frame_header = pack_frame_header(timestamp_ns)
            self._control_flag_shm.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = frame_header

            flag_bytes = np.asarray([flag_value], dtype=get_dtype(self.data_dtype)).tobytes()
            data_start = frame_offset + FRAME_HEADER_SIZE
            self._control_flag_shm.buf[data_start : data_start + len(flag_bytes)] = flag_bytes

            current_frames_count = min(self._leader_flag_write_index + 1, self._leader_flag_buffer_size)
            buffer_header = pack_buffer_header(
                self._leader_flag_buffer_size,
                current_frames_count,
                self._leader_flag_write_index,
            )
            self._control_flag_shm.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header
            self._leader_flag_write_index = (self._leader_flag_write_index + 1) % self._leader_flag_buffer_size
        except Exception as e:
            logger.warning(f"Failed to write leader control flag: {e}")

    def _calculate_pose_difference(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        """
        Calculate L2 norm difference between two joint poses.

        Args:
            pose1: First joint pose array
            pose2: Second joint pose array

        Returns:
            L2 norm of the difference
        """
        try:
            # Handle potential shape mismatches (14D with gripper vs 12D without)
            min_len = min(len(pose1), len(pose2))
            if min_len == 0:
                return float("inf")

            # Compare only joint positions (exclude gripper if present)
            # For I2RT: compare first 6 joints of each arm (12 total)
            compare_indices = list(range(6)) + list(range(7, 13))  # Skip gripper dims at [6] and [13]

            p1_joints = pose1[compare_indices] if len(pose1) >= 14 else pose1[:12]
            p2_joints = pose2[compare_indices] if len(pose2) >= 14 else pose2[:12]

            return float(np.linalg.norm(p1_joints - p2_joints))
        except Exception as e:
            logger.warning(f"Failed to calculate pose difference: {e}")
            return 0.0

    def _read_state_from_the_other_shm(self) -> Optional[np.ndarray]:
        """
        Read the other robot instance state from configured control SHMs.

        Returns:
            Other robot state array or None if unavailable
        """
        if not self.control_shm_names or len(self.control_shm_names) == 0:
            return None

        try:
            # Only consider explicit robot state streams (e.g. "*_data"), excluding
            # this robot's own default control stream.
            state_candidates: List[str] = []
            for shm_name in self.control_shm_names:
                if shm_name == self.default_control_shm_name:
                    continue
                if not shm_name.endswith("_data"):
                    continue
                state_candidates.append(shm_name)

            if len(state_candidates) == 0:
                logger.debug("No valid other-state SHM (*_data) found in control_shm_names")
                return None
            if len(state_candidates) > 1:
                logger.warning(
                    f"Multiple state SHM candidates found for other robot: {state_candidates}, "
                    f"using first: {state_candidates[0]}"
                )

            other_shm_name = state_candidates[0]

            # Use the same logic as reading control commands
            _timestamp_ns, other_state = self.read_latest_control_command(other_shm_name, min_timestamp_ns=0)

            if other_state is None:
                logger.info(f"Other robot state is None from SHM: {other_shm_name}")
                return None

            return other_state

        except Exception as e:
            logger.error(f"Failed to read other robot state from SHM: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

        # if log_count is not None:
        #     logger.info(
        #         f"I2RT monitored_call_enter source={source} "
        #         f"inflight_count={log_count} role={'leader' if self.is_leader_role else 'follower'}"
        #     )

    @staticmethod
    def _portal_client_stats(client: Any) -> Dict[str, Any]:
        """Read and reset one Portal client's interval counters safely."""
        stats_fn = getattr(client, "stats", None)
        if not callable(stats_fn):
            return {}

        client_lock = getattr(client, "lock", None)
        if client_lock is None:
            return dict(stats_fn())
        with client_lock:
            return dict(stats_fn())

    def _configure_portal_client_diagnostics(self, client: Any) -> None:
        """Fill the receive counter omitted by the installed Portal client."""
        if self._portal_stats_interval_s <= 0.0 or client is None:
            return
        if getattr(client, "_vam_recv_counter_enabled", False):
            return

        callbacks = getattr(getattr(client, "socket", None), "callbacks_recv", None)
        if callbacks is None or not hasattr(client, "recvrate"):
            return

        def count_response(_data: Any) -> None:
            client_lock = getattr(client, "lock", None)
            if client_lock is None:
                client.recvrate[0] += 1
                return
            with client_lock:
                client.recvrate[0] += 1

        callbacks.append(count_response)
        client._vam_recv_counter_enabled = True

    def _log_portal_client_stats_if_due(self, *, force: bool = False) -> None:
        """Log Portal queue/rate metrics when the reconnect experiment enables it."""
        if self._portal_stats_interval_s <= 0.0:
            return

        now = time.monotonic()
        if not force and now - self._portal_stats_last_log_monotonic < self._portal_stats_interval_s:
            return
        self._portal_stats_last_log_monotonic = now

        clients = (
            ("left", self._left_client, self.left_server_host, self.left_server_port),
            ("right", self._right_client, self.right_server_host, self.right_server_port),
        )
        for side, client, host, port in clients:
            if client is None:
                logger.info(
                    f"I2RT portal_stats run_id={self._diagnostic_run_id} "
                    f"device={self.device_id} "
                    f"role={'leader' if self.is_leader_role else 'follower'} "
                    f"side={side} endpoint={host}:{port} client=none"
                )
                continue
            try:
                stats = self._portal_client_stats(client)
                wake_pipe = portal_wake_pipe_stats(client)
                logger.info(
                    f"I2RT portal_stats run_id={self._diagnostic_run_id} "
                    f"device={self.device_id} "
                    f"role={'leader' if self.is_leader_role else 'follower'} "
                    f"side={side} endpoint={host}:{port} "
                    f"connected={getattr(client, 'connected', None)} "
                    f"inflight={stats.get('inflight')} "
                    f"maxinflight={getattr(client, 'maxinflight', None)} "
                    f"numsend={stats.get('numsend')} numrecv={stats.get('numrecv')} "
                    f"sendrate={stats.get('sendrate')} "
                    f"recvrate={stats.get('recvrate')} "
                    f"waitmean={stats.get('waitmean')} "
                    f"signal_pending={wake_pipe.get('signal_pending')} "
                    f"signal_capacity={wake_pipe.get('signal_capacity')}"
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to collect Portal stats for device={self.device_id} "
                    f"side={side} endpoint={host}:{port}: {exc}"
                )

    def _run_main_control_loop(self, loop_generation: int) -> None:
        """Run one generation of the main control loop."""
        logger.info(
            f"[I2RT main-loop g{loop_generation}] started role={'leader' if self.is_leader_role else 'follower'}"
        )
        try:
            assert self.fps >= self.command_fps
            iteration = 1
            command_iteration = int(self.fps // self.command_fps)
            while self.running and loop_generation == self._main_control_loop_generation:
                start_time = time.time()

                if hasattr(self, "_check_and_consume_and_reply_robot_halt_command"):
                    if hasattr(self, "_last_robot_halt_check_time"):
                        if time.time() - self._last_robot_halt_check_time > 0.1:  # Check every 100ms
                            self._check_and_consume_and_reply_robot_halt_command()
                            self._last_robot_halt_check_time = time.time()
                    else:
                        self._last_robot_halt_check_time = time.time()
                        self._check_and_consume_and_reply_robot_halt_command()

                # Check for external home command (from GUI reset)
                # Only check periodically to avoid performance impact
                if hasattr(self, "_last_home_check_time"):
                    if time.time() - self._last_home_check_time > 0.1:  # Check every 100ms
                        self._check_and_execute_home_command()
                        self._last_home_check_time = time.time()
                else:
                    self._last_home_check_time = time.time()
                    self._check_and_execute_home_command()

                if hasattr(self, "_check_and_execute_teleop_start_command"):
                    if hasattr(self, "_last_teleop_start_check_time"):
                        if time.time() - self._last_teleop_start_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_start_command()
                            self._last_teleop_start_check_time = time.time()
                    else:
                        self._last_teleop_start_check_time = time.time()
                        self._check_and_execute_teleop_start_command()

                if hasattr(self, "_check_and_execute_teleop_discard_command"):
                    if hasattr(self, "_last_teleop_discard_check_time"):
                        if time.time() - self._last_teleop_discard_check_time > 0.1:  # Check every 100ms
                            self._check_and_execute_teleop_discard_command()
                            self._last_teleop_discard_check_time = time.time()
                    else:
                        self._last_teleop_discard_check_time = time.time()
                        self._check_and_execute_teleop_discard_command()
                # Check for control mode switch BEFORE reading commands
                # This ensures we set the timestamp filter before reading any actions
                if self.dual_control_mode:
                    if self._control_flag_shm is None and not getattr(self, "is_leader_role", False):
                        self._connect_control_flag_shm()
                    current_control_flag = self._read_control_flag()

                    # Detect transition from human to policy control
                    if not hasattr(self, "_last_control_flag"):
                        self._last_control_flag = 0.0
                    if self._last_control_flag > 0.5 and current_control_flag <= 0.5:
                        # Clear control SHM to prevent stale actions from being executed
                        self._clear_control_shm()
                        # Record when policy control started - used to filter stale actions
                        self._policy_control_start_timestamp = time.time_ns()
                        logger.info(
                            f"Control mode switch detected in main loop: cleared control SHM, timestamp={self._policy_control_start_timestamp}"
                        )

                    if getattr(self, "is_follower_role", False):
                        alignment_flag = self._TAKEOVER_ALIGNMENT_FLAG
                        alignment_transition = (
                            abs(current_control_flag - alignment_flag) <= 1e-6
                            or abs(self._last_control_flag - alignment_flag) <= 1e-6
                        )
                        if alignment_transition:
                            # flag=0.5 captures and holds the follower pose while
                            # the leader aligns, without enabling teleoperation.
                            self._set_follower_control_mode_from_flag(current_control_flag)
                        elif (current_control_flag > 0.5) != (self._last_control_flag > 0.5):
                            # Preserve the legacy binary 0/1 transition path
                            # exactly when takeover alignment is not involved.
                            self._toggle_control_mode()

                    self._last_control_flag = current_control_flag

                control_dict = {}
                for shm_name in self.control_shm_names:
                    # Read latest control command
                    # Timestamp filtering rejects commands written before policy handover.
                    policy_start_ts = getattr(self, "_policy_control_start_timestamp", 0)
                    grace_period_ns = getattr(self, "_policy_control_grace_period_ns", 0)
                    last_home_command_timestamp = getattr(self, "_last_home_command_timestamp", 0)
                    min_timestamp_ns = policy_start_ts + grace_period_ns if policy_start_ts > 0 else 0
                    min_timestamp_ns = max(min_timestamp_ns, last_home_command_timestamp)
                    timestamp_ns, control_array = self.read_latest_control_command(shm_name, min_timestamp_ns)
                    control_dict[shm_name] = control_array

                if any([value is not None for value in control_dict.values()]):  # Any non-None values in control_dict
                    if iteration % command_iteration == 0:
                        # Skip command execution while recovery is in progress.
                        if not self._home_command_active and not self._robot_halt_command_active:
                            self.execute_control_command(control_dict)
                iteration = iteration % command_iteration + 1
                # Get current robot state
                state_array = self.get_robot_state()
                # Write state to shared memory
                if state_array is not None:
                    self._write_array_to_shm_with_timestamp(state_array, time.time_ns())

                # Maintain control frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info(f"Stopping {self.device_name}_{self.device_id} server...")
        except Exception as e:
            logger.error(f"Error in robot control loop: {e}")
        finally:
            is_latest_generation = loop_generation == self._main_control_loop_generation
            logger.info(
                f"[I2RT main-loop g{loop_generation}] exiting is_latest={is_latest_generation} running={self.running}"
            )

    def get_robot_state(self) -> Optional[np.ndarray]:
        try:
            if self._left_client is None or self._right_client is None:
                logger.error("I2RT robot state clients are not connected")
                return None

            timestamp_ns = time.time_ns()
            if self.is_leader_role:
                transition_lock = (
                    self._dagger_mode_transition_lock
                    if self.dual_control_mode and self.takeover_alignment_enabled
                    else None
                )
                if transition_lock is not None:
                    transition_lock.acquire()
                try:
                    left_pos, btn_left = self._leader_get_info(
                        self._left_client, "get_info_left_leader_get_robot_state"
                    )
                    right_pos, btn_right = self._leader_get_info(
                        self._right_client, "get_info_right_leader_get_robot_state"
                    )
                    state = np.concatenate([left_pos, right_pos], axis=0)
                    button_down = bool(btn_left[0] > 0.5 or btn_right[0] > 0.5)
                    button_pressed = button_down and not getattr(self, "_button_down_latched", False)
                    aligned_state = self._handle_mode_transition(
                        button_pressed,
                        current_leader_state=state,
                        button_down=button_down,
                    )
                    if aligned_state is not None:
                        state = aligned_state
                        left_pos = state[:7]
                        right_pos = state[7:14]
                        timestamp_ns = time.time_ns()
                    self._last_button_left = btn_left[0]
                    self._last_button_right = btn_right[0]
                    if button_pressed:
                        self._command_joint_pos(
                            self._left_client, left_pos[:6], "command_joint_pos_left_get_robot_state_leader"
                        )
                        self._command_joint_pos(
                            self._right_client, right_pos[:6], "command_joint_pos_right_get_robot_state_leader"
                        )
                    # Control flag stores mode state: 1.0 means leader controls follower.
                    self._write_leader_control_flag(
                        1.0 if self._control_mode == "leader_to_follower" else 0.0, timestamp_ns
                    )
                    return self._fit_state_shape(state)
                finally:
                    if transition_lock is not None:
                        transition_lock.release()

            left_pos = self._follower_get_info(self._left_client, "get_info_left_follower_get_robot_state")
            right_pos = self._follower_get_info(self._right_client, "get_info_right_follower_get_robot_state")
            state = np.concatenate([left_pos, right_pos], axis=0)
            fitted_state = self._fit_state_shape(state)
            self._latest_joint_state = fitted_state.copy()
            return fitted_state
        except Exception as e:
            logger.error(f"Failed to get I2RT robot state: {e}")
            # A zero-filled state is a valid-looking 14-D joint command whose
            # consumer can move the follower to rest. Do not publish one for a
            # failed hardware read; the control loop skips SHM writes for None.
            return None

    def _check_and_consume_and_reply_robot_halt_command(self) -> None:
        """Check if the robot halt command is present and consume it."""
        from pathlib import Path

        robot_halt_command_file = Path(
            self._robot_halt_command_pattern.format(device_name=self.device_name, device_id=self.device_id)
        )
        if robot_halt_command_file.exists():
            logger.info(
                f"ROBOT HALT COMMAND FILE FOUND FOR {self.device_name}_{self.device_id}, SETTING ROBOT HALT COMMAND ACTIVE"
            )
            self._robot_halt_command_active = True
            robot_halt_command_file.unlink()
            logger.info(f"Consumed robot halt command file for {self.device_name}_{self.device_id}")
            # reply
            robot_halt_reply_file = Path(
                self._robot_halt_reply_pattern.format(device_name=self.device_name, device_id=self.device_id)
            )
            robot_halt_reply_file.parent.mkdir(parents=True, exist_ok=True)
            robot_halt_reply_file.touch()
            logger.info(
                f"Replied robot halt command for {self.device_name}_{self.device_id}, file: {robot_halt_reply_file}"
            )
        else:
            pass

    def _update_trusted_get_info_state(self, state_left: np.ndarray, state_right: np.ndarray) -> bool:
        def _format_float_list(values: np.ndarray) -> str:
            return "[" + ", ".join(f"{float(value):.3f}" for value in values.tolist()) + "]"

        all_none = self._last_trusted_get_info_state_left is None and self._last_trusted_get_info_state_right is None
        all_not_none = (
            self._last_trusted_get_info_state_left is not None and self._last_trusted_get_info_state_right is not None
        )
        if not (all_none or all_not_none):
            logger.info(
                "both last_get_info_state_left and last_trusted_get_info_state_left and last_get_info_state_right and last_trusted_get_info_state_right must be None or not None"
            )
            return False

        if all_none:
            with self._trusted_state_lock:
                self._last_trusted_get_info_state_left = state_left
                self._last_trusted_get_info_state_right = state_right
                return True
        else:
            left_state_diff = np.abs(state_left - self._last_trusted_get_info_state_left)
            right_state_diff = np.abs(state_right - self._last_trusted_get_info_state_right)

            # leave out the gripper state
            if len(left_state_diff) == 7:
                left_state_diff = left_state_diff[:6]
            if len(right_state_diff) == 7:
                right_state_diff = right_state_diff[:6]

            left_is_trusted = np.all(
                left_state_diff < self.state_outlier_threshold
            )  # every joint and gripper is within the outlier threshold
            right_is_trusted = np.all(
                right_state_diff < self.state_outlier_threshold
            )  # every joint and gripper is within the outlier threshold

            # left judgement
            if left_is_trusted:
                with self._trusted_state_lock:
                    self._last_trusted_get_info_state_left = state_left
            else:
                logger.warning(
                    f"state_outlying! role: {'leader' if self.is_leader_role else 'follower'}, state_left: {_format_float_list(state_left)}, trusted_state_left: {_format_float_list(self._last_trusted_get_info_state_left)}, outlier_index: {np.where(left_state_diff > self.state_outlier_threshold)[0].tolist()}"
                )

            # right judgement
            if right_is_trusted:
                with self._trusted_state_lock:
                    self._last_trusted_get_info_state_right = state_right
            else:
                logger.warning(
                    f"state_outlying! role: {'leader' if self.is_leader_role else 'follower'}, state_right: {_format_float_list(state_right)}, trusted_state_right: {_format_float_list(self._last_trusted_get_info_state_right)}, outlier_index: {np.where(right_state_diff > self.state_outlier_threshold)[0].tolist()}"
                )

            return left_is_trusted and right_is_trusted

    def _monitor_trusted_state(self) -> None:
        """
        Background thread that keeps robot trusted state updated.
        """
        logger.info("Trusted state monitoring thread started")

        while self._trusted_state_monitor_evt.is_set():
            try:
                self._log_portal_client_stats_if_due()
                if self._left_client is None or self._right_client is None:
                    logger.error(
                        f"Trusted state monitor sees invalid clients: left={self._left_client is None}, right={self._right_client is None}"
                    )
                    time.sleep(self._trusted_state_check_interval)
                    continue

                if self.is_leader_role:
                    left_pos, btn_left = self._leader_get_info(self._left_client, "get_info_left_leader_trusted_state")
                    right_pos, btn_right = self._leader_get_info(
                        self._right_client, "get_info_right_leader_trusted_state"
                    )
                else:
                    left_pos = self._follower_get_info(self._left_client, "get_info_left_follower_trusted_state")
                    right_pos = self._follower_get_info(self._right_client, "get_info_right_follower_trusted_state")

                if not self._robot_halt_command_active:
                    self._update_trusted_get_info_state(left_pos, right_pos)
                else:
                    logger.warning(
                        f"robot halt command is active, skipping trusted state update for {self.device_name}_{self.device_id}"
                    )

                time.sleep(self._trusted_state_check_interval)

            except Exception as e:
                logger.error(f"Error in trusted state monitoring: {e}")
                time.sleep(self._trusted_state_check_interval)

        logger.info("Trusted state monitoring thread stopped")

    def execute_control_command(self, control_dict: Dict[str, np.ndarray]) -> None:
        try:
            if self._left_client is None or self._right_client is None:
                return
            if self.is_follower_role:
                if self._control_mode == self._CONTROL_MODE_ALIGNMENT:
                    hold_state = self._takeover_hold_joint_pos
                    if not self._valid_takeover_joint_state(hold_state):
                        logger.warning("I2RT follower takeover alignment has no valid hold pose")
                        return
                    control_array = np.asarray(hold_state, dtype=np.float64).reshape(-1)
                else:
                    use_policy = self.policy_control_only or (
                        self.dual_control_mode and self._control_mode != self._CONTROL_MODE_TELEOP
                    )
                    if use_policy:
                        policy_array = self._get_control_array(control_dict)
                        if policy_array is None:
                            return
                        control_array = np.asarray(policy_array, dtype=np.float64).reshape(-1)
                    else:
                        teleop_key = self._find_teleoperator_key(control_dict)
                        if teleop_key is None or control_dict.get(teleop_key) is None:
                            return
                        control_array = np.asarray(control_dict[teleop_key], dtype=np.float64).reshape(-1)

                if control_array.size != 14:
                    logger.warning(f"I2RT follower expects 14D command, got {control_array.size}")
                    return

                self._command_joint_pos(
                    self._left_client, control_array[:7], "command_joint_pos_left_execute_control_follower"
                )
                self._command_joint_pos(
                    self._right_client, control_array[7:14], "command_joint_pos_right_execute_control_follower"
                )
                return

            if not self.dual_control_mode:
                return
            if self._control_mode != "follower_autonomous":
                return
            follower_state = self._get_control_array(control_dict)
            if follower_state is None:
                return

            follow_array = np.asarray(follower_state, dtype=np.float64).reshape(-1)
            if follow_array.size < 14:
                logger.warning(f"I2RT leader follow expects at least 14D source, got {follow_array.size}")
                return
            # Leader arm accepts only 6-DoF command per arm, so gripper dimension is dropped.
            self._command_joint_pos(
                self._left_client, follow_array[:6], "command_joint_pos_left_execute_control_leader"
            )
            self._command_joint_pos(
                self._right_client, follow_array[7:13], "command_joint_pos_right_execute_control_leader"
            )

        except Exception as e:
            logger.error(f"Failed to execute I2RT control command: {e}")

    def _cleanup_control_shared_memory(self) -> None:
        """Cleanup SHM resources without unlinking externally-owned SHMs."""
        if self.control_shared_memory is not None:
            for shm_name, control_shm in list(self.control_shared_memory.items()):
                if control_shm is None:
                    continue
                try:
                    control_shm.close()
                except Exception as e:
                    logger.warning(f"Failed to close SHM {shm_name}: {e}")
                try:
                    if shm_name == self.default_control_shm_name:
                        control_shm.unlink()
                        logger.info(f"Unlinked owned control SHM: {shm_name}")
                except FileNotFoundError:
                    pass
                except PermissionError:
                    pass
                except Exception as e:
                    logger.warning(f"Failed to unlink SHM {shm_name}: {e}")
                finally:
                    self.control_shared_memory[shm_name] = None

        if self._leader_flag_owner and self._control_flag_shm is not None:
            try:
                self._control_flag_shm.close()
                self._control_flag_shm.unlink()
                logger.info(f"Cleaned up leader control flag SHM: {self._control_flag_shm_name}")
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Failed to cleanup leader control flag SHM: {e}")
            finally:
                self._control_flag_shm = None
                self._leader_flag_owner = False

    def _check_and_execute_home_command(self) -> None:
        """Check for external home/teleop command files and execute if present."""
        try:
            from pathlib import Path

            home_command_file = Path(
                self._robot_home_command_pattern.format(device_name=self.device_name, device_id=self.device_id)
            )
            has_home_command = home_command_file.exists()
            if not has_home_command:
                return
            logger.info(f"Received home command for {self.device_name}_{self.device_id}")
            try:
                home_command_file.unlink()
            except Exception:
                pass

            logger.info(f"unlinking home command file for {self.device_name}_{self.device_id}")
            self._home_command_active = True
            self._clear_control_shm()
            self._control_mode = "follower_autonomous"
            self._apply_bilateral_gains_for_mode()
            success = self._go_home(reason="episode_reset")
            home_end_timestamp_ns = time.time_ns()
            self._clear_control_shm()
            self._last_home_command_timestamp = max(
                getattr(self, "_last_home_command_timestamp", 0),
                home_end_timestamp_ns,
            )
            self._policy_control_start_timestamp = max(
                getattr(self, "_policy_control_start_timestamp", 0),
                home_end_timestamp_ns,
            )
            if success:
                self._notify_policy_runner_reset()
                logger.info(
                    "I2RT home command completed successfully; "
                    f"control commands before {home_end_timestamp_ns} are ignored, "
                )
            else:
                logger.error("I2RT home command failed")

            logger.info(
                f"FINISH CHECK_AND_EXECUTE_HOME_COMMAND, HOME COMMAND ACTIVE FOR {self.device_name}_{self.device_id} IS SET TO FALSE"
            )
            self._robot_halt_command_active = False

        except Exception as e:
            logger.debug(f"I2RT home command check failed: {e}")
        finally:
            self._home_command_active = False

    def _check_and_execute_teleop_start_command(self) -> None:
        """Consume teleop-start command file and switch to teleoperation mode."""
        from pathlib import Path

        teleop_start_command_file = Path(
            self._robot_teleop_start_command_pattern.format(device_name=self.device_name, device_id=self.device_id)
        )
        has_teleop_start_command = teleop_start_command_file.exists()
        if not has_teleop_start_command:
            return
        try:
            try:
                teleop_start_command_file.unlink()
            except Exception:
                pass

            self._control_mode = "leader_to_follower"
            self._apply_bilateral_gains_for_mode()
            curr_state = self.get_robot_state()
            if curr_state is not None:
                if self.is_leader_role:
                    self._command_joint_pos(
                        self._left_client,
                        curr_state[:6],
                        "command_joint_pos_left_check_and_execute_teleop_start_leader",
                    )
                    self._command_joint_pos(
                        self._right_client,
                        curr_state[7:13],
                        "command_joint_pos_right_check_and_execute_teleop_start_leader",
                    )
                else:
                    self._command_joint_pos(
                        self._left_client,
                        curr_state[:7],
                        "command_joint_pos_left_check_and_execute_teleop_start_follower",
                    )
                    self._command_joint_pos(
                        self._right_client,
                        curr_state[7:14],
                        "command_joint_pos_right_check_and_execute_teleop_start_follower",
                    )
                logger.info(f"Set joint pos after bilateral gains: {curr_state}")
            else:
                logger.error(f"Failed to get robot state for {self.device_name}_{self.device_id}")
                raise RuntimeError(f"Failed to get robot state for {self.device_name}_{self.device_id}")

            logger.info(f"Received teleop-start command for {self.device_name}_{self.device_id}")
        except Exception as e:
            logger.warning(f"Failed to process teleop-start command: {e}")

    def _check_and_execute_teleop_discard_command(self) -> None:
        """Consume teleop-discard command file and switch to autonomous mode."""
        from pathlib import Path

        teleop_discard_command_file = Path(
            self._robot_teleop_discard_command_pattern.format(device_name=self.device_name, device_id=self.device_id)
        )
        has_teleop_discard_command = teleop_discard_command_file.exists()
        if not has_teleop_discard_command:
            return
        logger.info(f"Received teleop-discard command for {self.device_name}_{self.device_id}")
        try:
            self._is_recording_episode = not self._is_recording_episode
            teleop_discard_command_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to process teleop-discard command: {e}")

    def _clear_control_shm(self) -> None:
        """Clear robot-owned control SHM to drop stale commands."""
        try:
            if self.control_shared_memory is None or self.default_control_shm_name not in self.control_shared_memory:
                return
            control_shm = self.control_shared_memory.get(self.default_control_shm_name)
            if control_shm is None:
                return

            try:
                control_shm.buf[0] = control_shm.buf[0]
                writable_shm = control_shm
            except (PermissionError, AttributeError):
                writable_shm = connect_to_shared_memory(self.default_control_shm_name, read_only=False)
                self.control_shared_memory[self.default_control_shm_name] = writable_shm

            buf = writable_shm.buf
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            frame_size = device_header["frame_size"]
            buffer_size = buffer_header["buffer_size"]
            zero_ts = pack_frame_header(0)
            for i in range(buffer_size):
                frame_offset = get_control_frame_offset(i, frame_size)
                buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = zero_ts
                data_start = frame_offset + FRAME_HEADER_SIZE
                buf[data_start : frame_offset + frame_size] = b"\x00" * (frame_size - FRAME_HEADER_SIZE)
            reset_header = pack_buffer_header(buffer_size, 0, 0)
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = reset_header
            logger.info(f"Cleared I2RT control SHM: {self.default_control_shm_name}")
        except Exception as e:
            logger.error(f"Failed to clear I2RT control SHM: {e}")

    def _should_randomize_init_pose(self, reason: str) -> bool:
        """Return whether the current go-home call should sample a new pose."""
        if not getattr(self, "random_init_pose", False):
            return False
        reason_key = {
            "startup": "randomize_on_startup",
            "episode_reset": "randomize_on_episode_reset",
        }.get(reason, "randomize_on_episode_reset")
        return bool(self.random_init_pose_params.get(reason_key, True))

    def _get_i2rt_kinematics(self):
        """Lazily initialize the preferred available YAM FK/IK implementation."""
        if self._i2rt_kinematics is None:
            from utils.i2rt_kinematics import create_i2rt_kinematics

            self._i2rt_kinematics, self._i2rt_kinematics_backend = create_i2rt_kinematics(site_name="grasp_site")
            logger.info(f"I2RT random-init kinematics backend: {self._i2rt_kinematics_backend}")
        return self._i2rt_kinematics

    def _sample_random_init_pose_offset(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a uniform position in a ball and independent Euler deltas."""
        radius_limit = float(self.random_init_pose_params["position_radius_m"])
        if radius_limit == 0.0:
            position_delta = np.zeros(3, dtype=np.float64)
        else:
            direction = self._random_init_pose_rng.normal(size=3)
            while np.linalg.norm(direction) < 1e-12:
                direction = self._random_init_pose_rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            radius = radius_limit * float(self._random_init_pose_rng.random()) ** (1.0 / 3.0)
            position_delta = direction * radius

        rotation_range_deg = np.asarray(
            self.random_init_pose_params["rotation_euler_range_deg"],
            dtype=np.float64,
        )
        euler_delta_deg = self._random_init_pose_rng.uniform(
            low=-rotation_range_deg,
            high=rotation_range_deg,
        )
        return position_delta, euler_delta_deg

    @staticmethod
    def _i2rt_arm_joint_limits() -> tuple[np.ndarray, np.ndarray]:
        """Return the six YAM joint limits encoded in the deployed MuJoCo model."""
        lower = np.array([-2.61799, 0.0, 0.0, -1.65, -1.5708, -2.0944])
        upper = np.array([3.13, 3.65, 3.13, 1.65, 1.5708, 2.0944])
        return lower, upper

    def _solve_randomized_home_arm(
        self,
        nominal_q: np.ndarray,
        position_delta: np.ndarray,
        euler_delta_deg: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Apply an EEF pose offset to one nominal arm pose and solve IK."""
        kinematics = self._get_i2rt_kinematics()
        nominal_q = np.asarray(nominal_q, dtype=np.float64).reshape(6)
        nominal_pose = np.asarray(kinematics.fk(nominal_q, "grasp_site"), dtype=np.float64)
        target_pose = nominal_pose.copy()
        target_pose[:3, 3] += np.asarray(position_delta, dtype=np.float64)
        delta_rotation = R.from_euler("xyz", euler_delta_deg, degrees=True)
        target_pose[:3, :3] = nominal_pose[:3, :3] @ delta_rotation.as_matrix()

        success, solution = kinematics.ik(
            target_pose,
            "grasp_site",
            init_q=nominal_q,
            max_iters=int(self.random_init_pose_params.get("ik_max_iters", 200)),
            pos_threshold=float(self.random_init_pose_params.get("ik_position_tolerance_m", 1e-4)),
            ori_threshold=np.deg2rad(float(self.random_init_pose_params.get("ik_orientation_tolerance_deg", 0.2))),
            verbose=False,
        )
        solution = np.asarray(solution, dtype=np.float64).reshape(-1)
        if not success or solution.size < 6 or not np.all(np.isfinite(solution[:6])):
            return None

        solution = solution[:6]
        lower, upper = self._i2rt_arm_joint_limits()
        if np.any(solution < lower) or np.any(solution > upper):
            return None
        return solution

    def _generate_randomized_home(self, nominal_home: np.ndarray) -> Optional[np.ndarray]:
        """Generate a valid randomized bimanual home joint target."""
        nominal_home = np.asarray(nominal_home, dtype=np.float64).reshape(-1)
        if nominal_home.size not in (12, 14):
            logger.error(f"Randomized I2RT home expects 12 or 14 values, got {nominal_home.size}")
            return None

        right_start = 7 if nominal_home.size == 14 else 6
        left_nominal_q = nominal_home[:6]
        right_nominal_q = nominal_home[right_start : right_start + 6]
        independent_per_arm = bool(self.random_init_pose_params.get("independent_per_arm", True))
        max_attempts = max(1, int(self.random_init_pose_params.get("max_ik_attempts", 20)))

        for attempt in range(1, max_attempts + 1):
            left_position_delta, left_euler_delta = self._sample_random_init_pose_offset()
            if independent_per_arm:
                right_position_delta, right_euler_delta = self._sample_random_init_pose_offset()
            else:
                right_position_delta = left_position_delta.copy()
                right_euler_delta = left_euler_delta.copy()

            try:
                left_solution = self._solve_randomized_home_arm(
                    left_nominal_q,
                    left_position_delta,
                    left_euler_delta,
                )
                right_solution = self._solve_randomized_home_arm(
                    right_nominal_q,
                    right_position_delta,
                    right_euler_delta,
                )
            except Exception as exc:
                logger.warning(f"I2RT random-init IK attempt {attempt}/{max_attempts} failed: {exc}")
                continue

            if left_solution is None or right_solution is None:
                logger.warning(f"I2RT random-init IK attempt {attempt}/{max_attempts} did not converge")
                continue

            randomized_home = nominal_home.copy()
            randomized_home[:6] = left_solution
            randomized_home[right_start : right_start + 6] = right_solution
            logger.info(
                "I2RT randomized home sampled: "
                f"left_position_delta_m={left_position_delta.tolist()}, "
                f"left_euler_delta_deg={left_euler_delta.tolist()}, "
                f"right_position_delta_m={right_position_delta.tolist()}, "
                f"right_euler_delta_deg={right_euler_delta.tolist()}, "
                f"target_q={randomized_home.tolist()}"
            )
            return randomized_home

        logger.error(f"I2RT random-init IK failed after {max_attempts} attempts")
        return None

    def _random_init_pose_sync_path(self):
        from pathlib import Path

        return Path(self.random_init_pose_params["sync_target_path"])

    def _write_random_init_pose_target(self, home: np.ndarray) -> None:
        """Publish the follower's completed home target for the leader to consume."""
        if not getattr(self, "random_init_pose_sync", False) or not self.is_follower_role:
            return
        path = self._random_init_pose_sync_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device_name": self.device_name,
            "device_id": int(self.device_id),
            "written_at_ns": time.time_ns(),
            "home_joint_pos": np.asarray(home, dtype=np.float64).reshape(-1).tolist(),
        }
        temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary_path, path)
        logger.info(f"Published synchronized I2RT home target for leader: {path}")

    def _invalidate_random_init_pose_target(self) -> None:
        """Invalidate the previous target before starting a new follower home."""
        if not getattr(self, "random_init_pose_sync", False) or not self.is_follower_role:
            return
        path = self._random_init_pose_sync_path()
        try:
            path.unlink()
            logger.info(f"Invalidated previous synchronized I2RT home target: {path}")
        except FileNotFoundError:
            pass

    def _read_synchronized_home_target(self) -> Optional[np.ndarray]:
        """Wait for a new follower target, preventing stale-target consumption."""
        if not getattr(self, "random_init_pose_sync", False) or not self.is_leader_role:
            return None
        path = self._random_init_pose_sync_path()
        timeout_s = float(self.random_init_pose_params.get("sync_wait_timeout_s", 15.0))
        time.sleep(max(0.0, float(self.random_init_pose_params.get("sync_wait_before_read_s", 0.5))))
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() <= deadline:
            try:
                current_mtime_ns = path.stat().st_mtime_ns
                if current_mtime_ns <= 0:
                    time.sleep(0.05)
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                target = np.asarray(payload["home_joint_pos"], dtype=np.float64).reshape(-1)
                if target.size not in (12, 14) or not np.all(np.isfinite(target)):
                    raise ValueError(f"invalid synchronized target shape/value: {target.shape}")
                logger.info(f"Consumed synchronized I2RT home target from follower: {path}")
                return target
            except FileNotFoundError:
                time.sleep(0.05)
            except Exception as exc:
                logger.warning(f"Failed to read synchronized I2RT home target: {exc}")
                time.sleep(0.05)
        logger.error(f"Timed out waiting for a new synchronized follower home target after {timeout_s:.1f}s: {path}")
        return None

    def _go_home(self, reason: str = "episode_reset") -> bool:
        """Move both I2RT arms to configured home joint positions."""
        logger.info(f"======== GOING HOME!!! for {self.device_name}_{self.device_id} =========")
        if self.home_joint_pos is None:
            logger.warning("home_joint_pos is not configured, skipping go-home")
            return True
        if self._left_client is None or self._right_client is None:
            logger.error("I2RT clients are not operational for go-home")
            return False
        try:
            home = np.asarray(self.home_joint_pos, dtype=np.float64).reshape(-1)
            self._invalidate_random_init_pose_target()
            if getattr(self, "random_init_pose_sync", False) and self.is_leader_role:
                synchronized_home = self._read_synchronized_home_target()
                if synchronized_home is None:
                    return False
                if synchronized_home.size == 14:
                    home[:6] = synchronized_home[:6]
                    home[7:13] = synchronized_home[7:13]
                elif synchronized_home.size == 12:
                    home[:6] = synchronized_home[:6]
                    home[7:13] = synchronized_home[6:12]
                else:
                    logger.error(f"Leader synchronized home expects 12 or 14 values, got {synchronized_home.size}")
                    return False
            if self._should_randomize_init_pose(reason):
                randomized_home = self._generate_randomized_home(home)
                if randomized_home is not None:
                    home = randomized_home
                elif not bool(self.random_init_pose_params.get("fallback_to_nominal_home", True)):
                    logger.error("Randomized home failed and nominal-home fallback is disabled")
                    return False
                else:
                    logger.warning("Falling back to nominal I2RT home joint positions")
            if self.is_follower_role:
                if home.size != 14:
                    logger.error(f"Follower go-home expects 14 values, got {home.size}")
                    return False
                # left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_home")
                # right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_home")

                if self._last_trusted_get_info_state_left is not None:
                    logger.info(
                        f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state"
                    )
                    left_current = self._last_trusted_get_info_state_left
                else:
                    logger.warning(
                        f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state"
                    )
                    left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_home")

                if self._last_trusted_get_info_state_right is not None:
                    logger.info(
                        f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state"
                    )
                    right_current = self._last_trusted_get_info_state_right
                else:
                    logger.warning(
                        f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state"
                    )
                    right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_home")

                left_target = home[:7]
                right_target = home[7:14]
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current,
                    right_current=right_current,
                )
                self._write_random_init_pose_target(home)
            else:
                if home.size == 14:
                    left_target = home[:6]
                    right_target = home[7:13]
                elif home.size == 12:
                    left_target = home[:6]
                    right_target = home[6:12]
                else:
                    logger.error(f"Leader go-home expects 12 or 14 values, got {home.size}")
                    return False

                # left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_home")
                # right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_home")

                if self._last_trusted_get_info_state_left is not None:
                    logger.info(
                        f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state"
                    )
                    left_current = self._last_trusted_get_info_state_left
                else:
                    logger.warning(
                        f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state"
                    )
                    left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_home")

                if self._last_trusted_get_info_state_right is not None:
                    logger.info(
                        f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is available, using trusted state"
                    )
                    right_current = self._last_trusted_get_info_state_right
                else:
                    logger.warning(
                        f"role: {'leader' if self.is_leader_role else 'follower'}, trusted state is not available, using current state"
                    )
                    right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_home")

                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current[:6],
                    right_current=right_current[:6],
                )

            logger.info(
                f"role: {'leader' if self.is_leader_role else 'follower'}, left_current: {left_current.tolist()}, right_current: {right_current.tolist()}"
            )
            self._last_button_left = 0.0
            self._last_button_right = 0.0
            self._last_control_flag = 0.0 if self.dual_control_mode else 1.0

            with self._trusted_state_lock:
                self._last_trusted_get_info_state_left = None
                self._last_trusted_get_info_state_right = None
                logger.info(f"reset trusted state for {self.device_name}_{self.device_id}")

            return True
        except Exception as e:
            logger.error(f"Failed to execute I2RT go-home: {e}")
            return False

    def _go_rest(self) -> bool:
        """Move both I2RT arms to configured rest joint positions."""
        logger.info(f"GOING REST!!! for {self.device_name}_{self.device_id}")
        if self._left_client is None or self._right_client is None:
            logger.error("I2RT clients are not operational for go-rest")
            return False
        try:
            rest_joint_pos = np.zeros_like(self.home_joint_pos)
            rest_joint_pos[6] = 0.99
            rest_joint_pos[13] = 0.99
            if self.is_follower_role:
                left_current = self._follower_get_info(self._left_client, "get_info_left_follower_go_rest")
                right_current = self._follower_get_info(self._right_client, "get_info_right_follower_go_rest")
                left_target = rest_joint_pos[:7]
                right_target = rest_joint_pos[7:14]
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current,
                    right_current=right_current,
                )
            else:
                left_target = rest_joint_pos[:6]
                right_target = rest_joint_pos[7:13]
                left_current, _ = self._leader_get_info(self._left_client, "get_info_left_leader_go_rest")
                right_current, _ = self._leader_get_info(self._right_client, "get_info_right_leader_go_rest")
                self._left_client.update_kp_kd(self._leader_default_kp, self._leader_default_kd)
                self._right_client.update_kp_kd(self._leader_default_kp, self._leader_default_kd)
                self._smooth_joint_move(
                    left_target=left_target,
                    right_target=right_target,
                    left_current=left_current[:6],
                    right_current=right_current[:6],
                )
            return True
        except Exception as e:
            logger.error(f"Failed to execute I2RT go-rest: {e}")
            return False

    def _read_control_flag(self) -> float:
        if self.is_leader_role:
            return self._control_mode == "leader_to_follower"

        try:
            device_header_data = bytes(self._control_flag_shm.buf[:DEVICE_HEADER_SIZE])
            device_info = unpack_device_header(device_header_data)
            frame_size = device_info["frame_size"]
            frame_data_size = frame_size - FRAME_HEADER_SIZE

            buffer_header_data = bytes(
                self._control_flag_shm.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]
            )
            buffer_info = unpack_buffer_header(buffer_header_data)
            if buffer_info["current_frames_count"] == 0:
                return 0.0
            _latest_index = get_latest_control_frame_index(buffer_info["write_index"], buffer_info["buffer_size"])
            frame_offset = get_latest_control_frame_offset(
                buffer_info["write_index"], buffer_info["buffer_size"], frame_size
            )
            timestamp_data = self._control_flag_shm.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE]
            timestamp_ns = int.from_bytes(bytes(timestamp_data), byteorder="little")
            if timestamp_ns == 0:
                return 0.0
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + frame_data_size
            numpy_dtype = get_dtype(device_info["data_dtype"])
            value = np.frombuffer(bytes(self._control_flag_shm.buf[data_start:data_end]), dtype=numpy_dtype)[0]
            return float(value)
        except Exception as e:
            logger.debug(f"Failed to read I2RT control flag: {e}")
            return 0.0

    def _find_teleoperator_key(self, control_dict: Dict[str, np.ndarray]) -> Optional[str]:
        for key, value in control_dict.items():
            if value is None:
                continue
            if key.endswith("_control_flag"):
                continue
            if key == self.default_control_shm_name:
                continue
            if key.endswith("_control"):
                continue
            return key
        return None

    def _get_control_array(self, control_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        # Leader should only follow explicitly configured/connected source SHMs
        # (typically follower state SHM), and must never consume its own default
        # control SHM fallback.
        if self.is_leader_role:
            for shm_name in self.control_shm_names:
                value = control_dict.get(shm_name)
                if value is not None and not shm_name.endswith("_control") and not shm_name.endswith("_control_flag"):
                    return value
            return None

        # Follower policy path: prefer their own control SHM,
        # then any available control-like stream.
        if control_dict.get(self.default_control_shm_name) is not None:
            return control_dict[self.default_control_shm_name]
        for key, value in control_dict.items():
            if value is not None and key.endswith("_control") and not key.endswith("_control_flag"):
                return value
        return None

    def _notify_policy_runner_reset(self) -> None:
        """Notify policy runner and policy connector to reset when robot is resetting."""
        try:
            from pathlib import Path

            # Create reset command file for policy runner (ActionExecutor)
            reset_command_file = Path(self._policy_runner_reset_command)
            reset_command_file.parent.mkdir(parents=True, exist_ok=True)
            reset_command_file.touch()

            # Also create reset signal file for policy connector
            # This notifies the policy connector to reset its RTC state
            reset_signal_file = Path(self._reset_signal)
            reset_signal_file.touch()

            logger.info("Notified policy runner and policy connector to reset due to robot reset")
        except Exception as e:
            logger.warning(f"Failed to notify policy runner/connector of reset: {e}")

    def _get_status(self) -> dict:
        return {
            "mode": self.mode,
            "is_leader_role": self.is_leader_role,
            "policy_control_only": self.policy_control_only,
            "takeover_alignment_enabled": self.takeover_alignment_enabled,
            "takeover_alignment_duration_s": self.takeover_alignment_duration_s,
            "takeover_alignment_settle_timeout_s": self.takeover_alignment_settle_timeout_s,
            "control_shm_names": self.control_shm_names,
            "control_flag_shm_name": self._control_flag_shm_name,
            "left_endpoint": f"{self.left_server_host}:{self.left_server_port}",
            "right_endpoint": f"{self.right_server_host}:{self.right_server_port}",
        }

    def get_status(self) -> dict:
        status = super().get_status()
        status.update(self._get_status())
        return status

    def _prepare_main_control_loop_resources(self) -> None:
        """Prepare SHM resources before entering main control loop."""
        if self.running:
            logger.info(f"Robot {self.device_name}_{self.device_id} is already running")
            return

        logger.info(f"Starting {self.device_name}_{self.device_id} server...")
        self.running = True

        # Create state shared memory
        self._create_shared_memory()
        if self.control_shm_names is not None:
            self.control_shared_memory = {shm_name: None for shm_name in self.control_shm_names}
            for shm_name in self.control_shm_names:
                connected = self._connect_to_existing_control_shm(shm_name)
                # Give peer devices a short window to create their SHMs.
                if not connected:
                    max_retries = 20
                    retry_interval_s = 0.2
                    for _ in range(max_retries):
                        time.sleep(retry_interval_s)
                        connected = self._connect_to_existing_control_shm(shm_name)
                        if connected:
                            break
                    if not connected:
                        logger.warning(f"Failed to connect control SHM after retries: {shm_name}")
        if self.dual_control_mode or self.control_shm_names is None:
            self._create_control_shared_memory()

        # Connect to control flag SHM (owned by I2RT leader) if in dual control mode
        if self.dual_control_mode and not getattr(self, "is_leader_role", False):
            self._connect_control_flag_shm()

        if not self.shared_memory or not any(self.control_shared_memory.values()):
            self.running = False
            raise RuntimeError("Failed to create/connect to shared memories")

        logger.info(f"Robot {self.device_name}_{self.device_id} server started successfully")

    def _start_main_control_loop(self) -> None:
        """Start the control loop once; never replace a live control thread."""
        if not self.running:
            return
        with self._main_control_loop_lock:
            thread = self._main_control_loop_thread
            if thread is not None and thread.is_alive():
                raise RuntimeError("I2RT main control loop is already running")
            self._main_control_loop_generation += 1
            generation = self._main_control_loop_generation
            thread = threading.Thread(
                target=self._run_main_control_loop,
                args=(generation,),
                daemon=True,
                name=f"I2RTRobot_{self.device_id}_main_control_loop_g{generation}",
            )
            self._main_control_loop_thread = thread
            thread.start()

    def start_server(self) -> None:
        try:
            self._i2rt_stopped = False
            logger.info(f" === starting start_server for {self.device_name}_{self.device_id} ===")
            logger.info(f"{self.device_name}_{self.device_id} Going to home position...")
            if not self._go_home(reason="startup"):
                logger.error(f"Failed to go to home position for {self.device_name}_{self.device_id}")
                raise RuntimeError(f"Failed to go to home position for {self.device_name}_{self.device_id}")
            self._apply_bilateral_gains_for_mode()
            curr_state = self.get_robot_state()
            if curr_state is not None:
                if self.is_leader_role:
                    self._command_joint_pos(
                        self._left_client, curr_state[:6], "command_joint_pos_left_start_server_leader"
                    )
                    self._command_joint_pos(
                        self._right_client, curr_state[7:13], "command_joint_pos_right_start_server_leader"
                    )
                else:
                    self._command_joint_pos(
                        self._left_client, curr_state[:7], "command_joint_pos_left_start_server_follower"
                    )
                    self._command_joint_pos(
                        self._right_client, curr_state[7:14], "command_joint_pos_right_start_server_follower"
                    )
                logger.info(f"Set joint pos after bilateral gains: {curr_state}")
            else:
                logger.error(f"Failed to get robot state for {self.device_name}_{self.device_id}")
                raise RuntimeError(f"Failed to get robot state for {self.device_name}_{self.device_id}")

            # Keep the filtered state cache available for normal homing.
            self._trusted_state_monitor_evt.set()
            self._trusted_state_thread = threading.Thread(
                target=self._monitor_trusted_state,
                daemon=True,
                name=f"I2RTRobot_{self.device_id}_trusted_state_monitor",
            )
            self._trusted_state_thread.start()

            self._prepare_main_control_loop_resources()
            self._start_main_control_loop()

            while self.running:
                try:
                    time.sleep(0.5)
                except KeyboardInterrupt:
                    break

        except Exception as e:
            logger.error(f"Failed to start {self.device_name}_{self.device_id} server: {e}")
            raise e

        finally:
            # Keep shutdown semantics aligned with BaseRobot.start_server:
            # always run stop_server() when start_server exits (including Ctrl+C
            # or external shutdown that flips self.running to False).
            try:
                # IMPORTANT: We should run super().stop_server() to ensure the shared memory is cleaned up before setting the robot home in self.stop_server()
                super().stop_server()
                self.stop_server()
            except Exception as e:
                logger.error(f"Error while stopping {self.device_name}_{self.device_id} in start_server finally: {e}")
            logger.info(f"=== Exiting start_server for {self.device_name}_{self.device_id} ===")

    def stop_server(self) -> None:
        """Stop robot and terminate locally launched minimum_gello servers.

        Order: _go_rest (needs live RPC servers) -> super cleanup -> terminate servers.
        """
        logger.info(
            f"=== Stopping {self.device_name}_{self.device_id} server... in I2RTRobot self.running={self.running} ==="
        )
        if hasattr(self, "_i2rt_stopped") and self._i2rt_stopped:
            return
        self._trusted_state_monitor_evt.clear()
        # Stop the trusted state sampler
        if self._trusted_state_thread is not None and self._trusted_state_thread.is_alive():
            logger.info("Stopping trusted state monitoring thread...")
            self._trusted_state_thread.join(timeout=5.0)
            if self._trusted_state_thread.is_alive():
                logger.warning("Trusted state monitoring thread did not stop gracefully")
            self._trusted_state_thread = None
        self._i2rt_stopped = True

        # stopping main control loop manually
        with self._main_control_loop_lock:
            self._main_control_loop_generation += 1
            control_loop_thread = self._main_control_loop_thread
            self._main_control_loop_thread = None
        if (
            control_loop_thread is not None
            and control_loop_thread.is_alive()
            and control_loop_thread is not threading.current_thread()
        ):
            logger.info("Stopping main control loop thread...")
            control_loop_thread.join(timeout=1.0)
            if control_loop_thread.is_alive():
                logger.warning("Main control loop thread did not stop gracefully")

        try:
            if not self._go_rest():
                logger.error(f"Failed to go to rest position for {self.device_name}_{self.device_id}")
        except Exception as e:
            logger.error(f"Exception during go-rest for {self.device_name}_{self.device_id}: {e}")
        super().stop_server()
        for proc in self._server_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception as e:
                logger.warning(f"Failed to terminate minimum_gello process: {e}")
        self._server_processes = []
