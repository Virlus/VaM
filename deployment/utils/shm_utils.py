#!/usr/bin/env python3
"""
Shared Memory (SHM) utilities for unified SHM format.
Windows Compatible Version.

Changes:
- Forced int() casting for all size calculations to satisfy Windows CreateFileMapping.

Author: Jun Lv , JunjieXu
"""

import struct
import time
import os
import tempfile
import multiprocessing.shared_memory as shm
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from contextlib import AbstractContextManager
import numpy as np
from utils.logger_config import logger

# Data type mapping for string to numpy dtype conversion
DTYPE_MAP = {
    "uint8": np.uint8,
    "int8": np.int8,
    "uint16": np.uint16,
    "int16": np.int16,
    "uint32": np.uint32,
    "int32": np.int32,
    "float32": np.float32,
    "float64": np.float64,
}


def get_dtype(dtype_str: str, default: np.dtype = np.float32) -> np.dtype:
    """Convert string dtype to numpy dtype."""
    return DTYPE_MAP.get(dtype_str, default)


# Constants
MAX_SHAPE_DIMENSIONS = 4

# SHM Names
DEVICE_SHM_PREFIX = "{device_type}_{device_id}_data"
SUMMARY_SHM_NAME = "device_summary_data"
SUMMARY_SHM_LOCK_DIR = tempfile.gettempdir()

# Header Sizes
DEVICE_HEADER_SIZE = (
    60 + 4 * MAX_SHAPE_DIMENSIONS
)  # device_type(20) + device_id(4) + fps(4) + data_dtype(20) + num_dims(4) + shape(16) + frame_size(4) + hardware_latency_ms(4)
BUFFER_HEADER_SIZE = 16  # buffer_size(4) + current_frames_count(4) + write_index(4) + padding(4)
MANAGER_HEADER_SIZE = (
    24  # device_count(4) + buffer_size(4) + current_frames_count(4) + write_index(4) + update_timestamp(8)
)
POLICY_HEADER_SIZE = 16  # device_count(4) + update_timestamp(8) + padding(4) - for policy SHM (old format)
FRAME_HEADER_SIZE = 8  # timestamp_ns(8) - for individual device data
SUMMARY_FRAME_HEADER_SIZE = 8  # frame_timestamp_ns(8) - for entire frame in summary SHM
ACTION_INDEX_SIZE = 8

# Format strings
DEVICE_HEADER_FORMAT = (
    "<20sIf20sI" + "I" * MAX_SHAPE_DIMENSIONS + "If"
)  # device_type, device_id, fps, data_dtype, num_dims, shape, frame_size, hardware_latency_ms
BUFFER_HEADER_FORMAT = "<IIII"  # buffer_size, current_frames_count, write_index, padding
MANAGER_HEADER_FORMAT = "<IIIIQ"  # device_count, buffer_size, current_frames_count, write_index, update_timestamp
POLICY_HEADER_FORMAT = "<IQI"  # device_count, update_timestamp, padding - for policy SHM
FRAME_HEADER_FORMAT = "<Q"  # timestamp_ns
SUMMARY_FRAME_HEADER_FORMAT = "<Q"  # frame_timestamp_ns
ACTION_INDEX_FORMAT = "<Q"  # int64 action_index


def pad_shape(shape, max_dims: int = MAX_SHAPE_DIMENSIONS) -> Tuple[int, ...]:
    """Pad shape to max_dims with zeros."""
    if isinstance(shape, list):
        shape = tuple(shape)

    if len(shape) > max_dims:
        raise ValueError(f"Shape {shape} has {len(shape)} dimensions, which exceeds the maximum of {max_dims}.")
    elif len(shape) < max_dims:
        return shape + (0,) * (max_dims - len(shape))
    return shape


def truncate_shape(shape: Tuple[int, ...], max_dims: int = MAX_SHAPE_DIMENSIONS) -> Tuple[int, ...]:
    """Truncate shape to max_dims."""
    return shape[:max_dims]


def validate_shape(shape, max_dims: int = MAX_SHAPE_DIMENSIONS) -> bool:
    """Validate shape dimensions."""
    if isinstance(shape, list):
        shape = tuple(shape)
    return len(shape) <= max_dims and all(dim >= 0 for dim in shape)


def pack_device_header(
    device_type: str,
    device_id: int,
    fps: float,
    data_dtype: str,
    shape: Tuple[int, ...],
    frame_size: int,
    hardware_latency_ms: float = 0.0,
) -> bytes:
    """Pack device header."""
    if not validate_shape(shape):
        raise ValueError(f"Invalid shape: {shape}")

    padded_shape = pad_shape(shape)
    device_type_bytes = device_type.encode("utf-8")[:20].ljust(20, b"\x00")
    data_dtype_bytes = data_dtype.encode("utf-8")[:20].ljust(20, b"\x00")
    num_dims = len(shape)

    return struct.pack(
        DEVICE_HEADER_FORMAT,
        device_type_bytes,
        int(device_id),
        float(fps),
        data_dtype_bytes,
        int(num_dims),
        *[int(x) for x in padded_shape],
        int(frame_size),
        float(hardware_latency_ms),
    )


def unpack_device_header(data: bytes) -> Dict[str, Any]:
    """Unpack device header."""
    if len(data) < DEVICE_HEADER_SIZE:
        raise ValueError(f"Data too short for device header: {len(data)} < {DEVICE_HEADER_SIZE}")

    values = struct.unpack(DEVICE_HEADER_FORMAT, data[:DEVICE_HEADER_SIZE])

    device_type = values[0].decode("utf-8").rstrip("\x00")
    device_id = values[1]
    fps = values[2]
    data_dtype = values[3].decode("utf-8").rstrip("\x00")
    num_dims = values[4]
    shape = tuple(values[5 : 5 + MAX_SHAPE_DIMENSIONS])
    frame_size = values[5 + MAX_SHAPE_DIMENSIONS]
    hardware_latency_ms = values[6 + MAX_SHAPE_DIMENSIONS]

    actual_shape = shape[:num_dims] if num_dims > 0 else (1,)

    return {
        "device_type": device_type,
        "device_id": device_id,
        "fps": fps,
        "data_dtype": data_dtype,
        "num_dims": num_dims,
        "shape": actual_shape,
        "frame_size": frame_size,
        "hardware_latency_ms": hardware_latency_ms,
    }


def pack_buffer_header(buffer_size: int, current_frames_count: int, write_index: int) -> bytes:
    """Pack buffer header."""
    return struct.pack(BUFFER_HEADER_FORMAT, int(buffer_size), int(current_frames_count), int(write_index), 0)


def unpack_buffer_header(data: bytes) -> Dict[str, int]:
    """Unpack buffer header."""
    if len(data) < BUFFER_HEADER_SIZE:
        raise ValueError(f"Data too short for buffer header: {len(data)} < {BUFFER_HEADER_SIZE}")
    values = struct.unpack(BUFFER_HEADER_FORMAT, data[:BUFFER_HEADER_SIZE])
    return {"buffer_size": values[0], "current_frames_count": values[1], "write_index": values[2], "padding": values[3]}


def pack_manager_header(
    device_count: int,
    buffer_size: int = 0,
    current_frames_count: int = 0,
    write_index: int = 0,
    timestamp: Optional[int] = None,
) -> bytes:
    """Pack manager header."""
    if timestamp is None:
        timestamp = time.time_ns()
    return struct.pack(
        MANAGER_HEADER_FORMAT,
        int(device_count),
        int(buffer_size),
        int(current_frames_count),
        int(write_index),
        int(timestamp),
    )


def pack_policy_header(device_count: int, timestamp: Optional[int] = None) -> bytes:
    """Pack policy header."""
    if timestamp is None:
        timestamp = time.time_ns()
    return struct.pack(POLICY_HEADER_FORMAT, int(device_count), int(timestamp), 0)


def unpack_manager_header(data: bytes) -> Dict[str, Any]:
    """Unpack manager header."""
    if len(data) < MANAGER_HEADER_SIZE:
        raise ValueError(f"Data too short for manager header: {len(data)} < {MANAGER_HEADER_SIZE}")
    values = struct.unpack(MANAGER_HEADER_FORMAT, data[:MANAGER_HEADER_SIZE])
    return {
        "device_count": values[0],
        "buffer_size": values[1],
        "current_frames_count": values[2],
        "write_index": values[3],
        "update_timestamp": values[4],
    }


def unpack_policy_header(data: bytes) -> Dict[str, Any]:
    """Unpack policy header."""
    if len(data) < POLICY_HEADER_SIZE:
        raise ValueError(f"Data too short for policy header: {len(data)} < {POLICY_HEADER_SIZE}")
    values = struct.unpack(POLICY_HEADER_FORMAT, data[:POLICY_HEADER_SIZE])
    return {"device_count": values[0], "update_timestamp": values[1], "padding": values[2]}


def pack_frame_header(timestamp_ns: int) -> bytes:
    """Pack frame header."""
    return struct.pack(FRAME_HEADER_FORMAT, int(timestamp_ns))


def unpack_frame_header(data: bytes) -> Dict[str, int]:
    """Unpack frame header."""
    if len(data) < FRAME_HEADER_SIZE:
        raise ValueError(f"Data too short for frame header: {len(data)} < {FRAME_HEADER_SIZE}")
    timestamp_ns = struct.unpack(FRAME_HEADER_FORMAT, data[:FRAME_HEADER_SIZE])[0]
    return {"timestamp_ns": timestamp_ns}


def pack_summary_frame_header(frame_timestamp_ns: int) -> bytes:
    """Pack summary frame header."""
    return struct.pack(SUMMARY_FRAME_HEADER_FORMAT, int(frame_timestamp_ns))


def unpack_summary_frame_header(data: bytes) -> Dict[str, int]:
    """Unpack summary frame header."""
    if len(data) < SUMMARY_FRAME_HEADER_SIZE:
        raise ValueError(f"Data too short for summary frame header: {len(data)} < {SUMMARY_FRAME_HEADER_SIZE}")
    frame_timestamp_ns = struct.unpack(SUMMARY_FRAME_HEADER_FORMAT, data[:SUMMARY_FRAME_HEADER_SIZE])[0]
    return {"frame_timestamp_ns": frame_timestamp_ns}


# -----------------------------------------------------------------------------
# SIZE CALCULATIONS (CRITICAL FOR WINDOWS: FORCE INT RETURN TYPE)
# -----------------------------------------------------------------------------


def calculate_device_shm_size(buffer_size: int, frame_data_size: int) -> int:
    """Calculate total device SHM size. Returns standard int."""
    # ### WINDOWS FIX: Ensure result is python int, not numpy int ###
    return int(DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + buffer_size * (FRAME_HEADER_SIZE + frame_data_size))


def calculate_summary_shm_size(device_count: int, frame_sizes: List[int], buffer_size: int) -> int:
    """Calculate total summary SHM size. Returns standard int."""
    frame_data_size = sum(frame_sizes)
    total_frame_size = SUMMARY_FRAME_HEADER_SIZE + frame_data_size
    # ### WINDOWS FIX: Ensure result is python int ###
    return int(MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + buffer_size * total_frame_size)


def calculate_policy_shm_size(device_count: int, frame_sizes: List[int]) -> int:
    """Calculate total policy SHM size. Returns standard int."""
    # ### WINDOWS FIX: Ensure result is python int ###
    return int(MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + sum(frame_sizes) + ACTION_INDEX_SIZE)


# -----------------------------------------------------------------------------
# OFFSET CALCULATIONS (CRITICAL FOR WINDOWS: FORCE INT RETURN TYPE)
# -----------------------------------------------------------------------------


def get_device_info_offset(device_index: int) -> int:
    return int(MANAGER_HEADER_SIZE + device_index * DEVICE_HEADER_SIZE)


def get_data_offset(device_index: int, device_count: int, frame_sizes: List[int]) -> int:
    return int(MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + sum(frame_sizes[:device_index]))


def get_policy_data_offset(device_index: int, device_count: int, frame_sizes: List[int]) -> int:
    return int(MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + sum(frame_sizes[:device_index]))


def get_policy_action_index_offset(device_count: int, frame_sizes: List[int]) -> int:
    """Get offset for action_index appended at the end of policy SHM."""
    return int(MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE + sum(frame_sizes))


def get_control_buffer_start_offset() -> int:
    """Get start offset of control frame ring-buffer area."""
    return int(DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE)


def get_control_frame_offset(frame_index: int, frame_size: int) -> int:
    """
    Get control frame start offset.

    Args:
        frame_index: Index in the control circular buffer.
        frame_size: Per-frame bytes including FRAME_HEADER_SIZE.
    """
    return int(get_control_buffer_start_offset() + int(frame_index) * int(frame_size))


def get_control_frame_offsets(buffer_size: int, frame_size: int) -> List[int]:
    """Get all control frame start offsets for a circular buffer."""
    return [get_control_frame_offset(i, frame_size) for i in range(int(buffer_size))]


def get_latest_control_frame_index(write_index: int, buffer_size: int) -> int:
    """Get latest control frame index from write_index in a circular buffer."""
    return int((int(write_index) - 1) % int(buffer_size))


def get_latest_control_frame_offset(write_index: int, buffer_size: int, frame_size: int) -> int:
    """Get latest control frame start offset from circular-buffer metadata."""
    latest_index = get_latest_control_frame_index(write_index, buffer_size)
    return get_control_frame_offset(latest_index, frame_size)


def get_frame_offset(device_index: int, buffer_size: int, frame_size: int, frame_index: int) -> int:
    return int(DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_index * (FRAME_HEADER_SIZE + frame_size))


def get_summary_frame_offset(
    device_index: int, frame_index: int, device_count: int, frame_sizes: List[int], buffer_size: int
) -> int:
    """Get frame offset for device in summary SHM with buffer support."""
    # Buffer layout: Frame0[FrameHeader+Device0+Device1+...] + Frame1[FrameHeader+Device0+Device1+...] + ...
    # Base offset: manager header + all device headers
    base_offset = MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE
    total_frame_size = SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes)
    frame_start_offset = base_offset + frame_index * total_frame_size
    device_offset_in_frame = SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes[:device_index])
    return int(frame_start_offset + device_offset_in_frame)


def get_summary_frame_start_offset(frame_index: int, device_count: int, frame_sizes: List[int]) -> int:
    base_offset = MANAGER_HEADER_SIZE + device_count * DEVICE_HEADER_SIZE
    total_frame_size = SUMMARY_FRAME_HEADER_SIZE + sum(frame_sizes)
    return int(base_offset + frame_index * total_frame_size)


def _sanitize_lock_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)


class SummarySHMLock(AbstractContextManager):
    """
    Cross-process exclusive lock for summary SHM access.

    This intentionally uses an exclusive lock for both readers and writers. It is
    not a shared-reader RW lock because history collection needs a stable summary
    buffer across several timestamp lookups.
    """

    _thread_locks: Dict[str, Any] = {}
    _thread_state = None

    def __init__(self, summary_shm_name: str = SUMMARY_SHM_NAME) -> None:
        self.summary_shm_name = summary_shm_name
        lock_filename = f"vam_{_sanitize_lock_name(summary_shm_name)}.lock"
        self.lock_path = Path(SUMMARY_SHM_LOCK_DIR) / lock_filename
        self._file = None
        self._thread_lock = self._get_thread_lock(str(self.lock_path))

    @classmethod
    def _get_thread_lock(cls, lock_key: str) -> Any:
        import threading

        if cls._thread_state is None:
            cls._thread_state = threading.local()

        lock = cls._thread_locks.get(lock_key)
        if lock is None:
            lock = threading.RLock()
            cls._thread_locks[lock_key] = lock
        return lock

    def _get_state(self) -> Dict[str, Any]:
        states = getattr(self._thread_state, "states", None)
        if states is None:
            states = {}
            self._thread_state.states = states
        return states.setdefault(str(self.lock_path), {"depth": 0, "file": None})

    def __enter__(self) -> "SummarySHMLock":
        self._thread_lock.acquire()
        try:
            state = self._get_state()
            if state["depth"] > 0:
                state["depth"] += 1
                return self

            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            state["file"] = self.lock_path.open("a+b")
            self._file = state["file"]
            self._lock_file(state["file"])
            state["depth"] = 1
            return self
        except Exception:
            state = self._get_state()
            if state.get("file") is not None:
                try:
                    state["file"].close()
                except Exception:
                    pass
            state["file"] = None
            state["depth"] = 0
            self._thread_lock.release()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            state = self._get_state()
            if state["depth"] > 1:
                state["depth"] -= 1
                return

            file_obj = state["file"]
            if file_obj is not None:
                self._unlock_file(file_obj)
                file_obj.close()
            state["depth"] = 0
            state["file"] = None
        finally:
            self._file = None
            self._thread_lock.release()

    def _lock_file(self, file_obj: Any) -> None:
        if os.name == "nt":
            import msvcrt

            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_LOCK, 1)
            return

        import fcntl

        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)

    def _unlock_file(self, file_obj: Any) -> None:
        if os.name == "nt":
            import msvcrt

            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
            return

        import fcntl

        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


class DataChannel:
    """Data channel for SharedMemory-based inter-process communication."""

    def __init__(self, shm_obj: shm.SharedMemory, read_only: bool = False):
        self._shm = shm_obj
        self._buffer = shm_obj.buf
        self._name = shm_obj.name
        self._read_only = read_only

    def __getitem__(self, key: Union[slice, int]):
        result = self._buffer[key]
        # For slices, return the raw buffer to maintain np.frombuffer compatibility
        # For single items, we can safely return the value
        if isinstance(key, slice):
            return result
        else:
            return result

    def __setitem__(self, key: Union[slice, int], value: Any):
        self._check_write_permission()
        self._buffer[key] = value

    def _check_write_permission(self):
        """Check if write operations are allowed."""
        if self._read_only:
            logger.error(f"Attempted write to read-only data channel: {self._name}")
            raise PermissionError(f"Data channel '{self._name}' is read-only")

    def __len__(self):
        return len(self._buffer)

    def __bytes__(self):
        return bytes(self._buffer)

    def __buffer__(self, flags):
        # Support Python 3.12+ buffer protocol
        if hasattr(self._buffer, "__buffer__"):
            return self._buffer.__buffer__(flags)
        else:
            # Fallback: create memoryview and return its buffer
            return memoryview(self._buffer).__buffer__(flags)

    def __getbuffer__(self, view, flags):
        # Support older buffer protocol
        if hasattr(self._buffer, "__getbuffer__"):
            return self._buffer.__getbuffer__(view, flags)
        else:
            # Fallback for objects that don't implement __getbuffer__
            mv = memoryview(self._buffer)
            return mv.__getbuffer__(view, flags)

    def __releasebuffer__(self, view):
        if hasattr(self._buffer, "__releasebuffer__"):
            return self._buffer.__releasebuffer__(view)

    # Add memoryview support for np.frombuffer compatibility
    def __array_interface__(self):
        """NumPy array interface support."""
        if hasattr(self._buffer, "__array_interface__"):
            return self._buffer.__array_interface__
        return None

    # Support memoryview() constructor
    def __memoryview__(self):
        return memoryview(self._buffer)

    # SharedMemory compatibility properties and methods
    @property
    def buf(self):
        """Return the buffer for SharedMemory compatibility."""
        if self._read_only:
            # Return readonly memoryview to prevent direct buffer modification
            return self._buffer.toreadonly()
        else:
            return self._buffer

    @property
    def name(self) -> str:
        """Get the name of the shared memory."""
        return self._name

    @property
    def size(self) -> int:
        """Get the size of the shared memory."""
        return self._shm.size

    @property
    def is_read_only(self) -> bool:
        """Check if this channel is read-only."""
        return self._read_only

    def close(self):
        """Close the underlying SharedMemory."""
        self._shm.close()

    def unlink(self):
        """Unlink the underlying SharedMemory if not read-only."""
        if self._read_only:
            logger.error(f"Attempted unlink on read-only data channel: {self._name}")
            raise PermissionError(f"Cannot unlink read-only data channel: {self._name}")
        else:
            self._shm.unlink()


# Backward compatibility aliases
ReadOnlyBuffer = DataChannel
ReadOnlySharedMemory = DataChannel
SharedMemoryChannel = DataChannel


def create_shared_memory(
    name: str, size: int, readonly_after_creation: bool = False
) -> Union[shm.SharedMemory, DataChannel]:
    """
    Create shared memory with unified interface.
    """
    # ### WINDOWS FIX: Ensure size is a standard python int ###
    size = int(size)

    # Clean up any existing shared memory with the same name
    try:
        existing_shm = shm.SharedMemory(name=name)
        existing_shm.close()
        existing_shm.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Create new shared memory
    # ### WINDOWS FIX: Explicitly passing int(size) ###
    shm_obj = shm.SharedMemory(create=True, size=size, name=name)

    if readonly_after_creation:
        logger.debug(f"Created SHM '{name}' with read-only wrapper")
        return DataChannel(shm_obj, read_only=True)
    else:
        return DataChannel(shm_obj, read_only=False)


def connect_to_shared_memory(name: str, read_only: bool = False) -> DataChannel:
    """
    Connect to existing shared memory with optional read-only protection.

    Args:
        name: SHM name
        read_only: If True, return read-only wrapper

    Returns:
        SharedMemory object or ReadOnlyBuffer wrapper

    Raises:
        FileNotFoundError: If SHM doesn't exist
    """
    try:
        # Prevent resource tracker from auto-cleaning (Required for Windows/Linux interop)
        import multiprocessing.resource_tracker as tracker

        original_register = tracker.register
        tracker.register = lambda *args, **kwargs: None

        try:
            shm_obj = shm.SharedMemory(name=name)

            logger.debug(f"Connected to SHM '{name}' in {'read-only' if read_only else 'read-write'} mode")
            return DataChannel(shm_obj, read_only=read_only)

        finally:
            tracker.register = original_register

    except FileNotFoundError:
        logger.error(f"SHM not found: {name}")
        raise
    except Exception as e:
        logger.error(f"Error connecting to SHM '{name}': {e}")
        raise


# Convenience functions
def create_device_shm(
    device_name: str, device_id: int, buffer_size: int, frame_data_size: int, readonly_after_creation: bool = False
) -> DataChannel:
    """Create device SHM with standard naming."""
    shm_name = f"{device_name}_{device_id}_data"
    # calculation function now returns pure int
    shm_size = calculate_device_shm_size(buffer_size, frame_data_size)
    return create_shared_memory(shm_name, shm_size, readonly_after_creation)


def connect_to_device_shm(device_name: str, device_id: int, read_only: bool = True) -> DataChannel:
    """Connect to device SHM with read-only default."""
    shm_name = f"{device_name}_{device_id}_data"
    return connect_to_shared_memory(shm_name, read_only=read_only)


def create_control_shm(
    device_name: str, device_id: int, buffer_size: int, frame_data_size: int, readonly_after_creation: bool = False
) -> DataChannel:
    """Create control SHM with standard naming."""
    shm_name = f"{device_name}_{device_id}_control"
    shm_size = calculate_device_shm_size(buffer_size, frame_data_size)
    return create_shared_memory(shm_name, shm_size, readonly_after_creation)


def connect_to_control_shm(device_name: str, device_id: int, read_only: bool = True) -> DataChannel:
    """Connect to control SHM with read-only default for consumers."""
    shm_name = f"{device_name}_{device_id}_control"
    return connect_to_shared_memory(shm_name, read_only=read_only)


def create_summary_shm(
    device_count: int,
    frame_sizes: List[int],
    buffer_size: int,
    readonly_after_creation: bool = False,
    name: str = SUMMARY_SHM_NAME,
) -> DataChannel:
    """Create summary SHM with standard naming."""
    shm_size = calculate_summary_shm_size(device_count, frame_sizes, buffer_size)
    return create_shared_memory(name, shm_size, readonly_after_creation)


def connect_to_summary_shm(read_only: bool = True, name: str = SUMMARY_SHM_NAME) -> DataChannel:
    """Connect to summary SHM with read-only default for consumers."""
    return connect_to_shared_memory(name, read_only=read_only)


def create_policy_shm(device_count: int, frame_sizes: List[int], readonly_after_creation: bool = False) -> DataChannel:
    """Create policy SHM with standard naming."""
    shm_name = "policy_data"
    shm_size = calculate_policy_shm_size(device_count, frame_sizes)
    return create_shared_memory(shm_name, shm_size, readonly_after_creation)


def connect_to_policy_shm(read_only: bool = False) -> DataChannel:
    """Connect to policy SHM with read-write default for policy runners."""
    return connect_to_shared_memory("policy_data", read_only=read_only)


def make_readonly(shm_obj: shm.SharedMemory) -> DataChannel:
    """Convert existing SharedMemory object to read-only wrapper."""
    return DataChannel(shm_obj, read_only=True)


class PolicyOutputSHM:
    """
    Generic shared memory handler for policy outputs.

    Supports creating, writing, and cleaning up SHM for:
    - Value function (V-function) outputs
    - Q-function outputs
    - Action chunks
    - Any other policy-generated data

    This class eliminates code duplication by providing a unified interface
    for all policy output SHM operations.

    Example:
        config = {
            'name': 'policy_value_estimation',
            'device_type': 'PolicyValue',
            'buffer_size': 100,
            'dtype': 'float64',
            'enabled': True
        }
        value_shm = PolicyOutputSHM(config=config, fps=50.0)
        value_shm.create(shape=(1,))
        value_shm.write(data=np.array([0.5]), timestamp_ns=1234567890)
        value_shm.cleanup()
    """

    def __init__(self, config: Dict[str, Any], fps: float, device_id: int = 0) -> None:
        """
        Initialize PolicyOutputSHM from configuration.

        Args:
            config: Configuration dictionary containing:
                - name: SHM name (e.g., 'policy_value_estimation')
                - device_type: Device type string (e.g., 'PolicyValue', 'ActionChunk')
                - buffer_size: Number of frames to buffer (default: 100)
                - dtype: Data type string (default: 'float64')
                - enabled: Whether this SHM is enabled (default: True)
            fps: Frames per second for this SHM
            device_id: Device ID for the header (default: 0)
        """
        self.name: str = config.get("name", "policy_output")
        self.device_type: str = config.get("device_type", "PolicyOutput")
        self.buffer_size: int = config.get("buffer_size", 100)
        self.dtype_str: str = config.get("dtype", "float64")
        self.enabled: bool = config.get("enabled", True)
        self.fps: float = fps
        self.device_id: int = device_id

        # Runtime state
        self._shm: Optional[DataChannel] = None
        self._created: bool = False
        self._shape: Optional[Tuple[int, ...]] = None
        self._frame_size: int = 0
        self._frame_data_size: int = 0
        self._write_index: int = 0
        self._current_frames_count: int = 0

        # Get numpy dtype
        self._numpy_dtype = get_dtype(self.dtype_str, np.float64)

    @property
    def shm(self) -> Optional[DataChannel]:
        """Get the underlying SHM object."""
        return self._shm

    @property
    def created(self) -> bool:
        """Check if SHM has been created."""
        return self._created

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get the current data shape."""
        return self._shape

    @property
    def frame_size(self) -> int:
        """Get the frame size (header + data) in bytes."""
        return self._frame_size

    @property
    def frame_data_size(self) -> int:
        """Get the frame data size (excluding header) in bytes."""
        return self._frame_data_size

    def create(self, shape: Tuple[int, ...]) -> None:
        """
        Create the shared memory with specified shape.

        Args:
            shape: Shape of the data array (e.g., (1,) for scalar, (100, 1) for Q-values)
        """
        if not self.enabled:
            logger.debug(f"PolicyOutputSHM '{self.name}' is disabled, skipping creation")
            return

        try:
            # Clean up existing SHM if it exists with different shape
            if self._shm is not None:
                try:
                    self._shm.close()
                    self._shm.unlink()
                except Exception:
                    pass
                self._shm = None
                self._created = False

            # Calculate sizes
            self._frame_data_size = int(np.prod(shape) * np.dtype(self._numpy_dtype).itemsize)
            self._frame_size = FRAME_HEADER_SIZE + self._frame_data_size
            total_shm_size = calculate_device_shm_size(self.buffer_size, self._frame_data_size)

            # Create SHM
            self._shm = create_shared_memory(name=self.name, size=total_shm_size, readonly_after_creation=False)

            # Pack and write device header
            device_header = pack_device_header(
                device_type=self.device_type,
                device_id=self.device_id,
                fps=self.fps,
                data_dtype=self.dtype_str,
                shape=shape,
                frame_size=self._frame_size,
                hardware_latency_ms=0.0,
            )

            # Pack and write buffer header
            buffer_header = pack_buffer_header(buffer_size=self.buffer_size, current_frames_count=0, write_index=0)

            # Write headers to SHM
            self._shm.buf[:DEVICE_HEADER_SIZE] = device_header
            self._shm.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header

            # Update state
            self._shape = shape
            self._write_index = 0
            self._current_frames_count = 0
            self._created = True

            logger.info(
                f"PolicyOutputSHM created: name={self.name}, "
                f"device_type={self.device_type}, shape={shape}, "
                f"size={total_shm_size} bytes"
            )

        except Exception as e:
            logger.error(f"Failed to create PolicyOutputSHM '{self.name}': {e}")
            self._shm = None
            self._created = False

    def write(self, data: np.ndarray, timestamp_ns: int) -> bool:
        """
        Write data frame to SHM.

        Args:
            data: Data array to write
            timestamp_ns: Timestamp in nanoseconds

        Returns:
            True if write succeeded, False otherwise
        """
        if not self._created or self._shm is None:
            logger.warning(f"PolicyOutputSHM '{self.name}' not created, cannot write")
            return False

        try:
            # Convert data to correct dtype
            data_array = np.asarray(data, dtype=self._numpy_dtype)

            # Check if shape changed
            if self._shape is not None and data_array.shape != self._shape:
                logger.info(
                    f"PolicyOutputSHM '{self.name}' shape changed from {self._shape} to "
                    f"{data_array.shape}, recreating SHM"
                )
                self.create(data_array.shape)
                if not self._created:
                    return False

            buf = self._shm.buf

            # Calculate frame offset
            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + self._write_index * self._frame_size

            # Write frame header (timestamp)
            frame_header = pack_frame_header(timestamp_ns)
            buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = frame_header

            # Write data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_bytes = data_array.tobytes()
            buf[data_start : data_start + len(data_bytes)] = data_bytes

            # Update write index and frame count
            self._write_index = (self._write_index + 1) % self.buffer_size
            self._current_frames_count = min(self._current_frames_count + 1, self.buffer_size)

            # Update buffer header
            buffer_header = pack_buffer_header(
                buffer_size=self.buffer_size,
                current_frames_count=self._current_frames_count,
                write_index=self._write_index,
            )
            buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header

            logger.debug(f"Wrote to PolicyOutputSHM '{self.name}' at index {self._write_index - 1}")
            return True

        except Exception as e:
            logger.error(f"Failed to write to PolicyOutputSHM '{self.name}': {e}")
            return False

    def write_multi(self, data_arrays: List[np.ndarray], timestamps: np.ndarray) -> bool:
        """
        Write multiple related data arrays (e.g., Q-values + timestamps) to paired SHMs.

        This is a convenience method that writes multiple arrays with the same timestamp
        to maintain synchronization. The first data array uses the first timestamp.

        Args:
            data_arrays: List of data arrays to write (all must have same length)
            timestamps: Array of timestamps (one per action in chunk)

        Returns:
            True if all writes succeeded, False otherwise
        """
        if len(data_arrays) == 0:
            return True

        if len(timestamps) == 0:
            logger.warning(f"No timestamps provided for PolicyOutputSHM '{self.name}'")
            return False

        # Use first timestamp as the frame timestamp
        frame_timestamp = int(timestamps[0])

        # Write each array
        success = True
        for data_array in data_arrays:
            if not self.write(data_array, frame_timestamp):
                success = False

        return success

    def get_config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary for this SHM.

        Useful for passing to rollout savers that need to connect to this SHM.

        Returns:
            Dictionary with SHM configuration
        """
        return {
            "shm": self.name,
            "shape": list(self._shape) if self._shape is not None else [],
            "dtype": self.dtype_str,
        }

    def cleanup(self) -> None:
        """Clean up the shared memory."""
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
                logger.debug(f"Cleaned up PolicyOutputSHM '{self.name}'")
            except Exception as e:
                logger.debug(f"Error cleaning up PolicyOutputSHM '{self.name}': {e}")
            finally:
                self._shm = None
                self._created = False
                self._shape = None
                self._write_index = 0
                self._current_frames_count = 0

    def close(self) -> None:
        """Close the shared memory connection without unlinking."""
        if self._shm is not None:
            try:
                self._shm.close()
                logger.debug(f"Closed PolicyOutputSHM '{self.name}'")
            except Exception as e:
                logger.debug(f"Error closing PolicyOutputSHM '{self.name}': {e}")
            finally:
                self._shm = None
                self._created = False
