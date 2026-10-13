#!/usr/bin/env python3
"""
Base Device Manager - Aggregates latest frames from all devices into a summary SHM.

This module connects to device SHMs, reads the latest frame from each device,
and writes them to a summary SHM for clients to access.
This is the base implementation that can be extended for different types of managers.

Author: Jun Lv
"""

import time
import signal
import numpy as np
from typing import Dict, List, Optional, Any

from utils.config_parser import ensure_config_dict, load_config, parse_device_configs
from utils.shm_utils import (
    pack_manager_header,
    pack_device_header,
    pack_frame_header,
    pack_summary_frame_header,
    unpack_device_header,
    unpack_buffer_header,
    unpack_frame_header,
    calculate_summary_shm_size,
    get_device_info_offset,
    get_summary_frame_offset,
    get_summary_frame_start_offset,
    MANAGER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    SUMMARY_FRAME_HEADER_SIZE,
    get_dtype,
    create_summary_shm,
    connect_to_device_shm,
    SummarySHMLock,
)
from utils.logger_config import logger


class BaseDeviceManager:
    """
    Base Device Manager for aggregating device data into summary SHM with buffer support.

    Summary SHM Structure:
    =====================

    [Manager Header - 24 bytes]
    ├── device_count (4 bytes)      - Number of devices
    ├── buffer_size (4 bytes)       - Buffer size in frames
    ├── current_frames_count (4 bytes)     - Current number of frames stored
    ├── write_index (4 bytes)       - Next write position (0 to buffer_size-1)
    └── update_timestamp (8 bytes)  - Last update timestamp in nanoseconds

    [Device Headers - N × 76 bytes]
    ├── Device 0 Header (76 bytes)  - Metadata for device 0
    ├── Device 1 Header (76 bytes)  - Metadata for device 1
    └── ... Device N Header

    [Buffer Data - buffer_size × (8 + sum(frame_sizes)) bytes]
    ├── Frame 0:
    │   ├── Frame Header: [frame_timestamp_ns (8B)]
    │   ├── Device 0: [timestamp_ns (8B) + data]
    │   ├── Device 1: [timestamp_ns (8B) + data]
    │   └── ... Device N: [timestamp_ns (8B) + data]
    ├── Frame 1:
    │   ├── Frame Header: [frame_timestamp_ns (8B)]
    │   ├── Device 0: [timestamp_ns (8B) + data]
    │   ├── Device 1: [timestamp_ns (8B) + data]
    │   └── ... Device N: [timestamp_ns (8B) + data]
    └── ...
    └── Frame (buffer_size-1): [Frame Header + All devices data]

    Buffer Management:
    ==================
    - Circular buffer: write_index wraps around when reaching buffer_size
    - Latest frame: (write_index - 1) % buffer_size
    - Oldest frame: (write_index - current_frames_count) % buffer_size (when buffer is full)
    - Frame validation: frame_index < current_frames_count

    Memory Layout Example (2 devices, buffer_size=3):
    =================================================
    Offset 0-23:    Manager Header
    Offset 24-99:   Device 0 Header (frame_size=100 bytes including 8B timestamp)
    Offset 100-175: Device 1 Header (frame_size=50 bytes including 8B timestamp)
    Offset 176-183: Frame 0 Header [frame_timestamp_ns (8B)]
    Offset 184-333: Frame 0 Data [Device0: 8B+92B] + [Device1: 8B+42B] = 150 bytes
    Offset 334-341: Frame 1 Header [frame_timestamp_ns (8B)]
    Offset 342-491: Frame 1 Data [Device0: 8B+92B] + [Device1: 8B+42B] = 150 bytes
    Offset 492-499: Frame 2 Header [frame_timestamp_ns (8B)]
    Offset 500-649: Frame 2 Data [Device0: 8B+92B] + [Device1: 8B+42B] = 150 bytes

    Total SHM size = 24 + 2×76 + 3×(8+150) = 650 bytes
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        config_path: Optional[str] = None,
        buffer_size: int = 100,
    ) -> None:
        self.config_path = config_path or "config.yaml"
        self.running = False
        self.devices = []
        self.summary_shm = None
        self.summary_shm_name = "device_summary_data"

        # Buffer management for summary SHM
        self.buffer_size = buffer_size
        self.current_frames_count = 0
        self.write_index = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Load configuration
        if config is None:
            self.config = load_config(config_path=self.config_path)
        else:
            self.config = ensure_config_dict(config)

        self.summary_shm_name = self.config.get("summary_shm", self.summary_shm_name)
        self.summary_shm_lock = SummarySHMLock(self.summary_shm_name)

        self._load_devices()

        # Create summary SHM
        self._create_summary_shm()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down")
        # A signal can interrupt a NumPy view or SHM write. Let the active
        # iteration unwind before start()'s finally block releases its buffers.
        self.running = False

    def _load_devices(self) -> None:
        """Load device configurations."""
        self.devices = parse_device_configs(self.config)

        # Add SHM connection field to each device
        for device in self.devices:
            device["shm"] = None
            device["last_timestamp"] = None
            device["data_size"] = 0
            device["frame_size"] = 0  # Store actual frame size from device SHM
            device["buffer_info"] = None  # Cache for buffer information
            device["data_shape"] = None
            device["data_dtype"] = None
            if self._connect_to_device_with_retry(device):
                # Cache buffer info for future use
                buffer_info = self._get_device_buffer_info(device)
                if buffer_info:
                    buffer_size, current_frames_count, write_index, frame_size, shape = buffer_info
                    device["frame_size"] = frame_size
                    device["buffer_info"] = buffer_info
                    device["data_shape"] = shape
                    logger.info(
                        f"Device {device['device_id']}: frame_size = {frame_size:,} bytes, buffer_size = {buffer_size}"
                    )
                else:
                    # Cannot read buffer info from device SHM
                    raise RuntimeError(
                        f"Cannot read buffer info from device {device['device_id']} SHM. Device may not be properly initialized."
                    )
            else:
                # Cannot connect to device SHM
                raise RuntimeError(f"Cannot connect to device {device['device_id']} SHM. Device may not be running.")

        logger.info(f"Loaded {len(self.devices)} devices")

    def _connect_to_device(self, device: Dict[str, Any]) -> bool:
        """Connect to a device's SHM using unified interface (read-only for manager)."""
        if device["shm"] is not None:
            return True

        try:
            # Get device name directly from device class
            device_name = device["device_class"]
            device_id = device["device_id"]

            # Use unified SHM interface for read-only connection
            device["shm"] = connect_to_device_shm(device_name, device_id, read_only=True)
            logger.info(f"Connected to {device_name}_{device_id}_data (READ-ONLY)")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            # Define device info for error logging
            try:
                device_name = device["device_class"]
                device_id = device["device_id"]
                shm_name = f"{device_name}_{device_id}_data"
            except Exception:
                shm_name = f"unknown_{device['device_id']}_data"
            logger.error(f"Error connecting to {shm_name}: {e}")
            return False

    def _connect_to_device_with_retry(
        self, device: Dict[str, Any], max_retries: int = 5, retry_delay: float = 2.0
    ) -> bool:
        """Connect to a device's SHM with retry mechanism and countdown."""
        if device["shm"] is not None:
            return True

        device_name = device["device_class"]
        shm_name = f"{device_name}_{device['device_id']}_data"

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to {shm_name} (attempt {attempt + 1}/{max_retries})")

                if attempt == 0:
                    # First attempt - try immediate connection
                    device["shm"] = connect_to_device_shm(device_name, device["device_id"], read_only=True)
                    logger.info(f"Connected to {shm_name} on first attempt (READ-ONLY)")
                    return True
                else:
                    # Subsequent attempts - show countdown
                    remaining_attempts = max_retries - attempt
                    logger.info(
                        f"Connection failed. Retrying in {retry_delay:.1f}s... ({remaining_attempts} attempts remaining)"
                    )
                    time.sleep(retry_delay)

                    device["shm"] = connect_to_device_shm(device_name, device["device_id"], read_only=True)
                    logger.info(f"Connected to {shm_name} on attempt {attempt + 1} (READ-ONLY)")
                    return True

            except FileNotFoundError:
                if attempt < max_retries - 1:
                    logger.warning(f"Device {shm_name} not ready yet (attempt {attempt + 1}/{max_retries})")
                else:
                    logger.error(f"Failed to connect to {shm_name} after {max_retries} attempts")
                continue
            except Exception as e:
                logger.error(f"Error connecting to {shm_name} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue

        logger.error(f"Failed to connect to {shm_name} after {max_retries} attempts")
        return False

    def _read_device_frame(self, device: Dict[str, Any], frame_index: Optional[int] = None) -> Optional[tuple]:
        """
        Read a frame from a device.

        Args:
            device: Device dictionary
            frame_index: Frame index in buffer (0 = oldest, current_frames_count-1 = newest)
                        If None, reads the latest frame (write_index - 1)

        Returns:
            tuple: (timestamp_ns, data_array) or None
        """
        if device["shm"] is None:
            return None

        buffer_info = self._get_device_buffer_info(device)
        if buffer_info is None:
            return None

        buffer_size, current_frames_count, write_index, frame_size, shape = buffer_info

        # If no frame_index specified, read the latest frame
        if frame_index is None:
            if current_frames_count == 0:
                return None
            frame_index = (write_index - 1) % buffer_size
        else:
            # Validate frame_index
            if current_frames_count == 0 or frame_index >= current_frames_count:
                return None

        try:
            buf = device["shm"].buf

            frame_offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + frame_index * frame_size

            # Read frame header
            frame_header = unpack_frame_header(buf[frame_offset : frame_offset + FRAME_HEADER_SIZE])
            timestamp_ns = frame_header["timestamp_ns"]

            # Read frame data
            data_start = frame_offset + FRAME_HEADER_SIZE
            data_end = data_start + frame_size - FRAME_HEADER_SIZE

            # Get data type
            data_dtype = device["config"].get("data_dtype", "uint8")
            dtype = get_dtype(data_dtype, np.uint8)

            # Create data array - use zero-copy view for large data, copy for small data
            data_size = data_end - data_start
            if data_size > 1024:  # Zero-copy for data > 1KB
                # Direct view into SHM (zero-copy)
                # Calculate count based on data size and numpy dtype itemsize
                np_dtype = np.dtype(dtype)
                # Zero-copy: buf is now raw buffer from DataChannel.buf
                data_array = np.frombuffer(buf, dtype=dtype, count=data_size // np_dtype.itemsize, offset=data_start)
                if len(shape) > 0:
                    data_array = data_array.reshape(shape)
            else:
                # Copy for small data (safer and similar performance)
                sliced_buf = buf[data_start:data_end]
                data_array = np.frombuffer(sliced_buf, dtype=dtype)
                if len(shape) > 0:
                    data_array = data_array.reshape(shape)

            return timestamp_ns, data_array

        except Exception as e:
            logger.error(f"Error reading frame at index {frame_index} from device {device['device_id']}: {e}")
            return None

    def _get_device_buffer_info(self, device: Dict[str, Any]) -> Optional[tuple]:
        """
        Get device buffer information.

        Returns:
            tuple: (buffer_size, current_frames_count, write_index, frame_size, shape) or None
        """
        if device["shm"] is None:
            return None

        # Use cached buffer info if available (static info doesn't change)
        if device.get("buffer_info") is not None:
            cached_info = device["buffer_info"]
            # Handle both old (6 elements) and new (5 elements) formats
            if len(cached_info) == 6:
                buffer_size, _, _, frame_size, _, shape = cached_info
            else:
                buffer_size, _, _, frame_size, shape = cached_info

            try:
                buf = device["shm"].buf

                # Only read dynamic values (current_frames_count, write_index)
                buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
                current_frames_count = buffer_header["current_frames_count"]
                write_index = buffer_header["write_index"]

                return buffer_size, current_frames_count, write_index, frame_size, shape

            except Exception as e:
                logger.error(f"Error reading dynamic buffer info from device {device['device_id']}: {e}")
                return None

        # Fallback to full buffer info reading (for initialization)
        try:
            buf = device["shm"].buf

            # Read device header
            device_header = unpack_device_header(buf[:DEVICE_HEADER_SIZE])
            frame_size = device_header["frame_size"]
            shape = device_header["shape"]

            # Read buffer header
            buffer_header = unpack_buffer_header(buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE])
            buffer_size = buffer_header["buffer_size"]
            current_frames_count = buffer_header["current_frames_count"]
            write_index = buffer_header["write_index"]

            return buffer_size, current_frames_count, write_index, frame_size, shape

        except Exception as e:
            logger.error(f"Error reading buffer info from device {device['device_id']}: {e}")
            return None

    def _get_device_hardware_latency(self, device: Dict[str, Any]) -> float:
        """
        Get hardware latency for a device from config.

        Returns:
            float: Hardware latency in milliseconds
        """
        return device["config"].get("hardware_latency_ms", 0.0)

    def _create_summary_shm(self) -> None:
        """Create summary SHM with buffer support using unified interface."""
        # Calculate device parameters
        device_count = len(self.devices)
        frame_sizes = [device["frame_size"] for device in self.devices]

        with self.summary_shm_lock:
            # Use unified SHM creation interface
            self.summary_shm = create_summary_shm(
                device_count,
                frame_sizes,
                self.buffer_size,
                name=self.summary_shm_name,
            )

            total_size = calculate_summary_shm_size(device_count, frame_sizes, self.buffer_size)
            logger.info(f"Created summary SHM with buffer: {total_size:,} bytes")
            logger.info(
                f"Manager header: {MANAGER_HEADER_SIZE:,} bytes, Device headers: {device_count * DEVICE_HEADER_SIZE:,} bytes"
            )
            logger.info(
                f"Buffer size: {self.buffer_size} frames, Total data: {sum(frame_sizes) * self.buffer_size:,} bytes"
            )

            # Initialize header
            self._initialize_summary_header()

    def _initialize_summary_header(self) -> None:
        """Initialize summary SHM header with device headers."""
        device_count = len(self.devices)

        # Pack manager header with buffer info
        manager_header = pack_manager_header(
            device_count, self.buffer_size, self.current_frames_count, self.write_index, 0
        )

        # Write manager header to SHM
        self.summary_shm.buf[:MANAGER_HEADER_SIZE] = manager_header

        # Pack and write device headers
        for i, device in enumerate(self.devices):
            # Get device name directly from device class
            device_type = device["device_class"]

            device_id = device["device_id"]
            fps = device["config"].get("fps", 0.0)
            data_dtype = device["config"].get("data_dtype", "unknown")
            frame_size = device["frame_size"]
            hardware_latency_ms = device["config"].get("hardware_latency_ms", 0.0)
            shape = device["config"].get("data_shape", (1,))

            # Pack device header
            device_header = pack_device_header(
                device_type, device_id, fps, data_dtype, shape, frame_size, hardware_latency_ms
            )

            # Write device header to SHM
            device_offset = get_device_info_offset(i)
            self.summary_shm.buf[device_offset : device_offset + DEVICE_HEADER_SIZE] = device_header

    def _update_manager_header(self) -> None:
        """Update manager header with current buffer state."""
        current_time_ns = time.time_ns()
        manager_header = pack_manager_header(
            len(self.devices), self.buffer_size, self.current_frames_count, self.write_index, current_time_ns
        )
        self.summary_shm.buf[:MANAGER_HEADER_SIZE] = manager_header

    def _read_all_device_frames(self) -> Optional[List[Optional[tuple]]]:
        """
        Read latest frames from all devices.

        Returns:
            List of (timestamp_ns, data_array) tuples for each device, or None if should skip update
        """
        device_frames = []

        for device in self.devices:
            # Try to connect to device if not connected
            if not self._connect_to_device(device):
                device_frames.append(None)
                continue

            # Read latest frame
            frame_data = self._read_device_frame(device)
            device_frames.append(frame_data)

        return device_frames

    def write_frames_to_summary_shm(self, device_frames: List[Optional[tuple]]) -> None:
        """
        Write device frames to summary SHM buffer.

        Args:
            device_frames: List of (timestamp_ns, data_array) tuples for each device, or None if failed
        """
        with self.summary_shm_lock:
            self._write_frames_to_summary_shm_unlocked(device_frames)

    def _write_frames_to_summary_shm_unlocked(self, device_frames: List[Optional[tuple]]) -> None:
        """Write device frames to summary SHM buffer while caller holds the summary lock."""
        device_count = len(self.devices)
        frame_sizes = [device["frame_size"] for device in self.devices]

        # Get frame start offset
        frame_start_offset = get_summary_frame_start_offset(self.write_index, device_count, frame_sizes)

        # Write frame header with current timestamp
        frame_timestamp_ns = time.time_ns()
        summary_frame_header = pack_summary_frame_header(frame_timestamp_ns)
        self.summary_shm.buf[frame_start_offset : frame_start_offset + SUMMARY_FRAME_HEADER_SIZE] = summary_frame_header

        # Write all device frames at current write_index
        for i, (device, frame_data) in enumerate(zip(self.devices, device_frames)):
            if frame_data is None:
                continue

            timestamp_ns, data_array = frame_data

            # Calculate frame offset for this device at current write_index
            frame_offset = get_summary_frame_offset(i, self.write_index, device_count, frame_sizes, self.buffer_size)

            # Write device frame header
            frame_header = pack_frame_header(timestamp_ns)
            self.summary_shm.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = frame_header

            # Write data - use zero-copy for large arrays
            data_start = frame_offset + FRAME_HEADER_SIZE

            if data_array.nbytes > 1024:  # Zero-copy for data > 1KB
                # Zero-copy method: direct write using numpy view
                shm_view = np.frombuffer(
                    self.summary_shm.buf, dtype=data_array.dtype, count=data_array.size, offset=data_start
                )
                shm_view[:] = data_array.flat
            else:
                # Traditional method for small data
                data_bytes = data_array.tobytes()
                data_end = data_start + len(data_bytes)
                self.summary_shm.buf[data_start:data_end] = data_bytes

            # Update device tracking
            device["last_timestamp"] = timestamp_ns

        # Update buffer indices
        self.write_index = (self.write_index + 1) % self.buffer_size
        self.current_frames_count = min(self.current_frames_count + 1, self.buffer_size)

        # Update manager header with new buffer state
        self._update_manager_header()

    def _update_summary_shm(self) -> None:
        """Update summary SHM with latest device data."""
        # Read all device frames first
        device_frames = self._read_all_device_frames()

        # Check if frames are available (subclasses may return None to skip update)
        if device_frames is None:
            return

        # Write frames to summary SHM
        self.write_frames_to_summary_shm(device_frames)

    def start(self) -> None:
        """Start the device manager."""
        logger.info("Starting Device Manager")

        self.running = True

        try:
            while self.running:
                self._update_summary_shm()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the device manager."""
        if getattr(self, "_closed", False):
            return

        logger.info("Stopping Device Manager")
        self.running = False
        self._closed = True

        # Close device SHMs
        for device in self.devices:
            if device["shm"]:
                try:
                    device["shm"].close()
                except Exception as e:
                    logger.warning(f"Error closing device SHM: {e}")

        # Clean up summary SHM
        if self.summary_shm:
            try:
                self.summary_shm.close()
                self.summary_shm.unlink()
                logger.info("Cleaned up summary SHM")
            except Exception as e:
                logger.warning(f"Error cleaning up summary SHM: {e}")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Base Device Manager with buffer support")
    parser.add_argument("--config", "-c", type=str, default=None, help="Configuration file path (default: config.yaml)")
    parser.add_argument("--buffer-size", "-b", type=int, default=100, help="Buffer size for summary SHM (default: 100)")

    args = parser.parse_args()

    manager = BaseDeviceManager(config_path=args.config or "config.yaml", buffer_size=args.buffer_size)
    manager.start()


if __name__ == "__main__":
    main()
