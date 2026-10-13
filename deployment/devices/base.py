#!/usr/bin/env python3
"""
Base device class for data generation and shared memory management.

Author: Jun Lv
"""

import time
import numpy as np
from typing import Dict, Any
from utils.shm_utils import (
    pack_device_header,
    pack_buffer_header,
    pack_frame_header,
    calculate_device_shm_size,
    DEVICE_HEADER_SIZE,
    BUFFER_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    get_dtype,
    create_device_shm,
)
from utils.logger_config import logger
from utils.time_control import precise_loop_timing


class BaseDevice:
    """
    Base device class that generates data and writes to shared memory.
    Designed to be extensible for camera devices in the future.
    Supports continuous memory buffer for multiple frames.
    """

    def __init__(
        self,
        device_id: int = 0,
        device_name=None,  # Use class name if not
        data_shape: tuple = (128,),
        fps: float = 10.0,
        data_dtype: Any = np.float64,
        buffer_size: int = 1,
        hardware_latency_ms: float = 0.0,
    ) -> None:
        """
        Initialize the base device.

        Args:
            device_id: Unique identifier for this device instance
            device_name: Name of the device (default: BaseDevice)
            data_shape: Shape of data to generate (tuple or list for multi-dimensional)
            fps: Frames per second for data generation
            data_dtype: Data type for numpy array (string or numpy dtype)
            buffer_size: Number of frames to store in buffer (default: 1)
            hardware_latency_ms: Hardware latency in milliseconds (default: 0.0)
        """
        self.device_name = device_name or self.__class__.__name__
        self.device_id = device_id

        # Convert data_shape to tuple if it's a list
        self.data_shape = tuple(data_shape) if isinstance(data_shape, list) else data_shape
        self.data_dtype = data_dtype

        self.fps = fps
        self.hardware_latency_ms = hardware_latency_ms
        self.update_interval = 1.0 / fps  # Calculate interval from fps
        self.shared_memory_name = f"{self.device_name}_{self.device_id}_data"
        self.running = False
        self.shared_memory = None

        # Buffer management
        self.buffer_size = buffer_size
        self.write_index = 0  # next place to write
        self.current_frames_count = 0

        # Calculate buffer layout
        self._calculate_buffer_layout()

    def _calculate_buffer_layout(self) -> None:
        """Calculate the layout of the continuous memory buffer."""
        # Calculate data size per frame
        numpy_dtype = get_dtype(self.data_dtype)
        bytes_per_element = np.dtype(numpy_dtype).itemsize
        data_size = np.prod(self.data_shape) * bytes_per_element

        # Frame header: timestamp (8 bytes)
        self.frame_header_size = FRAME_HEADER_SIZE
        self.frame_data_size = data_size
        self.frame_size = self.frame_header_size + self.frame_data_size

        # Buffer header: buffer_size (4) + current_frames_count (4) + write_index (4) + padding (4)
        self.buffer_header_size = BUFFER_HEADER_SIZE

        # Pre-calculate frame offsets
        self.frame_offsets = [
            DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + i * self.frame_size for i in range(self.buffer_size)
        ]

        # Total SHM size: device_header + buffer_header + frames
        self.total_shm_size = calculate_device_shm_size(self.buffer_size, self.frame_data_size)

    def _generate_random_array(self) -> np.ndarray:
        """
        Generate random data internally. This is a private method.

        Returns:
            numpy.ndarray: Generated random array
        """
        numpy_dtype = get_dtype(self.data_dtype)

        # Generate random data as numpy array with specified shape and dtype
        if np.issubdtype(numpy_dtype, np.integer):
            # For integer types, generate random integers
            if numpy_dtype == np.uint8:
                random_array = np.random.randint(0, 256, size=self.data_shape, dtype=numpy_dtype)
            elif numpy_dtype == np.int8:
                random_array = np.random.randint(-128, 128, size=self.data_shape, dtype=numpy_dtype)
            elif numpy_dtype == np.uint16:
                random_array = np.random.randint(0, 65536, size=self.data_shape, dtype=numpy_dtype)
            elif numpy_dtype == np.int16:
                random_array = np.random.randint(-32768, 32768, size=self.data_shape, dtype=numpy_dtype)
            elif numpy_dtype == np.uint32:
                random_array = np.random.randint(0, 4294967296, size=self.data_shape, dtype=numpy_dtype)
            elif numpy_dtype == np.int32:
                random_array = np.random.randint(-2147483648, 2147483648, size=self.data_shape, dtype=numpy_dtype)
            else:
                # For other integer types, use general approach
                random_array = np.random.randint(0, 1000, size=self.data_shape, dtype=numpy_dtype)
        else:
            # For floating point types, generate random floats
            random_array = np.random.rand(*self.data_shape).astype(numpy_dtype)

        return random_array

    def _create_shared_memory(self) -> None:
        """Create shared memory with buffer layout using unified interface."""
        # Use unified SHM creation interface
        self.shared_memory = create_device_shm(
            device_name=self.device_name,
            device_id=self.device_id,
            buffer_size=self.buffer_size,
            frame_data_size=self.frame_data_size,
        )

        logger.info(f"Created shared memory: {self.shared_memory_name} ({self.total_shm_size:,} bytes)")
        logger.info(f"Buffer size: {self.buffer_size} frames, Frame size: {self.frame_size:,} bytes")

        # Initialize buffer header
        self._initialize_buffer_header()

        # Verify creation
        try:
            from utils.shm_utils import connect_to_device_shm

            test_shm = connect_to_device_shm(self.device_name, self.device_id, read_only=True)
            test_shm.close()
            logger.info(f"Shared memory {self.shared_memory_name} verified successfully")
        except Exception as e:
            logger.warning(f"Could not verify shared memory creation: {e}")

    def _initialize_buffer_header(self) -> None:
        """Initialize the buffer header with metadata."""
        # Pack device header
        device_header = pack_device_header(
            self.device_name,
            self.device_id,
            self.fps,
            self.data_dtype,
            self.data_shape,
            self.frame_size,
            self.hardware_latency_ms,
        )

        # Pack buffer header
        buffer_header = pack_buffer_header(self.buffer_size, self.current_frames_count, self.write_index)

        # Write headers to SHM
        self.shared_memory.buf[:DEVICE_HEADER_SIZE] = device_header
        self.shared_memory.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header

    def _update_buffer_header(self) -> None:
        """Update the buffer header with current indices."""
        # Update current_frames_count and write_index in buffer header
        buffer_header = pack_buffer_header(self.buffer_size, self.current_frames_count, self.write_index)
        self.shared_memory.buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE] = buffer_header

    def _cleanup_shared_memory(self) -> None:
        """Clean up shared memory resources."""
        if self.shared_memory:
            try:
                self.shared_memory.close()
                # Only unlink if we're the creator (not a read-only consumer)
                if hasattr(self.shared_memory, "unlink"):
                    self.shared_memory.unlink()
                logger.info(f"Cleaned up shared memory: {self.shared_memory_name}")
            except (FileNotFoundError, PermissionError):
                pass  # Already cleaned up or no permission
            except Exception as e:
                logger.error(f"Error cleaning up shared memory: {e}")
            finally:
                self.shared_memory = None

    def _write_array_to_shm_with_timestamp(self, array: np.ndarray, timestamp_ns: int) -> None:
        """
        Write numpy array to shared memory buffer with provided timestamp.
        Uses zero-copy optimization for large arrays (>1KB) and fallback to tobytes() for small arrays.

        Args:
            array: Numpy array to write to shared memory
            timestamp_ns: Timestamp in nanoseconds

        Returns:
            bool: True if write successful, False otherwise
        """
        try:
            # Calculate frame offset
            frame_offset = self.frame_offsets[self.write_index]

            # Pack frame header: timestamp (8 bytes)
            frame_header = pack_frame_header(timestamp_ns)
            self.shared_memory.buf[frame_offset : frame_offset + FRAME_HEADER_SIZE] = frame_header

            # Choose write method based on data size
            data_start_offset = frame_offset + FRAME_HEADER_SIZE

            if array.nbytes > 1024:  # Use zero-copy for data > 1KB
                # Zero-copy method: direct write using numpy view
                # Zero-copy write: DataChannel.buf returns raw buffer
                buf = self.shared_memory.buf
                shm_view = np.frombuffer(buf, dtype=array.dtype, count=array.size, offset=data_start_offset)
                shm_view[:] = array.flat
            else:
                # Traditional method for small data (overhead of zero-copy not worth it)
                data_bytes = array.tobytes()
                self.shared_memory.buf[data_start_offset : data_start_offset + array.nbytes] = data_bytes

            # Update buffer indices
            self.write_index = (self.write_index + 1) % self.buffer_size
            self.current_frames_count = min(self.current_frames_count + 1, self.buffer_size)

            # Update buffer header
            self._update_buffer_header()

            return True

        except Exception as e:
            logger.error(f"Error writing array to shared memory buffer: {e}")
            return False

    def start_server(self) -> None:
        """Start the device server and run data generation loop."""
        if self.running:
            logger.info(f"Device {self.device_name}_{self.device_id} is already running")
            return

        logger.info(f"Starting {self.device_name}_{self.device_id} server...")
        self.running = True

        # Create shared memory
        self._create_shared_memory()
        if not self.shared_memory:
            self.running = False
            raise RuntimeError("Failed to create shared memory")

        logger.info(f"Server started. Shared memory: {self.shared_memory_name}")
        logger.info(f"Buffer configuration: {self.buffer_size} frames, {self.frame_size:,} bytes per frame")

        # Create precise timing function
        wait_for_next_iteration = precise_loop_timing(self.update_interval)

        # Main data generation loop
        while self.running:
            try:
                # Get timestamp immediately when starting data generation
                random_array = self._generate_random_array()
                timestamp_ns = time.time_ns()
                self._write_array_to_shm_with_timestamp(random_array, timestamp_ns)

                # Wait for next iteration using precise timing
                wait_for_next_iteration()
            except Exception as e:
                logger.error(f"Error in data generation: {e}")
                break

    def stop_server(self) -> None:
        """Stop the device server."""
        if not self.running:
            return

        logger.info(f"Stopping {self.device_name}_{self.device_id} server...")
        self.running = False
        self._cleanup_shared_memory()
        logger.info("Server stopped")

    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            "device_name": self.device_name,
            "device_id": self.device_id,
            "data_shape": self.data_shape,
            "data_dtype": str(self.data_dtype),
            "fps": self.fps,
            "update_interval": self.update_interval,
            "shared_memory_name": self.shared_memory_name,
            "running": self.running,
            "buffer_size": self.buffer_size,
            "frame_size": self.frame_size,
            "total_shm_size": self.total_shm_size,
        }

    def __del__(self) -> None:
        """Destructor to ensure shared memory is cleaned up."""
        try:
            if hasattr(self, "shared_memory") and self.shared_memory:
                self._cleanup_shared_memory()
        except Exception:
            pass  # Ignore errors during cleanup in destructor


def main() -> None:
    """Main function to run the device server."""
    import argparse

    parser = argparse.ArgumentParser(description="Base device server for data generation")
    parser.add_argument("--device-id", "-d", type=int, default=0, help="Device ID (default: 0)")
    parser.add_argument(
        "--shape", "-s", type=str, default="16,16", help="Data shape as comma-separated values (default: 16,16)"
    )
    parser.add_argument("--fps", "-f", type=float, default=5.0, help="Frames per second (default: 5.0)")
    parser.add_argument(
        "--dtype",
        "-t",
        default="uint8",
        choices=["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"],
        help="Data type (default: uint8)",
    )
    parser.add_argument("--buffer-size", "-b", type=int, default=1, help="Buffer size in frames (default: 1)")

    args = parser.parse_args()

    # Parse shape string to tuple
    try:
        data_shape = tuple(int(x.strip()) for x in args.shape.split(","))
    except ValueError:
        logger.error("Invalid shape format. Use comma-separated integers (e.g., '16,16' or '32')")
        return

    # Import common utilities
    from utils.shm_utils import get_dtype

    data_dtype = get_dtype(args.dtype)

    # Create device with parsed arguments
    device = BaseDevice(
        device_id=args.device_id,
        data_shape=data_shape,
        fps=args.fps,
        data_dtype=data_dtype,
        buffer_size=args.buffer_size,
    )

    logger.info("Base Device Server")
    logger.info("==================")
    logger.info(f"Device ID: {args.device_id}")
    logger.info(f"Data shape: {data_shape}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Data type: {args.dtype}")
    logger.info(f"Buffer size: {args.buffer_size} frames")
    logger.info("")

    try:
        logger.info("Device server is running. Press Ctrl+C to stop...")
        logger.info(f"Device info: {device.get_device_info()}")
        device.start_server()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        device.stop_server()


if __name__ == "__main__":
    main()
