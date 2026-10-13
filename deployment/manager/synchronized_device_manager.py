#!/usr/bin/env python3
"""
Synchronized Device Manager - Ensures all device data has very similar timestamps.

This manager reads the latest frame from a master device, then finds frames from
other devices with timestamps closest to the master device's timestamp, considering
hardware latency compensation.

Author: Jun Lv
"""

from typing import List, Optional, Dict, Any
from .base_device_manager import BaseDeviceManager
from utils.logger_config import logger


class SynchronizedDeviceManager(BaseDeviceManager):
    """
    Synchronized Device Manager for time-synchronized device data aggregation.

    This manager ensures that all device data has very similar timestamps by:
    1. Reading the latest frame from a master device
    2. Finding frames from other devices with timestamps closest to the master's timestamp
    3. Compensating for hardware latency differences
    """

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, *, config_path: Optional[str] = None, buffer_size: int = 100
    ) -> None:
        """
        Initialize the Synchronized Device Manager.

        Args:
            config: Configuration dictionary (for Hydra mode)
            config_path: Path to configuration file (for traditional mode)
            buffer_size: Buffer size for summary SHM
        """
        super().__init__(config=config, config_path=config_path, buffer_size=buffer_size)

        # Get master device ID from config
        self._get_master_device_id()

        # Track master device timestamp for update checking
        self.master_last_timestamp = None

    def _get_master_device_id(self) -> None:
        """Get master device index from config file and validate."""
        # Get master device index from config
        self.master_device_id = self.config.get("device_manager", {}).get("master_device_id")

        if self.master_device_id is None:
            raise ValueError("master_device_id not found in configuration file")

        # Validate master device index
        if self.master_device_id < 0 or self.master_device_id >= len(self.devices):
            raise ValueError(
                f"Master device index {self.master_device_id} is out of range. "
                f"Valid range: 0 to {len(self.devices) - 1}"
            )

        # Get master device by index
        master_device = self.devices[self.master_device_id]

        # Print complete master device information
        logger.info("=" * 50)
        logger.info("MASTER DEVICE CONFIGURATION")
        logger.info("=" * 50)
        logger.info(f"Master Device Index: {self.master_device_id}")
        logger.info(f"Device ID: {master_device['device_id']}")
        logger.info(f"Device Class: {master_device['device_class']}")
        logger.info("Device Config:")
        for key, value in master_device["config"].items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 50)

    def _calculate_adjusted_timestamp(self, master_timestamp_ns: int, device: Dict[str, Any]) -> int:
        """
        Calculate adjusted timestamp for a device considering hardware latency.

        Args:
            master_timestamp_ns: Master device timestamp in nanoseconds
            device: Device dictionary

        Returns:
            int: Adjusted timestamp in nanoseconds
        """
        # Get master device by index
        master_device = self.devices[self.master_device_id]

        # Get hardware latency for the device
        hardware_latency_ms = self._get_device_hardware_latency(device)
        master_latency_ms = self._get_device_hardware_latency(master_device)

        # Calculate latency difference
        latency_diff_ms = hardware_latency_ms - master_latency_ms
        latency_diff_ns = int(latency_diff_ms * 1_000_000)  # Convert ms to ns

        # Adjust timestamp
        adjusted_timestamp = master_timestamp_ns + latency_diff_ns

        logger.debug(
            f"Device {device['device_id']}: latency={hardware_latency_ms}ms, adjusted_timestamp={adjusted_timestamp}"
        )

        return adjusted_timestamp

    def _find_synchronized_frames(self, master_timestamp_ns: int) -> List[Optional[tuple]]:
        """
        Find synchronized frames for all devices based on master timestamp.

        Args:
            master_timestamp_ns: Master device timestamp in nanoseconds

        Returns:
            List of (timestamp_ns, data_array) tuples for each device, or None if failed
        """
        device_frames = []

        for device in self.devices:
            if device["device_id"] == self.master_device_id:
                # For master device, use the frame we already read
                frame_data = self._read_device_frame(device)
                device_frames.append(frame_data)
                continue

            # For other devices, find frame with closest timestamp
            adjusted_timestamp = self._calculate_adjusted_timestamp(master_timestamp_ns, device)

            # Try to connect to device if not connected
            if not self._connect_to_device(device):
                logger.warning(f"Cannot connect to device {device['device_id']}")
                device_frames.append(None)
                continue

            # Search for frame with closest timestamp
            search_result = self._search_device_buffer_for_timestamp(device, adjusted_timestamp)

            if search_result is None:
                logger.warning(f"Cannot find synchronized frame for device {device['device_id']}")
                device_frames.append(None)
                continue

            timestamp_ns, data_array, frame_index = search_result
            logger.debug(
                f"Device {device['device_id']}: found synchronized frame at index {frame_index}, "
                f"timestamp_diff={abs(timestamp_ns - adjusted_timestamp) / 1_000_000:.2f}ms"
            )
            device_frames.append((timestamp_ns, data_array))

        return device_frames

    def _search_device_buffer_for_timestamp(self, device: Dict[str, Any], target_timestamp_ns: int) -> Optional[tuple]:
        """
        Search device buffer for frame with timestamp closest to target using binary search.

        Args:
            device: Device dictionary
            target_timestamp_ns: Target timestamp in nanoseconds

        Returns:
            tuple: (timestamp_ns, data_array, frame_index) or None
        """
        buffer_info = self._get_device_buffer_info(device)
        if buffer_info is None:
            return None

        buffer_size, current_frames_count, write_index, frame_size, shape = buffer_info

        if current_frames_count == 0:
            return None

        try:
            # Binary search on buffer indices (frames are already time-ordered)
            result = self._binary_search_buffer_indices(device, target_timestamp_ns, buffer_info)

            if result is None:
                return None

            # Return the result directly (timestamp_ns, data_array, frame_index)
            return result

        except Exception as e:
            logger.error(
                f"Error searching buffer for timestamp {target_timestamp_ns} in device {device['device_id']}: {e}"
            )
            return None

    def _binary_search_buffer_indices(
        self, device: Dict[str, Any], target_timestamp_ns: int, buffer_info: tuple
    ) -> Optional[tuple]:
        """
        Binary search on buffer indices to find the frame with closest timestamp.
        Handles circular buffer time ordering correctly.

        Args:
            device: Device dictionary
            target_timestamp_ns: Target timestamp in nanoseconds
            buffer_info: Buffer information tuple

        Returns:
            tuple: (timestamp_ns, data_array, frame_index) or None if not found
        """
        buffer_size, current_frames_count, write_index, _, _ = buffer_info

        if current_frames_count == 0:
            return None

        if current_frames_count < buffer_size:
            return self._binary_search_range(device, target_timestamp_ns, 0, current_frames_count - 1, buffer_info)

        return self._binary_search_range(
            device, target_timestamp_ns, write_index, write_index + buffer_size - 1, buffer_info
        )

    def _binary_search_range(
        self, device: Dict[str, Any], target_timestamp_ns: int, start_idx: int, end_idx: int, buffer_info: tuple
    ) -> Optional[tuple]:
        """
        Binary search in a range of buffer indices (time-ordered).
        Handles circular buffer by converting virtual indices to actual buffer indices.

        Args:
            device: Device dictionary
            target_timestamp_ns: Target timestamp in nanoseconds
            start_idx: Start index (inclusive, can be virtual for circular buffer)
            end_idx: End index (inclusive, can be virtual for circular buffer)
            buffer_info: Buffer information tuple

        Returns:
            tuple: (timestamp_ns, data_array, frame_index) or None if not found
        """
        if start_idx > end_idx:
            return None

        buffer_size = buffer_info[0]  # Get buffer_size

        left, right = start_idx, end_idx
        closest_result = None
        closest_diff = float("inf")

        while left <= right:
            mid = (left + right) // 2

            # Convert virtual index to actual buffer index
            actual_idx = mid % buffer_size

            # Read timestamp at actual index
            frame_data = self._read_device_frame(device, actual_idx)
            if frame_data is None:
                # Skip invalid frame, continue search
                right = mid - 1
                continue

            timestamp_ns, data_array = frame_data
            current_diff = abs(timestamp_ns - target_timestamp_ns)

            # Update closest if current is better
            if current_diff < closest_diff:
                closest_diff = current_diff
                closest_result = (timestamp_ns, data_array, actual_idx)

            # If we found exact match, return it
            if timestamp_ns == target_timestamp_ns:
                return timestamp_ns, data_array, actual_idx

            # Binary search logic
            if timestamp_ns < target_timestamp_ns:
                left = mid + 1
            else:
                right = mid - 1

        return closest_result

    def _apply_hardware_latency_compensation(self, device: Dict[str, Any], timestamp_ns: int) -> int:
        """
        Apply hardware latency compensation to timestamp.

        Args:
            device: Device dictionary
            timestamp_ns: Original timestamp in nanoseconds

        Returns:
            int: Compensated timestamp in nanoseconds
        """
        hardware_latency_ms = self._get_device_hardware_latency(device)
        hardware_latency_ns = int(hardware_latency_ms * 1_000_000)  # Convert ms to ns
        compensated_timestamp_ns = timestamp_ns - hardware_latency_ns

        logger.debug(
            f"Device {device['device_id']}: original_timestamp={timestamp_ns}, "
            f"hardware_latency={hardware_latency_ms}ms, "
            f"compensated_timestamp={compensated_timestamp_ns}"
        )

        return compensated_timestamp_ns

    def write_frames_to_summary_shm(self, device_frames: List[Optional[tuple]]) -> None:
        """
        Write device frames to summary SHM buffer with hardware latency compensation.

        This method overrides the base implementation to apply hardware latency compensation
        to timestamps before writing them to the summary SHM.

        Args:
            device_frames: List of (timestamp_ns, data_array) tuples for each device, or None if failed
        """
        # Apply hardware latency compensation to each device frame
        compensated_frames = []
        for i, (device, frame_data) in enumerate(zip(self.devices, device_frames)):
            if frame_data is None:
                compensated_frames.append(None)
                continue

            timestamp_ns, data_array = frame_data
            compensated_timestamp_ns = self._apply_hardware_latency_compensation(device, timestamp_ns)
            compensated_frames.append((compensated_timestamp_ns, data_array))

        # Call parent method with compensated frames
        super().write_frames_to_summary_shm(compensated_frames)

    def _read_all_device_frames(self) -> Optional[List[Optional[tuple]]]:
        """
        Read synchronized frames from all devices.

        Returns:
            List of (timestamp_ns, data_array) tuples for each device, or None if master device hasn't updated
        """
        # Get master device by index
        master_device = self.devices[self.master_device_id]

        # Read master device frame first
        master_frame = self._read_device_frame(master_device)
        if master_frame is None:
            logger.error("Cannot read master device frame")
            return None

        master_timestamp_ns, master_data = master_frame

        # Check if master device has new data
        if self.master_last_timestamp is not None and master_timestamp_ns == self.master_last_timestamp:
            return None  # Master device hasn't updated, skip update

        # Update master timestamp
        self.master_last_timestamp = master_timestamp_ns

        # Find synchronized frames for all devices
        synchronized_frames = self._find_synchronized_frames(master_timestamp_ns)

        return synchronized_frames


def main() -> None:
    """Main entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Synchronized Device Manager with buffer support")
    parser.add_argument("--config", "-c", default="config.yaml", help="Configuration file path")
    parser.add_argument("--buffer-size", "-b", type=int, default=100, help="Buffer size for summary SHM (default: 100)")

    args = parser.parse_args()

    manager = SynchronizedDeviceManager(config_path=args.config, buffer_size=args.buffer_size)

    manager.start()


if __name__ == "__main__":
    main()
