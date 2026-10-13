#!/usr/bin/env python3
"""
Auto Action Manager - automatically finds optimal start index for action chunk connection.

This manager:
1. Calculates elapsed steps based on observation timestamp and current time
2. Retrieves current executing action from previous chunk
3. Searches for optimal start index by finding closest action in new chunk
4. Uses the optimal start index as the initial current_step for new chunk

The logic is migrated from PolicyConnector's smart start detection method.

Author: Assistant, Based on PolicyConnector implementation
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from utils.logger_config import logger
from .action_manager import BaseChunkManager


class JointActionManager(BaseChunkManager):
    """
    Auto action manager that automatically finds optimal start index for seamless chunk connection.

    This manager eliminates obsolete actions from each new action chunk by:
    1. Calculating elapsed time from observation timestamp to current time
    2. Estimating elapsed steps using execution FPS
    3. Searching for closest action to current executing action within tolerance range
    4. Setting optimal start index as initial current_step

    Args:
        execution_fps (float): Execution frequency in Hz for calculating elapsed steps
        search_tolerance (int): Tolerance for search range around estimated start index (default: 2)
        discount_factor (float): Discount factor for penalizing indices far from estimated_steps_elapsed.
                                 At estimated_steps_elapsed, discount=1.0; at ±1, discount=discount_factor;
                                 at ±2, discount=discount_factor^2, etc. (default: 1.0, no discount)
        direction_consistency_range (int): Number of steps to check for direction consistency after
                                           optimal start index. The search_end will be reduced by this
                                           value to ensure enough actions remain for consistency check.
                                           (default: 5)
    """

    def __init__(
        self,
        execution_fps: float = 100.0,
        search_tolerance: int = 2,
        discount_factor: float = 1.0,
        direction_consistency_range: int = 5,
    ):
        super().__init__()

        self.execution_fps = execution_fps
        self.search_tolerance = search_tolerance
        self.discount_factor = discount_factor
        self.direction_consistency_range = direction_consistency_range

        # Calculate dt (time per step) in seconds
        self.dt_seconds = 1.0 / execution_fps

        # Chunk counter (starts from 0, first chunk will be 1)
        self.chunk_counter = 0

        # Store original chunk info for action_index calculation
        self.current_optimal_start_index = 0  # The slice start index in original chunk
        self.current_original_chunk_length = 0  # The length of original chunk before slicing
        self._latest_original_action_index: Optional[int] = None

        logger.info(
            f"AutoActionManager initialized: execution_fps={execution_fps}Hz, "
            f"dt={self.dt_seconds}s, search_tolerance={search_tolerance}, "
            f"discount_factor={discount_factor}, "
            f"direction_consistency_range={direction_consistency_range}, "
        )

    def reset(self) -> None:
        """Reset the chunk manager state. Should be called when episode ends/resets."""
        # Reset base class state
        self.current_action_chunk_dict = None
        self.current_step = 0
        self.last_read_time = 0
        # Reset AutoActionManagerV2-specific state
        self.chunk_counter = 0
        self.current_optimal_start_index = 0
        self.current_original_chunk_length = 0
        self._latest_original_action_index = None
        logger.info("AutoActionManagerV2: State reset (chunk_counter=0, action_index cleared)")

    def put(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        observation_timestamp: Optional[int] = None,
        last_action_dict: Optional[Dict[str, np.ndarray]] = None,
        steps_elapsed: Optional[int] = None,
        search_tolerance: Optional[int] = None,
        discount_factor: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """
        Process new action chunk by finding optimal start index automatically.

        Args:
            action_chunk_dict: New action chunk dictionary from policy
            observation_timestamp: Timestamp when observation was taken (nanoseconds)
            last_action_dict: Not used in this manager (kept for compatibility)
            steps_elapsed: Not used in this manager (kept for compatibility)
            search_tolerance: Optional override for search tolerance
            discount_factor: Optional override for discount factor
            kwargs: Additional keyword arguments kept for forward compatibility
        """
        if kwargs:
            logger.debug(
                "JointActionManager: Received additional kwargs in put call: %s",
                list(kwargs.keys()),
            )

        # Use override values if provided, otherwise use configured defaults
        search_tolerance_value = search_tolerance if search_tolerance is not None else self.search_tolerance
        discount_factor_value = discount_factor if discount_factor is not None else self.discount_factor

        # Get device names and chunk length
        device_names = list(action_chunk_dict.keys())
        if not device_names:
            logger.warning("JointActionManager: Empty action_chunk_dict, skipping")
            return

        first_chunk = action_chunk_dict[device_names[0]]
        chunk_length = len(first_chunk)

        # Increment chunk counter and log
        self.chunk_counter += 1
        logger.info(f"JointActionManager: Processing action chunk #{self.chunk_counter}, length={chunk_length}")

        # If this is the first chunk, just store it and start from beginning
        if self.current_action_chunk_dict is None:
            logger.info(f"JointActionManager: First chunk received, length={chunk_length}, starting from index 0")
            # Deep copy to avoid shared memory reference issues
            self.current_action_chunk_dict = {k: v.copy() for k, v in action_chunk_dict.items()}
            self.last_read_time = observation_timestamp
            self.current_step = 0
            self.current_optimal_start_index = 0  # First chunk starts from index 0
            self.current_original_chunk_length = chunk_length
            self._latest_original_action_index = 0
            return

        # Calculate elapsed steps from observation timestamp to current time
        if observation_timestamp is None:
            logger.warning("JointActionManager: observation_timestamp is None, using current time")
            observation_timestamp = time.time_ns()

        current_time_ns = time.time_ns()
        elapsed_time_seconds = (current_time_ns - observation_timestamp) / 1e9
        estimated_steps_elapsed = int(elapsed_time_seconds / self.dt_seconds)

        logger.info(
            f"AutoActionManager: obs_timestamp={observation_timestamp}, "
            f"current_time={current_time_ns}, "
            f"elapsed_time={elapsed_time_seconds:.4f}s, "
            f"estimated_steps_elapsed={estimated_steps_elapsed}"
        )

        # Get current executing action from previous chunk
        current_executing_action = self._get_current_executing_action()  # current state
        device_names = list(action_chunk_dict.keys())
        if current_executing_action is None:
            raise ValueError("None Action")
        else:
            # Search for optimal start index
            # If estimated_steps_elapsed >= chunk_length, search the entire chunk due to severe delay
            optimal_start_index = self._find_optimal_start_index(
                action_chunk_dict=action_chunk_dict,
                current_executing_action=current_executing_action,
                estimated_steps_elapsed=estimated_steps_elapsed,
                search_tolerance=search_tolerance_value,
                discount_factor=discount_factor_value,
                chunk_length=chunk_length,
                direction_consistency_range=self.direction_consistency_range,
            )
            if optimal_start_index is None:
                logger.warning("JointActionManager: optimal_start_index is None, skipping")
                return

        # Store new chunk and set current_step to optimal_start_index
        # Deep copy to avoid shared memory reference issues
        new_action_chunk_dict = {k: v.copy()[optimal_start_index:,] for k, v in action_chunk_dict.items()}

        self.current_action_chunk_dict = new_action_chunk_dict
        self.last_read_time = observation_timestamp
        self.current_step = 0
        self.current_optimal_start_index = optimal_start_index  # Save slice start index for action_index calculation
        self.current_original_chunk_length = chunk_length  # Save original chunk length for action_index calculation
        self._latest_original_action_index = optimal_start_index

        logger.info(
            f"JointActionManager: Chunk #{self.chunk_counter} stored with optimal start index={optimal_start_index}, "
            f"chunk_length={chunk_length}, devices={device_names}"
        )

    def _get_current_executing_action(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get current executing action from previous chunk based on current_step.

        Returns:
            Dictionary of current actions for each device, or None if not available
        """
        if self.current_action_chunk_dict is None:
            return None

        if self.current_step < 0:
            return None

        # Get chunk length to validate current_step
        device_names = list(self.current_action_chunk_dict.keys())
        if not device_names:
            return None

        first_chunk = self.current_action_chunk_dict[device_names[0]]
        chunk_length = len(first_chunk)

        # If current_step is beyond chunk length, use the last action
        step_index = min(self.current_step, chunk_length - 1)

        current_executing_action = {}
        for device_name, chunk in self.current_action_chunk_dict.items():
            if step_index < len(chunk):
                current_executing_action[device_name] = chunk[step_index]
            else:
                logger.warning(f"AutoActionManager: step_index {step_index} out of range for {device_name}")
                return None

        logger.debug(f"AutoActionManager: Retrieved current executing action at step {step_index}/{chunk_length}")
        return current_executing_action

    def get(self, query_timestamp: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Get next action dict from cached current chunk dict.
        Override base class to use original chunk index for action_index calculation.

        Args:
            query_timestamp: Not used, kept for compatibility

        Returns:
            Dictionary of actions for each device, or None if no more actions
        """
        try:
            if self.current_action_chunk_dict is None:
                return None

            # Get the chunk length from any device (they should all have the same chunk length)
            device_names = list(self.current_action_chunk_dict.keys())
            if not device_names:
                return None

            first_chunk = self.current_action_chunk_dict[device_names[0]]
            sliced_chunk_length = len(first_chunk)

            # Check if we have more actions in current chunk
            if self.current_step < sliced_chunk_length:
                # Extract action for current step from all devices
                action_dict = {}
                for device_name, chunk in self.current_action_chunk_dict.items():
                    action_dict[device_name] = chunk[self.current_step]

                # Calculate action_index using original chunk index before incrementing
                # The current action being dispatched corresponds to original chunk at index:
                # optimal_start_index + current_step
                original_chunk_index = self.current_optimal_start_index + self.current_step

                # Increment current_step (consistent with base class behavior)
                self.current_step += 1

                # Clamp to valid range [0, original_chunk_length - 1]
                # This represents the index in the original chunk for the action that was just dispatched
                clamped_action_index = min(original_chunk_index, max(0, self.current_original_chunk_length - 1))
                self._latest_original_action_index = clamped_action_index

                logger.debug(
                    f"AutoActionManager: Dispatching action step {self.current_step - 1}/{sliced_chunk_length} "
                    f"(original index {clamped_action_index}/{self.current_original_chunk_length}) "
                    f"for devices: {device_names}, action_index={clamped_action_index}"
                )
                return action_dict

            # No more actions in current chunk
            return None

        except Exception as e:
            logger.error(f"AutoActionManager: Error getting next action from current chunk: {e}")
            return None

    def get_original_current_step(self) -> Optional[int]:
        """Return the latest action_index in the original chunk, if available."""
        return self._latest_original_action_index

    def _calc_joint_action_distance(self, curr_action: np.ndarray, prev_action: np.ndarray) -> float:
        """
        Calculate the distance between two joint actions.
        """
        return np.linalg.norm(curr_action - prev_action)

    def _find_optimal_start_index(
        self,
        action_chunk_dict: Dict[str, np.ndarray],
        current_executing_action: Dict[str, np.ndarray],
        estimated_steps_elapsed: int,
        search_tolerance: int,
        discount_factor: float,
        chunk_length: Optional[int],
        direction_consistency_range: int = 0,
    ) -> int:
        """
        Find optimal start index by searching for closest action to current executing action.

        Applies a discount factor to penalize indices far from estimated_steps_elapsed:
        - At estimated_steps_elapsed: discount = 1.0
        - At estimated_steps_elapsed ± n: discount = discount_factor^n
        - Weighted distance = raw_distance / discount (larger penalty for distant indices)

        If estimated_steps_elapsed >= chunk_length, indicates severe delay, and the entire chunk
        will be searched instead of a limited range around estimated_steps_elapsed.

        Args:
            action_chunk_dict: New action chunk dict
            current_executing_action: Current action being executed
            estimated_steps_elapsed: Estimated steps elapsed during observation + prediction
            search_tolerance: Tolerance for search range (only used when estimated_steps_elapsed < chunk_length)
            discount_factor: Discount factor for penalizing distant indices (1.0 = no discount)
            chunk_length: Length of the action chunk
            direction_consistency_range: Number of steps reserved for direction consistency check.
                                         search_end will be reduced by this value to avoid boundary issues.

        Returns:
            Optimal start index (0-based)
        """
        try:
            # Get chunk length from first device
            device_names = list(action_chunk_dict.keys())
            if not device_names:
                return None

            first_device = device_names[0]
            # Use provided chunk_length if available, otherwise calculate from first device
            if chunk_length is None:
                chunk_length = len(action_chunk_dict[first_device])

            # Ensure chunk_length is valid
            if chunk_length is None or chunk_length <= 0:
                logger.error(f"AutoActionManager: Invalid chunk_length={chunk_length}, using index 0")
                return None

            # Calculate max_search_end to reserve space for direction consistency check
            # This ensures that after slicing from optimal_start_index, there are enough actions
            # for direction consistency processing
            max_search_end = chunk_length - 1 - direction_consistency_range
            if max_search_end < 0:
                logger.warning(
                    f"AutoActionManager: chunk_length={chunk_length} is too small for "
                    f"direction_consistency_range={direction_consistency_range}, setting max_search_end=0"
                )
                max_search_end = 0

            # If estimated_steps_elapsed >= chunk_length, delay is severe, search entire available chunk
            if estimated_steps_elapsed >= chunk_length:
                search_start = 0
                search_end = max_search_end
                logger.warning(
                    f"AutoActionManager: Severe delay detected (estimated_steps_elapsed={estimated_steps_elapsed} "
                    f">= chunk_length={chunk_length}), searching entire available chunk [0, {max_search_end}]"
                )
            else:
                # Calculate search range [estimated_steps_elapsed - tolerance, estimated_steps_elapsed + tolerance]
                search_start = max(0, estimated_steps_elapsed - search_tolerance)
                search_end = min(max_search_end, estimated_steps_elapsed + search_tolerance)

            if search_start > search_end:
                logger.warning(f"AutoActionManager: Invalid search range [{search_start}, {search_end}], using index 0")
                search_start = 0
                search_end = max_search_end

            logger.info(
                f"AutoActionManager: Searching optimal start index in range [{search_start}, {search_end}] "
                f"(estimated_steps_elapsed={estimated_steps_elapsed}, chunk_length={chunk_length})"
            )

            # For each device, find the closest action
            best_indices = []
            min_distances = []
            logger.info(f"Num of deivces: {len(device_names)}")
            for device_name in device_names:
                if device_name not in current_executing_action:
                    logger.warning(f"AutoActionManager: Device {device_name} not in current_executing_action, skipping")
                    continue

                current_action = current_executing_action[device_name]
                new_chunk = action_chunk_dict[device_name]

                # Search for closest action
                min_distance = float("inf")
                best_index = search_start

                for idx in range(search_start, search_end + 1):
                    candidate_action = new_chunk[idx]

                    left_dist = self._calc_joint_action_distance(current_action[:7], candidate_action[:7])
                    right_dist = self._calc_joint_action_distance(current_action[7:14], candidate_action[7:14])
                    distance = (left_dist + right_dist) / 2.0

                    if distance < min_distance:
                        min_distance = distance
                        best_index = idx

                best_indices.append(best_index)
                min_distances.append(min_distance)
                logger.info(
                    f"AutoActionManager: Device {device_name} - "
                    f"best_index={best_index}, min_distance={min_distance:.6f}, "
                    f"search_range=[{search_start}, {search_end}]"
                )

            # Average the best indices from all devices
            optimal_index = int(np.round(np.mean(best_indices)))
            avg_min_distance = np.mean(min_distances) if min_distances else 0.0
            if avg_min_distance >= 0.1:
                print("ok")
            logger.info(
                f"AutoActionManager: optimal_index={optimal_index} (from indices {best_indices}), "
                f"avg_min_distance={avg_min_distance:.6f}, "
                f"search_range=[{search_start}, {search_end}], "
                f"skipping {optimal_index} obsolete actions"
            )
            return optimal_index

        except Exception as e:
            logger.error(f"AutoActionManager: Error finding optimal start index: {e}, falling back to 0")
            return None
