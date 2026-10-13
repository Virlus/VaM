import numpy as np
from utils.logger_config import logger
from typing import Dict, Optional


class BaseChunkManager:
    """
    The basic manager always carries out the action step-by-step
    in the latest action chunk immediately. The older chunks will be directly
    dropped whenever the new chunk arrives.
    """

    def __init__(self):
        self.current_action_chunk_dict = None
        self.current_step = 0  # Reset step counter for new chunk
        self.last_read_time = 0

    def is_empty(self):
        """Check if there is no chunk in the cache"""
        return self.current_action_chunk_dict is None

    def is_new_chunk(self, observation_timestamp: Optional[int] = None):
        """Return True if the current observation timestamp is newer than the previous one,"""
        return observation_timestamp is not None and observation_timestamp != self.last_read_time

    def put(self, action_chunk_dict: Dict[str, np.ndarray], observation_timestamp: Optional[int] = None) -> None:
        """Put the action chunk dict to local chunk cache"""
        # Deep copy to avoid shared memory reference issues
        self.current_action_chunk_dict = {k: v.copy() for k, v in action_chunk_dict.items()}
        self.current_step = 0
        self.last_read_time = observation_timestamp
        _device_names = list(action_chunk_dict.keys())

    def get(self, query_timestamp: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        """Get next action dict from cached current chunk dict."""
        try:
            if self.current_action_chunk_dict is None:
                return None

            # Get the chunk length from any device (they should all have the same chunk length)
            device_names = list(self.current_action_chunk_dict.keys())
            if not device_names:
                return None

            first_chunk = self.current_action_chunk_dict[device_names[0]]
            chunk_length = len(first_chunk)

            # Check if we have more actions in current chunk
            if self.current_step < chunk_length:
                # Extract action for current step from all devices
                action_dict = {}
                for device_name, chunk in self.current_action_chunk_dict.items():
                    action_dict[device_name] = chunk[self.current_step]

                self.current_step += 1
                return action_dict

            # No more actions in current chunk
            return None

        except Exception as e:
            logger.error(f"Error getting next action from current chunk: {e}")
            return None

    def reset(self) -> None:
        """Reset the chunk manager state - clear all cached actions."""
        self.current_action_chunk_dict = None
        self.current_step = 0
        self.last_read_time = 0
        logger.info("ChunkManager: Action cache cleared (reset called)")
