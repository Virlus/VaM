#!/usr/bin/env python3
"""
Base policy interface.

Defines the minimal interface for a policy used by the Policy Connector.

Author: Jun Lv
"""

from typing import Dict, Any, Union, List
import numpy as np


ObservationType = Union[np.ndarray, Dict[str, np.ndarray]]


class BasePolicy:
    """
    Base policy interface.

    Policies should implement a lightweight constructor and a predict
    method that maps observation arrays to action arrays.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.initialized: bool = False
        self.config: Dict[str, Any] = kwargs

    def load(self) -> None:
        """
        Optional load hook. For simple numpy policies this can be a no-op.
        For learned policies, load weights or backends here.
        """
        self.initialized = True

    def _to_vector(self, observation: ObservationType) -> np.ndarray:
        """
        Convert observation to a 1D numpy vector.
        - If dict: flatten each value (sorted by key for determinism) and concatenate
        - If ndarray: flatten
        """
        if isinstance(observation, dict):
            parts = []
            for key in sorted(observation.keys()):
                arr = observation[key]
                if arr is None:
                    continue
                parts.append(np.asarray(arr).reshape(-1))
            if not parts:
                return np.array([], dtype=np.float64)
            return np.concatenate(parts, axis=0).astype(np.float64)
        else:
            return np.asarray(observation).reshape(-1).astype(np.float64)

    def predict(
        self, observation: ObservationType, action_configs: List[Dict], chunk_length: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Predict action chunking from observation.
        Args:
            observation: Either a flattened vector or a dict of device arrays
            action_configs: List of action configs with device info [{'device_name': str, 'action_dim': int}, ...]
            chunk_length: Number of future steps to predict
        Returns:
            Dict mapping device names to action chunks: {'device_name': np.ndarray(chunk_length, action_dim)}
        """
        raise NotImplementedError("predict must be implemented by subclasses")
