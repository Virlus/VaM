"""
Latency Tracker for inference latency tracking and analysis.

This module provides a sliding-window latency tracker that records
inference latencies and provides max/percentile queries for dynamic
RTC (Real-Time Control) configuration.

Author: Linzhi Wu
"""

from collections import deque
import math

import numpy as np


class LatencyTracker:
    """Tracks recent latencies and provides max/percentile queries.

    Args:
        maxlen (int | None): Optional sliding window size. If provided, only the
            most recent ``maxlen`` latencies are kept. If ``None``, keeps all.
    """

    def __init__(self, maxlen: int = 100) -> None:
        self._values: deque = deque(maxlen=maxlen)
        self.max_latency: float = 0.0

    def reset(self) -> None:
        """Clear all recorded latencies."""
        self._values.clear()
        self.max_latency = 0.0

    def add(self, latency: float) -> None:
        """Add a latency sample (seconds)."""
        # Ensure numeric and non-negative
        val = float(latency)
        if val < 0:
            return
        self._values.append(val)
        self.max_latency = max(self.max_latency, val)

    def __len__(self) -> int:
        return len(self._values)

    def max(self) -> float:
        """Return the maximum latency or 0.0 if empty."""
        return self.max_latency

    def percentile(self, q: float) -> float:
        """Return the q-quantile (q in [0,1]) of recorded latencies or 0.0 if empty."""
        if not self._values:
            return 0.0
        q = float(q)
        if q <= 0.0:
            return min(self._values)
        if q >= 1.0:
            return self.max_latency
        vals = np.array(list(self._values), dtype=np.float32)
        return float(np.quantile(vals, q))

    def p95(self) -> float:
        """Return the 95th percentile latency or 0.0 if empty."""
        return self.percentile(0.95)

    def mean(self) -> float:
        """Return the mean latency or 0.0 if empty."""
        if not self._values:
            return 0.0
        return float(np.mean(list(self._values)))

    def compute_inference_delay_steps(
        self, execution_fps: float, use_percentile: float = 0.95, default_delay_steps: int = 10
    ) -> int:
        """Compute the inference delay in discrete time steps.

        Based on the tracked latencies and the execution FPS, compute how many
        time steps the inference takes on average (using the specified percentile).

        Args:
            execution_fps: The execution frequency in Hz (e.g., 25 Hz).
            use_percentile: Which percentile of latency to use (default: 0.95).
            default_delay_steps: Default delay steps to return if no data available.

        Returns:
            The number of time steps that the inference occupies.
        """
        if not self._values or execution_fps <= 0:
            return default_delay_steps

        # Get the representative latency (e.g., 95th percentile)
        representative_latency = self.percentile(use_percentile)

        # Convert latency (seconds) to time steps: delay_steps = ceil(latency * fps)
        delay_steps = int(math.ceil(representative_latency * execution_fps))

        # Ensure at least 1 step delay
        return max(1, delay_steps)
