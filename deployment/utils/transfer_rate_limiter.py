"""Interruptible payload pacing shared across files in a transfer direction."""

import math
import threading
import time


class TransferRateLimiter:
    """Reserve bytes before I/O without accumulating idle-time burst credit."""

    def __init__(self, bytes_per_second: float, stop_event: threading.Event):
        if not math.isfinite(bytes_per_second) or bytes_per_second <= 0:
            raise ValueError("bytes_per_second must be finite and positive")
        self.bytes_per_second = bytes_per_second
        self._stop_event = stop_event
        self._lock = threading.Lock()
        self._next_time = 0.0

    def chunk_size(self, requested: int) -> int:
        """Bound each request to 256 KiB or 50 ms of bandwidth."""
        if requested <= 0:
            raise ValueError("Transfer chunk size must be positive")
        return max(1, min(requested, 256 * 1024, int(self.bytes_per_second * 0.05)))

    def acquire(self, size: int) -> None:
        if size < 0:
            raise ValueError("Transfer size must not be negative")
        with self._lock:
            now = time.monotonic()
            self._next_time = max(now, self._next_time) + size / self.bytes_per_second
            delay = self._next_time - now
        if self._stop_event.wait(delay):
            raise RuntimeError("Transfer interrupted because DataUploadManager is stopping")
