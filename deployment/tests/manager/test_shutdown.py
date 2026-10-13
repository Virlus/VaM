"""Exercise a shutdown signal while a real SHM view is exported, without devices."""

from multiprocessing.shared_memory import SharedMemory
from types import MethodType

import numpy as np
import pytest

from manager.base_device_manager import BaseDeviceManager


def test_signal_defers_cleanup_until_update_returns():
    manager = object.__new__(BaseDeviceManager)
    shm = SharedMemory(create=True, size=16)
    manager.devices = []
    manager.summary_shm = shm
    events = []

    def update(self):
        view = np.ndarray((2,), dtype=np.int64, buffer=shm.buf)
        self._signal_handler(2, None)
        assert not self.running
        view[0] = 42  # This must remain writable after the signal handler returns.
        events.append(int(view[0]))

    manager._update_summary_shm = MethodType(update, manager)
    try:
        manager.start()
        assert events == [42]
        assert shm.buf is None
        manager.stop()  # Repeated shutdown must be harmless.
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=shm.name)
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
