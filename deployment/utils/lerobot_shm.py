"""Incremental readers for existing device/control SHM rings (no new producer)."""

from __future__ import annotations

import numpy as np

from utils.shm_utils import (
    BUFFER_HEADER_SIZE,
    DEVICE_HEADER_SIZE,
    FRAME_HEADER_SIZE,
    connect_to_shared_memory,
    get_dtype,
    unpack_buffer_header,
    unpack_device_header,
)


class RingStream:
    def __init__(self, name, *, channel=None, index_is_latest=False):
        self.name = name
        self.index_is_latest = index_is_latest
        self.channel = channel or connect_to_shared_memory(name, read_only=True)
        self.device = unpack_device_header(bytes(self.channel.buf[:DEVICE_HEADER_SIZE]))
        self.frame_size = self.device["frame_size"]
        self.dtype = np.dtype(get_dtype(self.device["data_dtype"]))
        self.shape = tuple(self.device["shape"])
        self.reset(0)

    def reset(self, minimum_timestamp):
        self.minimum_timestamp = minimum_timestamp
        self.cursor = None
        self.last_slot = None
        self.last_slot_timestamp = None
        self.last_timestamp = 0

    def _timestamp(self, index):
        offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + index * self.frame_size
        return int.from_bytes(bytes(self.channel.buf[offset : offset + FRAME_HEADER_SIZE]), "little")

    def read_new(self):
        buf = self.channel.buf
        header = unpack_buffer_header(bytes(buf[DEVICE_HEADER_SIZE : DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE]))
        count, capacity, write = (header[k] for k in ("current_frames_count", "buffer_size", "write_index"))
        if not count:
            return []
        if capacity < 2:
            raise ValueError("Recording requires SHM capacity >= 2")
        if self.index_is_latest:
            # I2RT's flag producer commits the latest slot; all other producers
            # commit the next write slot. Normalize without changing hardware code.
            write = (write + 1) % capacity
        if self.cursor is None:
            # Exclude a full ring's next write slot: the producer may already be
            # replacing it before committing the next buffer header.
            available = min(count, capacity - 1)
            indices = [(write - available + i) % capacity for i in range(available)]
        else:
            if self._timestamp(self.last_slot) != self.last_slot_timestamp:
                raise RuntimeError(f"SHM overrun/reset during recording: {self.name}")
            indices = [(self.cursor + i) % capacity for i in range((write - self.cursor) % capacity)]
        result = []
        for index in indices:
            before = self._timestamp(index)
            if before >= self.minimum_timestamp and before > self.last_timestamp:
                offset = DEVICE_HEADER_SIZE + BUFFER_HEADER_SIZE + index * self.frame_size
                payload = bytes(buf[offset + FRAME_HEADER_SIZE : offset + self.frame_size])
                if before != self._timestamp(index):
                    raise RuntimeError(f"SHM frame overwritten during read: {self.name}")
                array = np.frombuffer(payload, dtype=self.dtype).reshape(self.shape).copy()
                result.append((before, array))
                self.last_timestamp = before
            elif before and before < self.last_timestamp:
                raise RuntimeError(f"SHM timestamps regressed: {self.name}")
            self.last_slot, self.last_slot_timestamp = index, before
        self.cursor = write
        return result

    def close(self):
        self.channel.close()
