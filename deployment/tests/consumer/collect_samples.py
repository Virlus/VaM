"""Deterministic independent camera and joint streams for collect tests."""

import numpy as np


def make_streams(config, count=240):
    data, timestamps = {}, {}
    for i, (role, device) in enumerate(config["inputs"].items()):
        if role.startswith("img"):
            n = max(1, int(count * 30 / 100))
            shape = config["features"]["base_image"]["shape"]
            values = (np.arange(n) * 3 + i * 11) % 256
            data[device] = np.broadcast_to(values[:, None, None, None], (n, *shape)).astype(np.uint8)
            timestamps[device] = np.arange(n) / 30 + 10 + 0.033
        else:
            data[device] = (np.arange(count * 14).reshape(count, 14) / 1000 + i).astype(np.float32)
            timestamps[device] = np.arange(count) / 100 + 10
    return data, timestamps
