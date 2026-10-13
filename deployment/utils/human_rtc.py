"""Helpers for constructing RTC prefixes from streamed human commands."""

import numpy as np


def build_human_rtc_prefix(
    latest_action: np.ndarray,
    previous_action: np.ndarray,
    sample_dt_s: float,
    chunk_length: int,
    control_fps: float,
    velocity_decay_steps: int,
    max_velocity: float,
) -> np.ndarray:
    """Extend the latest human command into a smooth, future-facing RTC prefix."""
    latest = np.asarray(latest_action, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous_action, dtype=np.float64).reshape(-1)
    if latest.shape != previous.shape:
        raise ValueError(f"Human action shapes differ: {latest.shape} != {previous.shape}")

    horizon = max(1, int(chunk_length))
    decay_steps = max(1, min(int(velocity_decay_steps), horizon - 1))
    velocity = (latest - previous) / max(float(sample_dt_s), 1e-6)
    velocity = np.clip(velocity, -abs(float(max_velocity)), abs(float(max_velocity)))

    prefix = np.empty((horizon, latest.size), dtype=np.float64)
    prefix[0] = latest
    control_dt = 1.0 / max(float(control_fps), 1e-6)
    for step in range(1, horizon):
        progress = min(step / decay_steps, 1.0)
        velocity_scale = 1.0 - (3.0 * progress**2 - 2.0 * progress**3)
        prefix[step] = prefix[step - 1] + velocity * velocity_scale * control_dt
    return prefix
