# Utilities for observation-aligned terminal sparse rewards during dataset conversion.
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def downsampled_action_indices_with_terminal(
    available_action_indices: Sequence[int],
    *,
    end_idx: int,
    frame_stride: int,
) -> list[int]:
    """Downsample aligned action indices while retaining the segment's final observation."""
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    in_segment = sorted({int(idx) for idx in available_action_indices if int(idx) <= end_idx})
    if not in_segment:
        return []
    selected = [idx for idx in in_segment if idx % frame_stride == 0]
    terminal_observation_idx = in_segment[-1]
    if not selected or selected[-1] != terminal_observation_idx:
        selected.append(terminal_observation_idx)
    return selected


def reward_horizon_covers_action(action_idx: int, reward_idx: int, reward_horizon: int) -> bool:
    """Return whether a frame starting at action_idx can expose reward_idx."""
    if reward_horizon <= 0:
        raise ValueError(f"reward_horizon must be positive, got {reward_horizon}")
    return action_idx <= reward_idx < action_idx + reward_horizon


def select_observation_aligned_sparse_reward_index(
    frame_action_indices: Sequence[int],
    terminal_idx: int,
) -> int | None:
    """Return the saved observation-aligned action index closest to the segment terminal."""
    eligible_indices = sorted({int(index) for index in frame_action_indices if int(index) <= terminal_idx})
    if not eligible_indices:
        return None
    return eligible_indices[-1]


def align_terminal_sparse_reward_to_observation(
    rewards: np.ndarray,
    frame_action_indices: Sequence[int],
    terminal_idx: int,
    terminal_reward: float,
) -> tuple[np.ndarray, int | None]:
    """Place terminal sparse reward at the segment's last saved observation-aligned action index."""
    if terminal_idx < 0 or terminal_idx >= len(rewards):
        raise IndexError(f"terminal_idx={terminal_idx} is outside rewards length {len(rewards)}")

    adjusted = np.asarray(rewards, dtype=np.float32).copy()
    sparse_reward_idx = select_observation_aligned_sparse_reward_index(frame_action_indices, terminal_idx)
    if sparse_reward_idx is None:
        return adjusted, None

    adjusted[terminal_idx] = np.float32(0.0)
    adjusted[sparse_reward_idx] = np.float32(terminal_reward)
    return adjusted, sparse_reward_idx


def sparse_reward_applies_to_chunk(
    action_idx: int,
    sparse_reward_idx: int | None,
    reward_horizon: int,
) -> bool:
    """Return whether a frame's reward chunk contains the sparse reward event."""
    return sparse_reward_idx is not None and reward_horizon_covers_action(action_idx, sparse_reward_idx, reward_horizon)


def terminate_for_action_chunk_or_sparse_reward(
    action_idx: int,
    end_idx: int,
    action_horizon: int,
    sparse_reward_idx: int | None,
    reward_horizon: int | None = None,
) -> np.ndarray:
    """Terminate when the action chunk exceeds the segment end or includes the sparse reward event."""
    reward_horizon = action_horizon if reward_horizon is None else reward_horizon
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if reward_horizon <= 0:
        raise ValueError(f"reward_horizon must be positive, got {reward_horizon}")
    terminates = action_idx + action_horizon > end_idx or sparse_reward_applies_to_chunk(
        action_idx,
        sparse_reward_idx,
        reward_horizon,
    )
    return np.array([float(terminates)], dtype=np.float32)


def build_reward_chunk(rewards: np.ndarray, action_idx: int, end_idx: int, reward_horizon: int) -> np.ndarray:
    """Build a fixed-size reward chunk for an action-aligned frame."""
    if reward_horizon <= 0:
        raise ValueError(f"reward_horizon must be positive, got {reward_horizon}")
    reward_chunk = np.zeros(reward_horizon, dtype=np.float32)
    reward_end_idx = min(action_idx + reward_horizon, end_idx + 1)
    if reward_end_idx > action_idx:
        reward_chunk[: reward_end_idx - action_idx] = rewards[action_idx:reward_end_idx]
    return np.ascontiguousarray(reward_chunk)
