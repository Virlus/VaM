from __future__ import annotations

import numpy as np


def require_finite_next_obs_value(next_obs_value: float, frame_idx: int, *, method: str) -> None:
    if np.isfinite(next_obs_value):
        return
    raise ValueError(
        f"{method} advantage requires next-observation value predictions, "
        f"but frame {frame_idx} has no finite next_obs value."
    )


def calculate_episode_horizon_advantages(
    *,
    all_pred_values: np.ndarray,
    all_next_obs_values: np.ndarray,
    all_data_sources: np.ndarray,
    all_terminates: np.ndarray,
    ep_starts: list[int],
    ep_ends: list[int],
    advantage_obs_horizon: int,
) -> np.ndarray:
    if advantage_obs_horizon < 1:
        raise ValueError(f"advantage_obs_horizon must be >= 1 when set, got {advantage_obs_horizon}")

    all_advantages = np.zeros_like(all_pred_values, dtype=np.float64)

    for start, end in zip(ep_starts, ep_ends, strict=True):
        ep_pred_values = all_pred_values[start:end]
        ep_next_obs_values = all_next_obs_values[start:end]
        ep_data_sources = all_data_sources[start:end]
        ep_terminates = all_terminates[start:end]

        for idx, data_source in enumerate(ep_data_sources):
            if data_source < 0.5:
                all_advantages[start + idx] = 1.0
                continue

            next_obs_value = ep_next_obs_values[idx]
            require_finite_next_obs_value(next_obs_value, start + idx, method="value_avg")

            if ep_terminates[idx] >= 0.5:
                values_to_average = [0.0]
            else:
                future_start = idx + 1
                future_end = min(len(ep_pred_values), idx + 1 + advantage_obs_horizon)
                values_to_average = [next_obs_value]
                values_to_average.extend(ep_pred_values[future_start:future_end].tolist())

            if not np.all(np.isfinite(values_to_average)):
                raise ValueError(
                    "advantage_obs_horizon requires finite future value predictions, "
                    f"but frame {start + idx} has non-finite values in its averaging window."
                )
            if not np.isfinite(ep_pred_values[idx]):
                raise ValueError(
                    "advantage_obs_horizon requires finite current-observation value predictions, "
                    f"but frame {start + idx} has value {ep_pred_values[idx]}."
                )

            all_advantages[start + idx] = float(np.mean(values_to_average) - ep_pred_values[idx])

    return all_advantages


def calculate_episode_gae_advantages(
    *,
    all_pred_values: np.ndarray,
    all_next_obs_values: np.ndarray,
    all_data_sources: np.ndarray,
    all_reward_sums: np.ndarray,
    all_terminates: np.ndarray,
    ep_starts: list[int],
    ep_ends: list[int],
    advantage_obs_horizon: int,
    gamma: float,
    gae_lambda: float,
    action_horizon: int,
    action_step_frames: int,
    all_action_indices: np.ndarray | None = None,
) -> np.ndarray:
    if advantage_obs_horizon < 1:
        raise ValueError(f"advantage_obs_horizon must be >= 1 when set, got {advantage_obs_horizon}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
    if action_step_frames < 1:
        raise ValueError(f"action_step_frames must be >= 1, got {action_step_frames}")

    # Retained rows need not be uniformly spaced (camera alignment and terminal retention).
    # Bootstrap values are supplied at the actual chunk endpoint by the converter.
    # For multi-step GAE, interpolate the continuation on the retained raw-action grid.
    if all_action_indices is not None:
        positions = np.asarray(all_action_indices, dtype=np.float64)
        if positions.shape != all_pred_values.shape or not np.all(np.isfinite(positions)):
            raise ValueError("all_action_indices must be finite and match all_pred_values")
        result = np.zeros_like(all_pred_values, dtype=np.float64)
        discount = gamma**action_horizon
        for start, end in zip(ep_starts, ep_ends, strict=True):
            x = positions[start:end]
            if not len(x):
                continue
            if np.any(np.diff(x) <= 0):
                raise ValueError("source action indices must be strictly increasing within each episode")
            online = all_data_sources[start:end] >= 0.5
            bootstrap = all_terminates[start:end] < 0.5
            values = all_next_obs_values[start:end]
            if not np.all(np.isfinite(values[online])):
                raise ValueError("GAE requires finite next-observation value predictions")
            delta = all_reward_sums[start:end] + discount * bootstrap * values - all_pred_values[start:end]
            delta = np.where(online, delta, 1.0)
            gae = delta.copy()
            for _ in range(advantage_obs_horizon - 1):
                continuation = np.interp(x + action_horizon, x, gae, right=0.0)
                gae = np.where(online, delta + discount * gae_lambda * bootstrap * continuation, 1.0)
            result[start:end] = gae
        return result

    all_advantages = np.zeros_like(all_pred_values, dtype=np.float64)
    transition_discount = gamma**action_horizon
    gae_discount = transition_discount * gae_lambda

    for start, end in zip(ep_starts, ep_ends, strict=True):
        ep_pred_values = all_pred_values[start:end]
        ep_next_obs_values = all_next_obs_values[start:end]
        ep_data_sources = all_data_sources[start:end]
        ep_reward_sums = all_reward_sums[start:end]
        ep_terminates = all_terminates[start:end]

        for idx, data_source in enumerate(ep_data_sources):
            if data_source < 0.5:
                all_advantages[start + idx] = 1.0
                continue

            gae = 0.0
            weight = 1.0
            step_idx = idx
            for _ in range(advantage_obs_horizon):
                if step_idx >= len(ep_pred_values):
                    break
                next_obs_value = ep_next_obs_values[step_idx]
                require_finite_next_obs_value(next_obs_value, start + step_idx, method="GAE")

                bootstrap_mask = 1.0 - float(ep_terminates[step_idx] >= 0.5)
                delta = (
                    ep_reward_sums[step_idx]
                    + transition_discount * bootstrap_mask * next_obs_value
                    - ep_pred_values[step_idx]
                )
                gae += weight * delta
                if bootstrap_mask == 0.0:
                    break
                weight *= gae_discount

                next_step_idx = step_idx + action_step_frames
                if next_step_idx >= len(ep_pred_values):
                    break
                step_idx = next_step_idx

            all_advantages[start + idx] = gae

    return all_advantages
