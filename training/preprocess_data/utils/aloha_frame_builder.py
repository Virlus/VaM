# ruff: noqa: FBT001, FBT003
# Preserve the legacy converter helper signatures.
"""Shared ALOHA frame construction; independent of recorded data readers and model code."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import logging
from typing import Any

import numpy as np

from preprocess_data.utils.lerobot_ops_common import fill_missing_frame_features
from preprocess_data.utils.sparse_reward_utils import align_terminal_sparse_reward_to_observation
from preprocess_data.utils.sparse_reward_utils import build_reward_chunk
from preprocess_data.utils.sparse_reward_utils import terminate_for_action_chunk_or_sparse_reward

logger = logging.getLogger(__name__)
IMAGE_KEYS = ("img_ego", "img_left", "img_right")
VALUE_PRED_TARGET_KEY = "value_prediction"
VALID_REWARD_RANGES = {"signed", "binary"}


def get_clipped_pose_window(
    pose_data: np.ndarray,
    start_idx: int,
    end_idx: int,
    window_size: int,
) -> np.ndarray:
    window = pose_data[start_idx : min(start_idx + window_size, end_idx + 1)]
    if len(window) == 0:
        raise ValueError(f"Cannot build clipped pose window from start_idx={start_idx}, end_idx={end_idx}")
    if len(window) < window_size:
        pad_count = window_size - len(window)
        window = np.concatenate([window, np.repeat(window[-1:], pad_count, axis=0)], axis=0)
    return np.ascontiguousarray(window.copy())


def build_episode_rewards(
    length: int,
    success: bool | None,
    *,
    reward_range: str = "signed",
    control_flag: np.ndarray | None = None,
) -> np.ndarray:
    if reward_range not in VALID_REWARD_RANGES:
        raise ValueError(f"reward_range must be one of {sorted(VALID_REWARD_RANGES)}, got {reward_range!r}")

    rewards = np.zeros(length, dtype=np.float32)
    if length == 0:
        return rewards

    if success:
        rewards[-1] = 1.0
    elif success is not None and not success and reward_range == "signed":
        rewards[-1] = -1.0

    if control_flag is not None and reward_range == "signed":
        control_flag = np.asarray(control_flag).reshape(-1)
        if len(control_flag) > 0 and not np.all(control_flag < 0):
            diff = np.diff(control_flag, prepend=control_flag[0])
            rewards[: len(diff)] += np.where(diff == 1, -1.0, 0.0).astype(np.float32)

    return rewards


def align_timestamp_to_index(target_timestamp: float, device_timestamps: np.ndarray, max_time_diff_s: float) -> int:
    if device_timestamps is None or len(device_timestamps) == 0:
        return -1
    min_idx = int(np.argmin(np.abs(device_timestamps - target_timestamp)))
    return min_idx if abs(device_timestamps[min_idx] - target_timestamp) <= max_time_diff_s else -1


def get_pose_window(
    pose_data: np.ndarray,
    pose_timestamps: np.ndarray,
    start_idx: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    end_idx = min(start_idx + window_size, len(pose_data))
    pose_window = pose_data[start_idx:end_idx]
    timestamp_window = pose_timestamps[start_idx:end_idx]

    if len(pose_window) < window_size:
        pad_count = window_size - len(pose_window)
        pose_window = np.concatenate([pose_window, np.repeat(pose_window[-1:], pad_count, axis=0)], axis=0)
        timestamp_window = np.concatenate(
            [timestamp_window, np.repeat(timestamp_window[-1:], pad_count, axis=0)],
            axis=0,
        )

    return pose_window, timestamp_window


def complete_action_frame(frame, actions, rewards, action_idx, end_idx, window_size, sparse_reward_idx):
    """Apply the legacy clipped future windows, sparse rewards and termination."""
    frame = frame.copy()
    frame["actions"] = get_clipped_pose_window(actions, action_idx, end_idx, window_size)
    frame["next_actions"] = get_clipped_pose_window(
        actions, min(action_idx + window_size, end_idx), end_idx, window_size
    )
    frame["reward"] = np.ascontiguousarray(build_reward_chunk(rewards, action_idx, end_idx, window_size))
    frame["terminate"] = terminate_for_action_chunk_or_sparse_reward(
        action_idx, end_idx, window_size, sparse_reward_idx
    )
    return frame


@dataclass
class AlohaFrameBuilder:
    """The legacy converter and live recorder share the same alignment implementation."""

    window_size: int = 100
    max_time_diff_ms: float = 50.0
    data_mode: str = "offline"
    features: dict = field(default_factory=dict)
    missing_feature_defaults: dict = field(default_factory=dict)
    hardware_latency_s: dict = field(default_factory=dict)

    def _apply_hardware_latency_compensation(
        self,
        timestamp_dict: dict[str, np.ndarray],
        feat: dict[str, str],
    ) -> dict[str, np.ndarray]:
        if not self.hardware_latency_s:
            return timestamp_dict
        compensated = {}
        for device_key, timestamps in timestamp_dict.items():
            latency_s = self.hardware_latency_s.get(device_key, 0.0)
            if latency_s == 0.0:
                for feat_key, feat_device_key in feat.items():
                    if feat_device_key == device_key:
                        latency_s = self.hardware_latency_s.get(feat_key, 0.0)
                        break
            compensated[device_key] = timestamps - latency_s
        min_timestamp = min(ts[0] for ts in compensated.values() if len(ts) > 0)
        if min_timestamp < 0:
            compensated = {key: timestamps - min_timestamp for key, timestamps in compensated.items()}
        return compensated

    def _required_device(self, feat: dict[str, str], data_dict: dict[str, np.ndarray], key: str) -> str | None:
        device_key = feat.get(key)
        if device_key is None or device_key not in data_dict:
            logger.warning(f"Required stream {key} not found")
            return None
        return device_key

    def _camera_alignment(
        self,
        feat: dict[str, str],
        data_dict: dict[str, np.ndarray],
        timestamp_dict: dict[str, np.ndarray],
    ) -> tuple[list[str], list[np.ndarray]]:
        camera_keys = [self._required_device(feat, data_dict, key) for key in IMAGE_KEYS]
        if any(key is None for key in camera_keys):
            raise ValueError("Missing one or more camera streams")
        sorted_cameras = sorted(
            enumerate([timestamp_dict[key] for key in camera_keys]),
            key=lambda item: len(item[1]),
        )
        sorted_keys = [camera_keys[index] for index, _timestamps in sorted_cameras]
        sorted_timestamps = [timestamps for _index, timestamps in sorted_cameras]
        return sorted_keys, sorted_timestamps

    def _build_base_frames(
        self,
        data_dict: dict[str, np.ndarray],
        timestamp_dict: dict[str, np.ndarray],
        feat: dict[str, str],
        task: str,
        success: bool | None,
        action_data: np.ndarray,
        action_timestamps: np.ndarray,
        control_flag: np.ndarray | None,
        value_predictions: np.ndarray | None,
    ) -> dict[int, dict[str, Any]]:
        max_time_diff_s = self.max_time_diff_ms / 1000.0
        follower_key = self._required_device(feat, data_dict, "follower_joint_pos")
        if follower_key is None:
            raise ValueError("Missing follower_joint_pos stream")

        follower_data = data_dict[follower_key]
        follower_timestamps = timestamp_dict[follower_key]
        camera_keys, camera_timestamps = self._camera_alignment(feat, data_dict, timestamp_dict)
        reference_key, secondary_key, third_key = camera_keys
        reference_timestamps, secondary_timestamps, third_timestamps = camera_timestamps
        monte_carlo_values = (
            np.arange(0, len(reference_timestamps), dtype=np.float32)[:, None] / len(reference_timestamps)
            if success
            else np.zeros((len(reference_timestamps), 1), dtype=np.float32)
        )
        repack_img_keys = {
            "base_image": feat["img_ego"],
            "left_wrist_image": feat["img_left"],
            "right_wrist_image": feat["img_right"],
        }
        frames_by_action_idx = {}

        for img_idx, ref_img_ts in enumerate(reference_timestamps):
            state_idx = align_timestamp_to_index(ref_img_ts, follower_timestamps, max_time_diff_s)
            action_idx = align_timestamp_to_index(ref_img_ts, action_timestamps, max_time_diff_s)
            secondary_img_idx = align_timestamp_to_index(ref_img_ts, secondary_timestamps, max_time_diff_s)
            third_img_idx = align_timestamp_to_index(ref_img_ts, third_timestamps, max_time_diff_s)
            if min(state_idx, action_idx, secondary_img_idx, third_img_idx) < 0:
                continue

            next_action_idx = min(action_idx + self.window_size, len(action_data) - 1)
            next_timestamp = action_timestamps[next_action_idx]
            next_state_idx = align_timestamp_to_index(next_timestamp, follower_timestamps, max_time_diff_s)
            if next_state_idx < 0:
                if self.data_mode == "offline":
                    next_state_idx = min(state_idx + self.window_size, len(follower_data) - 1)
                    next_timestamp = follower_timestamps[next_state_idx]
                else:
                    continue

            next_ref_img_idx = align_timestamp_to_index(next_timestamp, reference_timestamps, max_time_diff_s)
            next_secondary_img_idx = align_timestamp_to_index(next_timestamp, secondary_timestamps, max_time_diff_s)
            next_third_img_idx = align_timestamp_to_index(next_timestamp, third_timestamps, max_time_diff_s)
            if self.data_mode == "offline":
                if next_ref_img_idx < 0:
                    next_ref_img_idx = len(reference_timestamps) - 1
                if next_secondary_img_idx < 0:
                    next_secondary_img_idx = len(secondary_timestamps) - 1
                if next_third_img_idx < 0:
                    next_third_img_idx = len(third_timestamps) - 1
            elif min(next_ref_img_idx, next_secondary_img_idx, next_third_img_idx) < 0:
                continue

            # Keep views into data_dict: it remains alive until save_episode() drains the image writer.
            curr_img_dict = {
                reference_key: data_dict[reference_key][img_idx],
                secondary_key: data_dict[secondary_key][secondary_img_idx],
                third_key: data_dict[third_key][third_img_idx],
            }
            next_img_dict = {
                reference_key: data_dict[reference_key][next_ref_img_idx],
                secondary_key: data_dict[secondary_key][next_secondary_img_idx],
                third_key: data_dict[third_key][next_third_img_idx],
            }
            action_window, _ = get_pose_window(action_data, action_timestamps, action_idx, self.window_size)
            next_action_window, _ = get_pose_window(action_data, action_timestamps, next_action_idx, self.window_size)

            frame_dict = {key: curr_img_dict[device_key] for key, device_key in repack_img_keys.items()}
            frame_dict.update({"next_" + key: next_img_dict[device_key] for key, device_key in repack_img_keys.items()})
            frame_dict.update(
                {
                    "state": follower_data[state_idx].copy(),
                    "actions": np.ascontiguousarray(action_window.copy()),
                    "task": task,
                    "value": monte_carlo_values[img_idx],
                    "success": np.array(
                        [success if success is not None else 1.0],
                        dtype=np.float32,
                    ),
                    "next_state": follower_data[next_state_idx].copy(),
                    "next_actions": np.ascontiguousarray(next_action_window.copy()),
                    "data_source": np.array(
                        [1.0 if self.data_mode == "online" or success is None else 0.0],
                        dtype=np.float32,
                    ),
                }
            )
            if "control_flag" in self.features and control_flag is not None:
                frame_dict["control_flag"] = control_flag[action_idx : action_idx + 1].astype(np.float32)
            if VALUE_PRED_TARGET_KEY in self.features and value_predictions is not None:
                frame_dict[VALUE_PRED_TARGET_KEY] = value_predictions[action_idx].astype(np.float32)
            fill_missing_frame_features(frame_dict, self.features, self.missing_feature_defaults)
            frames_by_action_idx[action_idx] = frame_dict

        return frames_by_action_idx

    def build_offline_frames(self, data, timestamps, feat, task):
        """Build every aligned offline frame in source-action order."""
        return list(self.build_offline_frames_by_action_index(data, timestamps, feat, task).values())

    def build_offline_frames_by_action_index(self, data, timestamps, feat, task):
        """Build a complete successful offline episode from absolute-second streams.

        Images may be read-only memmaps. Returned views remain valid until both
        LeRobot image writers have drained. Numeric input is float32, as in recorded data.
        Preserve source-action indices for legacy stride selection, not row striding.
        """
        if self.data_mode != "offline":
            raise ValueError("Live collect only supports offline construction")
        for key in feat.values():
            if key not in data or key not in timestamps or not len(timestamps[key]):
                raise ValueError(f"Empty or missing stream: {key}")
            if len(data[key]) != len(timestamps[key]):
                raise ValueError(f"Data/timestamp length mismatch: {key}")
            if not np.all(np.isfinite(timestamps[key])) or np.any(np.diff(timestamps[key]) < 0):
                raise ValueError(f"Non-finite or decreasing timestamps: {key}")
        origin = min(ts[0] for ts in timestamps.values())
        timestamps = {key: ts - origin for key, ts in timestamps.items()}
        timestamps = self._apply_hardware_latency_compensation(timestamps, feat)
        action_key = feat["leader_joint_pos"]
        actions, action_ts = data[action_key], timestamps[action_key]
        frames = self._build_base_frames(data, timestamps, feat, task, True, actions, action_ts, None, None)
        if not frames:
            raise ValueError("No valid aligned observations")
        indices = sorted(frames)
        end = len(actions) - 1
        rewards = build_episode_rewards(len(actions), True, reward_range="binary")
        rewards, sparse_idx = align_terminal_sparse_reward_to_observation(rewards, indices, end, 1.0)
        return {
            idx: complete_action_frame(frames[idx], actions, rewards, idx, end, self.window_size, sparse_idx)
            for idx in indices
        }
