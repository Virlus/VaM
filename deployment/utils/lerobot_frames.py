"""Array-only implementation of the deployed recorded data converter's online recipe.

No recorded data files or converter entry points are used. Keep the intentionally legacy
prefix/reward/nearest-neighbour behaviour: storage migration must not fix labels.
The frozen cloud oracle in tests verifies these rules on identical source arrays.
"""

from __future__ import annotations

import sys
from collections import OrderedDict

import numpy as np

from utils.control_timeline import ExcludedTimeline
from utils.lerobot_images import legacy_jpeg_rgb
from utils.lerobot_recording import TRAINING_ROOT


def legacy_helpers():
    if str(TRAINING_ROOT) not in sys.path:
        sys.path.append(str(TRAINING_ROOT))
    from preprocess_data.utils import sparse_reward_utils
    from preprocess_data.utils.advantage_utils import calculate_episode_gae_advantages

    return sparse_reward_utils, calculate_episode_gae_advantages


def align(times, timestamp, tolerance):
    """Match the converter's float64-seconds argmin, including first-row ties."""
    if not len(times):
        return -1
    index = int(np.argmin(np.abs(times - timestamp)))
    return index if abs(times[index] - timestamp) <= tolerance else -1


def clipped_window(actions, index, end, size):
    values = actions[index : min(index + size, end + 1)]
    if len(values) < size:
        values = np.concatenate([values, np.repeat(values[-1:], size - len(values), axis=0)], axis=0)
    return np.ascontiguousarray(values.copy())


class OnlineFrames:
    def __init__(self, data, timestamps, config, policy, result, start_ns, end_ns):
        from utils.lerobot_direct import CAMERAS, STREAMS, validate_config

        self.cfg = validate_config(config)
        self.policy, self.result = dict(policy), result
        if result not in {"success", "failure"}:
            raise ValueError("An explicit success/failure result is required")
        for name in ("policy_id", "checkpoint_index"):
            if type(policy.get(name)) is not int or not 0 <= policy[name] <= 2**24:
                raise ValueError(f"Invalid exact policy identity: {name}")
        self.data, self.camera_names = data, CAMERAS
        self.times, self.rows, self.source_ns = {}, {}, {}
        self.audit = {"action": [], "value": []}
        self.image_cache = OrderedDict()
        self.sampling = {}
        self.tolerance = self.cfg["alignment_tolerance_ms"] / 1000
        # The original saver compacts all streams BEFORE converter latency correction.
        excluded = self.cfg.get("excluded_control_flag_values", [])
        intervals, begin = [], None
        last_timestamp = max(int(ts[-1]) for ts in timestamps.values() if len(ts))
        for ts, flag in zip(timestamps["control_flag"], np.asarray(data["control_flag"]).reshape(-1)):
            is_excluded = any(np.isclose(flag, value, rtol=0, atol=1e-6) for value in excluded)
            if is_excluded and begin is None:
                begin = int(ts)
            elif not is_excluded and begin is not None:
                intervals.append((begin, int(ts)))
                begin = None
        if begin is not None:
            intervals.append((begin, last_timestamp + 1))
        timeline = ExcludedTimeline(intervals)
        self.compacted_intervals = timeline.intervals
        for name in STREAMS:
            ts = np.asarray(timestamps[name], dtype=np.int64)
            if len(ts) != len(data[name]) or np.any(np.diff(ts) <= 0):
                raise ValueError(f"Unordered or incomplete stream: {name}")
            if not len(ts) and name not in {"policy_action", "human_action"}:
                raise ValueError(f"Missing stream: {name}")
            latency = config.get("latency_ms", {}).get(name, 0)
            mask = (ts <= end_ns) & np.array([not timeline.contains(int(t)) for t in ts], dtype=bool)
            compacted = np.array([timeline.shift(int(t)) for t in ts[mask]], dtype=np.int64)
            self.rows[name] = np.flatnonzero(mask)
            self.source_ns[name] = ts[mask] - int(latency * 1e6)
            self.times[name] = compacted.astype(np.float64) / 1e9
            self.sampling[name] = {"recorded": len(ts), "retained": int(mask.sum())}

        # Match recorded data decode order: absolute ns -> float seconds -> common origin,
        # then hardware latency, then shift a negative origin back to zero.
        origin = min(ts[0] for ts in self.times.values() if len(ts))
        self.times = {
            name: ts - origin - self.cfg.get("latency_ms", {}).get(name, 0) / 1000 for name, ts in self.times.items()
        }
        origin = min(ts[0] for ts in self.times.values() if len(ts))
        if origin < 0:
            self.times = {name: ts - origin for name, ts in self.times.items()}
        flag_times = self.times["control_flag"]
        flags = np.asarray(data["control_flag"])[self.rows["control_flag"]].reshape(-1)
        if not len(flags) or not np.isin(flags, [0, 0.5, 1]).all():
            raise ValueError("Missing or invalid control flags")
        # Audit only: legacy filtering removes commands, not observations, in .5 intervals.
        self.intervals = []
        for index, flag in enumerate(flags):
            if flag == 0.5:
                a = int(self.source_ns["control_flag"][index])
                b = int(self.source_ns["control_flag"][index + 1]) if index + 1 < len(flags) else end_ns + 1
                if self.intervals and self.intervals[-1][1] == a:
                    self.intervals[-1][1] = b
                else:
                    self.intervals.append([a, b])
        pieces = []
        for name, selected in (("human_action", 1), ("policy_action", 0)):
            for index, ts in enumerate(self.times[name]):
                # The old merge chooses teleoperation first on equal timestamps,
                # and uses the first flag even before its first timestamp.
                flag_index = max(0, int(np.searchsorted(flag_times, ts, side="right")) - 1)
                if flags[flag_index] == selected:
                    pieces.append(
                        (
                            ts,
                            name,
                            int(self.rows[name][index]),
                            selected,
                            int(self.source_ns[name][index]),
                        )
                    )
        pieces.sort(key=lambda item: item[0])
        if not pieces:
            raise ValueError("No selected policy/human commands in this episode")
        self.action_seconds = np.asarray([p[0] for p in pieces], dtype=np.float64)
        self.action_times = np.asarray([p[4] for p in pieces], dtype=np.int64)
        self.actions = np.asarray([data[p[1]][p[2]] for p in pieces], dtype=np.float32)
        self.flags = [p[3] for p in pieces]
        self.action_provenance = [(p[4], p[1], p[2], p[3]) for p in pieces]
        if self.actions.shape != (len(pieces), 14) or not np.isfinite(self.actions).all():
            raise ValueError("Commands must be finite 14D targets")
        self.end = len(pieces) - 1
        self.prefixes = [
            {"end_index": i, "terminal_reason": "policy_to_human"}
            for i in range(1, len(self.flags))
            if self.flags[i] - self.flags[i - 1] == 1
        ]
        if not self.prefixes or self.prefixes[-1]["end_index"] != self.end:
            self.prefixes.append({"end_index": self.end, "terminal_reason": "episode_end"})
        self.reference = min(CAMERAS.values(), key=lambda key: len(self.times[key]))
        self.value_rows = []
        for ts in self.action_seconds:
            index = align(self.times["value_prediction"], ts, self.tolerance)
            if index < 0:
                # Preserve the old value fallback, even beyond alignment tolerance.
                index = int(np.argmin(np.abs(self.times["value_prediction"] - ts)))
            self.value_rows.append(index)
        self.observations = {}
        for reference_index, ts in enumerate(self.times[self.reference]):
            index = align(self.action_seconds, ts, self.tolerance)
            if index < 0:
                continue
            current = {name: align(self.times[name], ts, self.tolerance) for name in (*CAMERAS.values(), "state")}
            current[self.reference] = reference_index
            next_index = min(index + self.cfg["window_size"], self.end)
            after = {
                name: align(self.times[name], self.action_seconds[next_index], self.tolerance)
                for name in (*CAMERAS.values(), "state")
            }
            if min(*current.values(), *after.values()) < 0:
                continue  # Legacy drops this observation, not the whole episode.
            self.observations[index] = (reference_index, current, after)
        if not self.observations:
            raise ValueError("No valid camera observations after legacy alignment")
        self.prefixes = [
            prefix for prefix in self.prefixes if any(index <= prefix["end_index"] for index in self.observations)
        ]

    def _image(self, name, row):
        key = (name, row)
        if key in self.image_cache:
            self.image_cache.move_to_end(key)
            return self.image_cache[key]
        image = np.asarray(self.data[name][row])
        # Preserve the old saver JPEG-95 and converter decode pixels in memory.
        # There is no recorded data file, SDK, or converter process in the direct path.
        self.image_cache[key] = legacy_jpeg_rgb(image, self.cfg["image_color_order"])
        if len(self.image_cache) > 16:
            self.image_cache.popitem(last=False)
        return self.image_cache[key]

    def _read(self, name, index):
        row = int(self.rows[name][index])
        if name in self.camera_names.values():
            value = self._image(name, row)
        else:
            value = np.asarray(self.data[name][row]).astype(np.float32, copy=True)
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite source: {name}")
        return value, {"row": row, "timestamp_ns": int(self.source_ns[name][index])}

    def _value(self, index):
        return self._read("value_prediction", self.value_rows[index])

    def _frame(self, index, end, reward, sparse_index):
        rules, _ = legacy_helpers()
        reference_index, current, after = self.observations[index]
        frame, sources = {}, {}
        for prefix, matches in (("", current), ("next_", after)):
            for output, name in {**self.camera_names, "state": "state"}.items():
                frame[prefix + output], sources[prefix + output] = self._read(name, matches[name])
        next_index = min(index + self.cfg["window_size"], end)
        frame["value_prediction"], sources["value_prediction"] = self._value(index)
        frame["next_value_prediction"], sources["next_value_prediction"] = self._value(next_index)
        success = self.result == "success"
        progress = np.float32(reference_index) / len(self.times[self.reference]) if success else 0
        for name, value in {
            "success": success,
            "value": progress,
            "data_source": 1,
            "control_flag": self.flags[index],
            "policy_id": self.policy["policy_id"],
            "checkpoint_index": self.policy["checkpoint_index"],
        }.items():
            frame[name] = np.asarray([value], dtype=np.float32)
        size = self.cfg["window_size"]
        frame.update(
            actions=clipped_window(self.actions, index, end, size),
            next_actions=clipped_window(self.actions, next_index, end, size),
            reward=rules.build_reward_chunk(reward, index, end, size),
            terminate=rules.terminate_for_action_chunk_or_sparse_reward(index, end, size, sparse_index),
            source_action_index=np.asarray([index], dtype=np.int64),
            task=self.cfg["task_name"],
        )
        return frame, sources

    def frames(self, role, prefix_end, episode_index=0):
        rules, gae = legacy_helpers()
        indices = rules.downsampled_action_indices_with_terminal(
            sorted(self.observations),
            end_idx=prefix_end,
            frame_stride=self.cfg["value_stride"] if role == "value" else 1,
        )
        if not indices:
            return
        rewards = np.zeros(len(self.actions), dtype=np.float32)
        if self.result == "success":
            rewards[-1] = 1
        if role == "value":
            rewards[: max(0, prefix_end - 1)] = 0
        rewards, sparse_index = rules.align_terminal_sparse_reward_to_observation(
            rewards, indices, prefix_end, float(rewards[prefix_end])
        )
        raw = None
        if role == "action":
            # The deployed action recipe has no source_action_index/next_value column:
            # preserve its row-offset bootstrap rather than introducing a new endpoint rule.
            values = np.asarray([self._value(i)[0].reshape(-1)[0] for i in indices], dtype=np.float64)
            size = self.cfg["window_size"]
            step = max(
                1,
                round(size * self.cfg["fps"] / self.cfg["advantage"].get("action_fps", 100)),
            )
            next_values = values[np.minimum(np.arange(len(values)) + step, len(values) - 1)]
            chunks = np.asarray(
                [rules.build_reward_chunk(rewards, i, prefix_end, size) for i in indices],
                dtype=np.float64,
            )
            terminates = np.asarray(
                [
                    rules.terminate_for_action_chunk_or_sparse_reward(i, prefix_end, size, sparse_index)[0]
                    for i in indices
                ]
            )
            gamma = self.cfg["advantage"]["gamma"]
            raw = gae(
                all_pred_values=values,
                all_next_obs_values=next_values,
                all_data_sources=np.ones(len(values)),
                all_reward_sums=chunks @ np.array([gamma**t for t in range(size)], dtype=np.float64),
                all_terminates=terminates,
                ep_starts=[0],
                ep_ends=[len(values)],
                advantage_obs_horizon=self.cfg["advantage"]["horizon"],
                gamma=gamma,
                gae_lambda=self.cfg["advantage"]["gae_lambda"],
                action_horizon=size,
                action_step_frames=step,
            )
        # Compute action GAE on the dense observations before selecting output
        # rows. Use the same source-action-index rule and terminal retention as
        # value sampling, without changing action chunks or bootstrap offsets.
        selected = set(
            rules.downsampled_action_indices_with_terminal(
                indices,
                end_idx=prefix_end,
                frame_stride=self.cfg[f"{role}_stride"],
            )
        )
        output_position = 0
        for position, index in enumerate(indices):
            if index not in selected:
                continue
            frame, sources = self._frame(index, prefix_end, rewards, sparse_index)
            if raw is not None:
                frame["advantage_raw"] = np.asarray([raw[position]], dtype=np.float64)
            self.audit[role].append(
                {
                    "local_episode_index": episode_index,
                    "frame_index": output_position,
                    "source_action_index": index,
                    "next_source_action_index": min(index + self.cfg["window_size"], prefix_end),
                    "prefix_end_index": prefix_end,
                    "sources": sources,
                }
            )
            output_position += 1
            yield frame
