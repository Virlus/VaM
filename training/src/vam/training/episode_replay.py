"""Stateful newest-first, round-robin traversal for growing episode buffers."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import hashlib
import random
import threading

import vam.training.config as _config


@dataclasses.dataclass(frozen=True, order=True)
class EpisodeKey:
    repo_id: str
    episode_index: int


@dataclasses.dataclass(frozen=True)
class FrameRef:
    frame_index: int
    sample_index: int


@dataclasses.dataclass
class _EpisodeCursor:
    frame_ids: tuple[int, ...]
    order: tuple[int, ...]
    cursor: int = 0


@dataclasses.dataclass
class _ReplayPool:
    config: _config.EpisodeReplayConfig
    pass_index: int = 0
    episodes: dict[EpisodeKey, _EpisodeCursor] = dataclasses.field(default_factory=dict)
    queue: deque[EpisodeKey] = dataclasses.field(default_factory=deque)
    new_queue: deque[EpisodeKey] = dataclasses.field(default_factory=deque)
    completed: set[EpisodeKey] = dataclasses.field(default_factory=set)
    total_samples: int = 0
    last_turn_was_new: bool = False


@dataclasses.dataclass(frozen=True)
class EpisodeReplayStats:
    episode_count: int
    completed_episode_count: int
    pass_index: int
    total_samples: int
    newest_episode_index: int | None
    oldest_episode_index: int | None


class EpisodeReplayState:
    """Mutable traversal progress shared by samplers belonging to one async stream.

    A snapshot is passed to a background dataloader rebuild. This avoids cursor races with the old loader while
    preserving progress across view updates. Semantic keys keep the state valid when concat-dataset offsets change.
    """

    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)
        self._pools: dict[str, _ReplayPool] = {}
        self._lock = threading.Lock()

    def snapshot(self) -> EpisodeReplayState:
        clone = EpisodeReplayState(seed=self.seed)
        with self._lock:
            clone._pools = copy.deepcopy(self._pools)
        return clone

    @staticmethod
    def _newest_first(keys: Sequence[EpisodeKey]) -> list[EpisodeKey]:
        return sorted(keys, key=lambda key: (key.episode_index, key.repo_id), reverse=True)

    def _permutation(
        self,
        *,
        pool_key: str,
        episode_key: EpisodeKey,
        frame_ids: tuple[int, ...],
        pass_index: int,
        shuffle: bool,
    ) -> tuple[int, ...]:
        order = list(frame_ids)
        if shuffle and len(order) > 1:
            payload = (
                f"{self.seed}\0{pool_key}\0{episode_key.repo_id}\0{episode_key.episode_index}\0{pass_index}"
            ).encode()
            permutation_seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
            random.Random(permutation_seed).shuffle(order)
        return tuple(order)

    def _new_cursor(
        self,
        *,
        pool_key: str,
        episode_key: EpisodeKey,
        frame_ids: tuple[int, ...],
        pass_index: int,
        shuffle: bool,
    ) -> _EpisodeCursor:
        return _EpisodeCursor(
            frame_ids=frame_ids,
            order=self._permutation(
                pool_key=pool_key,
                episode_key=episode_key,
                frame_ids=frame_ids,
                pass_index=pass_index,
                shuffle=shuffle,
            ),
        )

    @staticmethod
    def _normalize_candidates(
        candidates: Mapping[EpisodeKey, Sequence[FrameRef]],
    ) -> dict[EpisodeKey, tuple[FrameRef, ...]]:
        normalized = {}
        for episode_key, refs in candidates.items():
            ordered = tuple(sorted(refs, key=lambda ref: ref.frame_index))
            if not ordered:
                continue
            frame_ids = [ref.frame_index for ref in ordered]
            if len(frame_ids) != len(set(frame_ids)):
                raise ValueError(f"Episode replay found duplicate frame_index values for {episode_key}.")
            normalized[episode_key] = ordered
        if not normalized:
            raise ValueError("Episode replay requires at least one non-empty episode.")
        return normalized

    def _refresh_pool(
        self,
        *,
        pool_key: str,
        pool: _ReplayPool,
        candidates: Mapping[EpisodeKey, tuple[FrameRef, ...]],
    ) -> None:
        candidate_keys = set(candidates)
        pool_was_empty = not pool.episodes
        removed = set(pool.episodes) - candidate_keys
        if removed:
            for episode_key in removed:
                del pool.episodes[episode_key]
            pool.completed.difference_update(removed)
            pool.queue = deque(key for key in pool.queue if key not in removed)
            pool.new_queue = deque(key for key in pool.new_queue if key not in removed)

        added = []
        for episode_key, refs in candidates.items():
            frame_ids = tuple(ref.frame_index for ref in refs)
            cursor = pool.episodes.get(episode_key)
            if cursor is None or cursor.frame_ids != frame_ids:
                pool.episodes[episode_key] = self._new_cursor(
                    pool_key=pool_key,
                    episode_key=episode_key,
                    frame_ids=frame_ids,
                    pass_index=pool.pass_index,
                    shuffle=pool.config.shuffle_frames,
                )
                pool.completed.discard(episode_key)
                pool.queue = deque(key for key in pool.queue if key != episode_key)
                pool.new_queue = deque(key for key in pool.new_queue if key != episode_key)
                added.append(episode_key)

        if pool_was_empty:
            pool.queue.extend(self._newest_first(added))
        elif added:
            # A new episode gets the next available freshness turn. If an even newer view supersedes an episode
            # before that turn, move the older pending episode into the normal FIFO backlog so continuous arrivals
            # cannot starve episodes that were already waiting.
            pool.queue.extend(pool.new_queue)
            pool.new_queue.clear()
            pool.new_queue.extend(self._newest_first(added))

    def _start_next_pass(self, *, pool_key: str, pool: _ReplayPool) -> None:
        pool.pass_index += 1
        pool.completed.clear()
        for episode_key, cursor in pool.episodes.items():
            pool.episodes[episode_key] = self._new_cursor(
                pool_key=pool_key,
                episode_key=episode_key,
                frame_ids=cursor.frame_ids,
                pass_index=pool.pass_index,
                shuffle=pool.config.shuffle_frames,
            )
        pool.queue = deque(self._newest_first(tuple(pool.episodes)))
        pool.new_queue.clear()
        pool.last_turn_was_new = False

    def sample(
        self,
        *,
        pool_key: str,
        candidates: Mapping[EpisodeKey, Sequence[FrameRef]],
        count: int,
        config: _config.EpisodeReplayConfig,
    ) -> list[int]:
        """Return dataset indices while advancing the traversal state."""
        if count < 0:
            raise ValueError(f"Episode replay count must be non-negative, got {count}.")
        if count == 0:
            return []
        if config.mode != "newest_first_round_robin":
            raise ValueError(f"Unsupported episode replay mode: {config.mode!r}.")
        if config.frames_per_turn < 1:
            raise ValueError(f"Episode replay frames_per_turn must be >= 1, got {config.frames_per_turn}.")

        normalized = self._normalize_candidates(candidates)
        refs_by_episode = {
            episode_key: {ref.frame_index: ref.sample_index for ref in refs} for episode_key, refs in normalized.items()
        }
        with self._lock:
            pool = self._pools.get(pool_key)
            if pool is None:
                pool = _ReplayPool(config=config)
                self._pools[pool_key] = pool
            elif pool.config != config:
                raise ValueError(
                    f"Episode replay pool {pool_key!r} was reused with a different config: "
                    f"{pool.config!r} != {config!r}."
                )
            self._refresh_pool(pool_key=pool_key, pool=pool, candidates=normalized)

            sampled = []
            while len(sampled) < count:
                if not pool.queue and not pool.new_queue:
                    self._start_next_pass(pool_key=pool_key, pool=pool)
                use_new_queue = bool(pool.new_queue) and (not pool.queue or not pool.last_turn_was_new)
                episode_key = pool.new_queue.popleft() if use_new_queue else pool.queue.popleft()
                cursor = pool.episodes[episode_key]
                remaining = len(cursor.order) - cursor.cursor
                take = min(config.frames_per_turn, remaining, count - len(sampled))
                selected_frame_ids = cursor.order[cursor.cursor : cursor.cursor + take]
                sampled.extend(refs_by_episode[episode_key][frame_id] for frame_id in selected_frame_ids)
                cursor.cursor += take
                if cursor.cursor == len(cursor.order):
                    pool.completed.add(episode_key)
                else:
                    pool.queue.append(episode_key)
                pool.last_turn_was_new = use_new_queue
            pool.total_samples += len(sampled)
            return sampled

    def stats(self, pool_key: str) -> EpisodeReplayStats | None:
        with self._lock:
            pool = self._pools.get(pool_key)
            if pool is None:
                return None
            episode_indices = [key.episode_index for key in pool.episodes]
            return EpisodeReplayStats(
                episode_count=len(pool.episodes),
                completed_episode_count=len(pool.completed),
                pass_index=pool.pass_index,
                total_samples=pool.total_samples,
                newest_episode_index=max(episode_indices, default=None),
                oldest_episode_index=min(episode_indices, default=None),
            )
