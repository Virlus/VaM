from __future__ import annotations

import concurrent.futures
import queue

import datasets as hf_datasets

from vam.training import async_control
import vam.training.config as _config
from vam.training.data_loader import DualSourceDataLoader
from vam.training.data_loader import LiveCurrentRepoDataset
from vam.training.data_loader import WeightedColumnGroupSampler
from vam.training.episode_replay import EpisodeKey
from vam.training.episode_replay import EpisodeReplayState
from vam.training.episode_replay import FrameRef


def _candidates(lengths: dict[int, int], *, repo_id: str = "repo"):
    return {
        EpisodeKey(repo_id, episode_index): [
            FrameRef(frame_index=frame_index, sample_index=episode_index * 100 + frame_index)
            for frame_index in range(length)
        ]
        for episode_index, length in lengths.items()
    }


def test_newest_first_round_robin_traverses_every_frame_once() -> None:
    state = EpisodeReplayState(seed=7)
    config = _config.EpisodeReplayConfig(shuffle_frames=False)

    sampled = state.sample(
        pool_key="stream:on_policy:policy=0",
        candidates=_candidates({0: 3, 1: 1, 2: 2}),
        count=6,
        config=config,
    )

    assert sampled == [200, 100, 0, 201, 1, 2]
    assert len(set(sampled)) == 6
    stats = state.stats("stream:on_policy:policy=0")
    assert stats is not None
    assert stats.completed_episode_count == 3
    assert stats.pass_index == 0


def test_new_episode_moves_to_front_without_resetting_existing_cursors() -> None:
    state = EpisodeReplayState(seed=7)
    config = _config.EpisodeReplayConfig(shuffle_frames=False)
    pool_key = "stream:on_policy:policy=0"
    initial = _candidates({0: 3, 1: 3})

    assert state.sample(pool_key=pool_key, candidates=initial, count=2, config=config) == [100, 0]
    grown = {**initial, **_candidates({2: 2})}

    assert state.sample(pool_key=pool_key, candidates=grown, count=4, config=config) == [200, 101, 1, 201]


def test_multiple_new_episodes_alternate_with_existing_backlog_in_newest_order() -> None:
    state = EpisodeReplayState(seed=7)
    config = _config.EpisodeReplayConfig(shuffle_frames=False)
    pool_key = "stream:on_policy:policy=0"
    initial = _candidates({0: 2, 1: 2})
    assert state.sample(pool_key=pool_key, candidates=initial, count=1, config=config) == [100]
    grown = {**initial, **_candidates({2: 1, 3: 1})}

    assert state.sample(pool_key=pool_key, candidates=grown, count=4, config=config) == [300, 0, 200, 101]


def test_continuous_new_episodes_cannot_starve_existing_backlog() -> None:
    state = EpisodeReplayState(seed=7)
    config = _config.EpisodeReplayConfig(shuffle_frames=False)
    pool_key = "stream:on_policy:policy=0"
    candidates = _candidates({0: 4, 1: 4})
    assert state.sample(pool_key=pool_key, candidates=candidates, count=1, config=config) == [100]

    sampled = []
    for episode_index in range(2, 8):
        candidates = {**candidates, **_candidates({episode_index: 1})}
        sampled.extend(state.sample(pool_key=pool_key, candidates=candidates, count=1, config=config))

    # New arrivals get the first available turn, but every second turn is reserved for the existing queue.
    assert sampled == [200, 0, 400, 101, 600, 1]
    for episode_index in range(8, 30):
        candidates = {**candidates, **_candidates({episode_index: 1})}
        sampled.extend(state.sample(pool_key=pool_key, candidates=candidates, count=1, config=config))
    assert {0, 1, 2, 3, 100, 101, 102, 103}.issubset({100, *sampled})


def test_snapshot_preserves_progress_without_sharing_mutations() -> None:
    state = EpisodeReplayState(seed=3)
    config = _config.EpisodeReplayConfig(shuffle_frames=False)
    pool_key = "stream:on_policy:policy=0"
    candidates = _candidates({0: 2, 1: 2})
    assert state.sample(pool_key=pool_key, candidates=candidates, count=1, config=config) == [100]

    snapshot = state.snapshot()
    assert state.sample(pool_key=pool_key, candidates=candidates, count=1, config=config) == [0]
    assert snapshot.sample(pool_key=pool_key, candidates=candidates, count=1, config=config) == [0]

    assert state.sample(pool_key=pool_key, candidates=candidates, count=1, config=config) == [101]
    assert snapshot.sample(pool_key=pool_key, candidates=candidates, count=1, config=config) == [101]


def test_removed_episode_is_not_sampled_after_view_shrinks() -> None:
    state = EpisodeReplayState(seed=3)
    config = _config.EpisodeReplayConfig(shuffle_frames=False)
    pool_key = "stream:on_policy:policy=0"
    initial = _candidates({0: 3, 1: 3})
    assert state.sample(pool_key=pool_key, candidates=initial, count=2, config=config) == [100, 0]

    remaining = _candidates({1: 3})
    assert state.sample(pool_key=pool_key, candidates=remaining, count=2, config=config) == [101, 102]


def test_frame_shuffle_is_deterministic_and_without_replacement_per_pass() -> None:
    config = _config.EpisodeReplayConfig(shuffle_frames=True)
    candidates = _candidates({4: 8})
    first = EpisodeReplayState(seed=11).sample(pool_key="pool", candidates=candidates, count=8, config=config)
    second = EpisodeReplayState(seed=11).sample(pool_key="pool", candidates=candidates, count=8, config=config)

    assert first == second
    assert sorted(first) == list(range(400, 408))
    assert first != sorted(first)


def test_concurrent_prefetch_reservations_do_not_duplicate_frames_within_a_pass() -> None:
    state = EpisodeReplayState(seed=11)
    config = _config.EpisodeReplayConfig(shuffle_frames=True)
    candidates = _candidates({episode_index: 10 for episode_index in range(8)})

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(state.sample, pool_key="pool", candidates=candidates, count=10, config=config)
            for _ in range(8)
        ]
    sampled = [sample_index for future in futures for sample_index in future.result()]

    assert len(sampled) == 80
    assert len(set(sampled)) == 80


class _SamplerDataset:
    repo_id = "repo"

    def __init__(self, *, include_new_episode: bool = False) -> None:
        columns = {
            "data_source": [0, 0, 1, 1, 1, 1],
            "policy_id": [-1, -1, 0, 0, 0, 0],
            "episode_index": [10, 11, 0, 0, 1, 1],
            "frame_index": [0, 0, 0, 1, 0, 1],
        }
        if include_new_episode:
            columns["data_source"].extend([1, 1])
            columns["policy_id"].extend([0, 0])
            columns["episode_index"].extend([2, 2])
            columns["frame_index"].extend([0, 1])
        self.hf_dataset = hf_datasets.Dataset.from_dict(columns)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int):
        return self.hf_dataset[index]


def test_weighted_sampler_keeps_group_quota_and_uses_episode_replay_within_on_policy() -> None:
    config = _weighted_sampler_config()
    sampler = WeightedColumnGroupSampler(_SamplerDataset(), batch_size=4, sampler_config=config, seed=0)
    iterator = iter(sampler)
    first_batch = [next(iterator) for _ in range(4)]
    second_batch = [next(iterator) for _ in range(4)]

    assert len([index for index in first_batch if index < 2]) == 2
    assert len([index for index in first_batch if index >= 2]) == 2
    assert set(index for index in first_batch if index >= 2) == {2, 4}
    assert set(index for index in second_batch if index >= 2) == {3, 5}


def _weighted_sampler_config() -> _config.SamplerConfig:
    return _config.SamplerConfig(
        groups=(
            _config.SamplerGroupConfig("offline", 1.0, {"data_source": 0}),
            _config.SamplerGroupConfig(
                "on_policy",
                1.0,
                {"policy_id_eq_current": True},
                episode_replay=_config.EpisodeReplayConfig(shuffle_frames=False),
            ),
        ),
        current_policy_id=0,
    )


def test_weighted_sampler_rebuild_maps_snapshot_progress_into_grown_view() -> None:
    state = EpisodeReplayState(seed=0)
    initial_sampler = WeightedColumnGroupSampler(
        _SamplerDataset(),
        batch_size=4,
        sampler_config=_weighted_sampler_config(),
        seed=0,
        episode_replay_state=state,
        episode_replay_scope="stream",
    )
    initial_iterator = iter(initial_sampler)
    initial_batch = [next(initial_iterator) for _ in range(4)]
    assert set(index for index in initial_batch if index >= 2) == {2, 4}

    rebuilt_sampler = WeightedColumnGroupSampler(
        _SamplerDataset(include_new_episode=True),
        batch_size=4,
        sampler_config=_weighted_sampler_config(),
        seed=0,
        episode_replay_state=state.snapshot(),
        episode_replay_scope="stream",
    )
    rebuilt_iterator = iter(rebuilt_sampler)
    rebuilt_batch = [next(rebuilt_iterator) for _ in range(4)]

    assert set(index for index in rebuilt_batch if index >= 2) == {5, 6}


class _LiveBackingDataset:
    def __init__(self) -> None:
        self.hf_dataset = hf_datasets.Dataset.from_dict(
            {
                "episode_index": [0, 0, 1, 1],
                "frame_index": [0, 1, 0, 1],
            }
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int):
        row = self.hf_dataset[index]
        return {"sample_index": index, **row}


def test_dual_source_live_rows_use_the_episode_replay_scheduler() -> None:
    live = LiveCurrentRepoDataset.__new__(LiveCurrentRepoDataset)
    live.repo_id = "repo"
    live._dataset = _LiveBackingDataset()  # noqa: SLF001
    loader = DualSourceDataLoader(
        stable_loader=None,
        live_dataset=live,
        local_batch_size=2,
        current_batch=2,
        sharding=object(),
        seed=5,
        episode_replay_config=_config.EpisodeReplayConfig(shuffle_frames=False),
        episode_replay_state=EpisodeReplayState(seed=5),
        episode_replay_pool_key="stream:on_policy:policy=0",
    )

    first = loader._sample_live_rows()  # noqa: SLF001
    second = loader._sample_live_rows()  # noqa: SLF001

    assert [row["sample_index"] for row in first] == [2, 0]
    assert [row["sample_index"] for row in second] == [3, 1]


def test_dual_source_discards_prefetch_from_before_live_grow() -> None:
    live = LiveCurrentRepoDataset.__new__(LiveCurrentRepoDataset)
    live.repo_id = "repo"
    live._dataset = _LiveBackingDataset()  # noqa: SLF001
    loader = DualSourceDataLoader(
        stable_loader=None,
        live_dataset=live,
        local_batch_size=1,
        current_batch=1,
        sharding=object(),
        seed=5,
    )
    live_queue = queue.Queue()
    live_queue.put((-1, [{"sample_index": 0}]))
    live_queue.put((0, [{"sample_index": 1}]))

    assert loader._take_current_live_rows(live_queue) == [{"sample_index": 1}]  # noqa: SLF001
    assert live_queue.empty()


def test_v1_dataset_view_keeps_config_owned_episode_replay() -> None:
    replay = _config.EpisodeReplayConfig(frames_per_turn=2)
    base_config = _config.DataConfig(
        sampler=_config.SamplerConfig(
            groups=(
                _config.SamplerGroupConfig(
                    "on_policy",
                    1.0,
                    {"policy_id_eq_current": True},
                    episode_replay=replay,
                ),
            ),
        )
    )
    view = async_control.DatasetView(
        view_id="view",
        repo_specs=(_config.RepoSpec("repo"),),
        sampler=_config.SamplerConfig(
            groups=(_config.SamplerGroupConfig("on_policy", 3.0, {"policy_id_eq_current": True}),),
        ),
        target_policy_id=2,
    )

    applied = async_control.apply_dataset_view(base_config, view)

    assert applied.sampler is not None
    assert applied.sampler.groups[0].weight == 3.0
    assert applied.sampler.groups[0].episode_replay == replay
    assert applied.sampler.current_policy_id == 2


def test_active_aloha_async_samplers_enable_replay_only_for_current_policy_groups() -> None:
    action_groups = _config.ALOHA_ASYNC_ACTION_SAMPLER.groups
    value_groups = _config.ALOHA_ASYNC_VALUE_SAMPLER.groups

    assert [group.name for group in action_groups if group.episode_replay is not None] == [
        "on_policy_rollout_or_dagger"
    ]
    assert [group.name for group in value_groups if group.episode_replay is not None] == ["on_policy_trajectory"]
    assert all(group.episode_replay is None for group in action_groups[:2])
    assert value_groups[0].episode_replay is None
